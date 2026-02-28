from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import side effect: registers all built-in adapters.
import baseline.adapters  # noqa: F401

from .core.config import ExperimentConfig, validate_experiment_config
from .core.registry import (
    get_dataset_adapter,
    get_logger_adapter,
    get_model_adapter,
    get_split_adapter,
    get_tokenizer_adapter,
)
from .core.types import RunResult, TokenizedCorpus


class LMWindowDataset(Dataset):
    """Fixed-window LM dataset with optional tail padding and key padding mask."""

    def __init__(self, tokens: list[int], seq_len: int, stride: int, pad_id: int):
        if not tokens:
            raise ValueError("Token stream is empty after preprocessing.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        self.tokens = tokens
        self.seq_len = seq_len
        self.window = seq_len + 1
        self.stride = stride
        self.pad_id = pad_id
        self.starts = self._build_starts()

    def _build_starts(self) -> list[int]:
        token_count = len(self.tokens)
        if token_count <= self.window:
            return [0]

        full_limit = token_count - self.window + 1
        starts = list(range(0, full_limit, self.stride))
        if not starts:
            starts = [0]

        next_start = starts[-1] + self.stride
        if next_start < token_count:
            starts.append(next_start)
        return starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        sample = self.tokens[start : start + self.window]
        if len(sample) < self.window:
            sample = sample + [self.pad_id] * (self.window - len(sample))

        sample_tensor = torch.tensor(sample, dtype=torch.long)
        input_seq = sample_tensor[:-1]
        target_seq = sample_tensor[1:]
        key_padding_mask = input_seq != self.pad_id
        return input_seq, target_seq, key_padding_mask


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_optimizer_state_to_device(
    optimizer: optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_uncompiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def maybe_compile_model(
    model: torch.nn.Module, device: torch.device, config: ExperimentConfig
) -> tuple[torch.nn.Module, bool, str]:
    if not config.run.use_torch_compile:
        return model, False, "disabled"
    if not hasattr(torch, "compile"):
        return model, False, "torch.compile unavailable"
    if device.type != "cuda":
        return model, False, f"skipped on {device.type}"

    try:
        compiled_model = torch.compile(
            model,
            mode=config.run.torch_compile_mode,
            fullgraph=bool(config.run.torch_compile_fullgraph),
            dynamic=bool(config.run.torch_compile_dynamic),
        )
        return compiled_model, True, "enabled"
    except Exception as exc:  # pragma: no cover - backend-specific failure paths.
        return model, False, f"failed: {exc}"


def should_enable_bf16_autocast(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker):
        return False
    return bool(checker())


def get_autocast_context(device: torch.device, use_bf16: bool):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def truncate_stream_by_fraction_at_eos(
    token_stream: list[int], data_fraction: float, eos_id: int
) -> list[int]:
    if not 0 < data_fraction <= 1:
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")
    if data_fraction >= 1:
        return token_stream

    target_len = max(1, int(len(token_stream) * data_fraction))
    if target_len >= len(token_stream):
        return token_stream

    prefix = token_stream[:target_len]
    if eos_id in prefix:
        cutoff = max(i for i, token in enumerate(prefix) if token == eos_id) + 1
        return token_stream[:cutoff]

    for idx in range(target_len, len(token_stream)):
        if token_stream[idx] == eos_id:
            return token_stream[: idx + 1]
    return token_stream


def find_latest_artifact_dir_with_checkpoint(
    models_root: Path,
    checkpoint_filename: str,
) -> Path | None:
    latest_dir: Path | None = None
    latest_mtime = float("-inf")

    if not models_root.exists():
        return None

    for entry in models_root.iterdir():
        if not entry.is_dir():
            continue
        checkpoint_path = entry / checkpoint_filename
        if not checkpoint_path.exists():
            continue

        checkpoint_mtime = checkpoint_path.stat().st_mtime
        if checkpoint_mtime > latest_mtime:
            latest_mtime = checkpoint_mtime
            latest_dir = entry

    return latest_dir


def prepare_run_artifact_paths(config: ExperimentConfig) -> dict[str, Path]:
    models_root = Path(config.run.artifacts_root).expanduser().resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    if config.run.run_name:
        run_dir = models_root / config.run.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    elif config.run.resume_from_checkpoint:
        run_dir = find_latest_artifact_dir_with_checkpoint(
            models_root=models_root,
            checkpoint_filename=config.run.checkpoint_filename,
        )
        if run_dir is None:
            run_dir = models_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Resuming artifacts from: {run_dir}")
    else:
        run_dir = models_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_artifact_dir": run_dir,
        "checkpoint_path": run_dir / config.run.checkpoint_filename,
        "final_model_path": run_dir / config.run.final_model_filename,
        "model_diagram_path": run_dir / "baseline_model_architecture",
        "run_config_path": run_dir / "run_config.json",
        "inference_config_path": run_dir / "inference_config.json",
    }
    print(f"Run artifacts will be saved to: {run_dir}")
    return paths


def write_run_metadata(
    config: ExperimentConfig,
    tokenized: TokenizedCorpus,
    run_paths: dict[str, Path],
) -> None:
    run_paths["run_config_path"].write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    special = tokenized.vocab.special
    inference_config = {
        "model_name": config.model.name,
        "tokenizer_name": config.tokenizer.name,
        "base_vocab_size": special.base_vocab_size,
        "num_special_tokens": special.num_special_tokens,
        "vocab_size": special.vocab_size,
        "d_model": config.model.d_model,
        "n_heads": config.model.n_heads,
        "layers": config.model.layers,
        "training_seq_len": config.train.seq_len,
        "tokenizer_vocab_path": str(Path(config.tokenizer.vocab_path).expanduser().resolve()),
    }
    run_paths["inference_config_path"].write_text(
        json.dumps(inference_config, indent=2),
        encoding="utf-8",
    )


def build_data_loaders(
    config: ExperimentConfig,
    tokenized: TokenizedCorpus,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    special = tokenized.vocab.special

    token_stream = truncate_stream_by_fraction_at_eos(
        token_stream=tokenized.token_stream,
        data_fraction=config.train.data_fraction,
        eos_id=special.eos_id,
    )

    retained_eos = sum(1 for token in token_stream if token == special.eos_id)
    print(
        f"Encoded stream tokens: {len(token_stream):,} | "
        f"EOS inserted: {tokenized.eos_inserted:,} | EOS retained: {retained_eos:,} | "
        f"UNK replacements: {tokenized.unk_replacements:,}"
    )

    if any(token < 0 or token >= tokenized.vocab.vocab_size for token in token_stream):
        raise ValueError("Token stream contains token ids outside model vocab range.")

    dataset = LMWindowDataset(
        token_stream,
        seq_len=config.train.seq_len,
        stride=config.train.stride,
        pad_id=special.pad_id,
    )

    print(
        f"Dataset samples: {len(dataset):,} | "
        f"seq_len={config.train.seq_len} | stride={config.train.stride} | pad_id={special.pad_id}"
    )

    split_adapter = get_split_adapter(config.split.name)
    train_set, val_set = split_adapter.split(dataset=dataset, cfg=config.split)

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CrossEntropyLoss,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    pad_id = int(loss_fn.ignore_index)
    total_tokens = 0
    total_loss = 0.0
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for input_seq, target_seq, key_padding_mask in loader:
            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)
            key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

            with get_autocast_context(device=device, use_bf16=use_bf16):
                output = model(input_seq, key_padding_mask=key_padding_mask)
                loss_sum = loss_fn(
                    output.reshape(-1, output.size(-1)),
                    target_seq.reshape(-1),
                )

            valid_target_mask = target_seq != pad_id
            tokens = int(valid_target_mask.sum().item())
            if tokens == 0:
                continue

            total_tokens += tokens
            total_loss += loss_sum.item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }


class ForwardMetricCollector:
    def __init__(self) -> None:
        self.capture_activation_norms = False
        self.capture_attention_entropy = False
        self.activation_norms: dict[str, torch.Tensor] = {}
        self.attention_entropy: dict[str, torch.Tensor] = {}

    def begin_step(
        self,
        *,
        capture_activation_norms: bool,
        capture_attention_entropy: bool,
    ) -> None:
        self.capture_activation_norms = capture_activation_norms
        self.capture_attention_entropy = capture_attention_entropy
        self.activation_norms.clear()
        self.attention_entropy.clear()

    def take_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in self.activation_norms.items():
            metrics[key] = float(value.item())
        for key, value in self.attention_entropy.items():
            metrics[key] = float(value.item())
        self.activation_norms.clear()
        self.attention_entropy.clear()
        return metrics


def get_decoder_layer_labels(model: torch.nn.Module) -> dict[int, list[str]]:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    layer_count = len(dec_layers)
    if layer_count <= 0:
        return {}

    selected = [
        ("first", 0),
        ("middle", layer_count // 2),
        ("last", layer_count - 1),
    ]

    layer_labels: dict[int, list[str]] = {}
    for label, idx in selected:
        layer_labels.setdefault(idx, []).append(label)
    return layer_labels


def register_forward_metric_hooks(
    model: torch.nn.Module,
    collector: ForwardMetricCollector,
    layer_labels: dict[int, list[str]],
    attention_head_cap: int,
    attention_token_cap: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return handles

    for layer_idx, labels in layer_labels.items():
        layer = dec_layers[layer_idx]
        label_tuple = tuple(labels)

        def activation_hook(_module, _inputs, output, label_tuple=label_tuple):
            if not collector.capture_activation_norms:
                return
            if not torch.is_tensor(output):
                return
            activation_norm = output.detach().float().pow(2).mean().sqrt()
            for label in label_tuple:
                collector.activation_norms[f"activation_norm_{label}"] = activation_norm

        handles.append(layer.register_forward_hook(activation_hook))

        attn = getattr(layer, "multi_head_attention", None)
        softmax_module = getattr(attn, "softmax", None)
        if softmax_module is None:
            continue

        def attention_entropy_hook(_module, _inputs, output, label_tuple=label_tuple):
            if not collector.capture_attention_entropy:
                return
            if not torch.is_tensor(output) or output.dim() != 4:
                return

            _, heads, query_len, key_len = output.shape
            sampled_heads = min(attention_head_cap, heads)
            sampled_tokens = min(attention_token_cap, query_len, key_len)
            if sampled_heads <= 0 or sampled_tokens <= 0:
                return

            probs = output[:, :sampled_heads, :sampled_tokens, :sampled_tokens].detach().float()
            probs = probs.clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            for label in label_tuple:
                collector.attention_entropy[f"attention_entropy_{label}"] = entropy

        handles.append(softmax_module.register_forward_hook(attention_entropy_hook))

    return handles


def compute_global_grad_norm(model: torch.nn.Module) -> float | None:
    grad_norm_sq: torch.Tensor | None = None
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().pow(2).sum()
        grad_norm_sq = grad_sq if grad_norm_sq is None else grad_norm_sq + grad_sq

    if grad_norm_sq is None:
        return None
    return float(grad_norm_sq.sqrt().item())


def compute_layernorm_grad_norms(
    model: torch.nn.Module,
    layer_labels: dict[int, list[str]],
) -> dict[str, float]:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    metrics: dict[str, float] = {}
    for layer_idx, labels in layer_labels.items():
        layer = dec_layers[layer_idx]

        weight_sq: torch.Tensor | None = None
        for grad in (layer.ln1.gamma.grad, layer.ln2.gamma.grad):
            if grad is None:
                continue
            grad_sq = grad.detach().float().pow(2).sum()
            weight_sq = grad_sq if weight_sq is None else weight_sq + grad_sq

        bias_sq: torch.Tensor | None = None
        for grad in (layer.ln1.beta.grad, layer.ln2.beta.grad):
            if grad is None:
                continue
            grad_sq = grad.detach().float().pow(2).sum()
            bias_sq = grad_sq if bias_sq is None else bias_sq + grad_sq

        for label in labels:
            if weight_sq is not None:
                metrics[f"ln_weight_grad_norm_{label}"] = float(weight_sq.sqrt().item())
            if bias_sq is not None:
                metrics[f"ln_bias_grad_norm_{label}"] = float(bias_sq.sqrt().item())

    return metrics


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
    logger,
    device: torch.device,
    use_bf16: bool,
    run_paths: dict[str, Path],
) -> tuple[int, float, dict[str, float]]:
    checkpoint_model = get_uncompiled_model(model)
    pad_id = int(loss_fn.ignore_index)
    wandb_cfg = config.logging.wandb
    wandb_enabled = config.logging.provider == "wandb"
    has_decoder_layers = bool(getattr(checkpoint_model, "dec_layers", None))
    layer_labels = get_decoder_layer_labels(checkpoint_model) if has_decoder_layers else {}
    metric_collector = ForwardMetricCollector()
    tokens_seen_train = 0
    run_label = config.run.run_name or Path(run_paths["run_artifact_dir"]).name
    checkpoint_artifact_name = f"{run_label}-checkpoint"
    final_model_artifact_name = f"{run_label}-model"

    def should_log_every(step: int, every_n_steps: int) -> bool:
        return step > 0 and step % every_n_steps == 0

    def save_checkpoint(
        epoch: int,
        next_batch_idx: int,
        global_step: int,
        *,
        aliases: tuple[str, ...] = ("latest",),
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "batch_idx": next_batch_idx,
            "global_step": global_step,
            "tokens_seen_train": tokens_seen_train,
            "model_state_dict": checkpoint_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        }
        torch.save(checkpoint, run_paths["checkpoint_path"])
        logger.save(
            str(run_paths["checkpoint_path"]),
            artifact_name=checkpoint_artifact_name,
            artifact_type="checkpoint",
            aliases=aliases,
            metadata={
                "epoch": int(epoch),
                "batch_idx": int(next_batch_idx),
                "global_step": int(global_step),
                "tokens_seen_train": int(tokens_seen_train),
                "run_name": run_label,
                "group_name": config.run.group_name,
            },
        )

    def load_checkpoint_if_available() -> tuple[int, int, int, int]:
        if not config.run.resume_from_checkpoint:
            return 0, 0, 0, 0

        checkpoint_path = run_paths["checkpoint_path"]
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}, starting fresh.")
            return 0, 0, 0, 0

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = dict(checkpoint["model_state_dict"])
        model_state_dict.pop("pos_encoding.pos_enc_cache", None)
        checkpoint_model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)

        start_epoch = int(checkpoint.get("epoch", 0))
        start_batch_idx = int(checkpoint.get("batch_idx", 0))
        global_step = int(checkpoint.get("global_step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen_train", 0))
        print(
            f"Resumed from {checkpoint_path} "
            f"at epoch={start_epoch}, batch={start_batch_idx}, step={global_step}, "
            f"tokens_seen_train={tokens_seen}"
        )
        return start_epoch, start_batch_idx, global_step, tokens_seen

    model.train()
    non_blocking = device.type == "cuda"

    start_epoch, start_batch_idx, global_step, tokens_seen_train = load_checkpoint_if_available()

    last_avg_train_loss = 0.0
    last_val_metrics = {"val_loss": 0.0, "val_perplexity": 0.0}
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    if wandb_enabled and layer_labels and (
        wandb_cfg.enable_activation_norms or wandb_cfg.enable_attention_entropy
    ):
        hook_handles = register_forward_metric_hooks(
            model=checkpoint_model,
            collector=metric_collector,
            layer_labels=layer_labels,
            attention_head_cap=wandb_cfg.attention_entropy_head_cap,
            attention_token_cap=wandb_cfg.attention_entropy_token_cap,
        )

    periodic_val_enabled = (
        wandb_enabled
        and wandb_cfg.val_every_n_steps > 0
        and (wandb_cfg.enable_val_loss_vs_tokens or wandb_cfg.enable_perplexity)
    )

    try:
        for epoch in tqdm(range(start_epoch, config.train.epochs), desc="Epochs"):
            epoch_train_loss_sum = 0.0
            epoch_token_count = 0

            for batch_idx, (input_seq, target_seq, key_padding_mask) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue

                next_global_step = global_step + 1
                should_log_step_metrics = (
                    wandb_enabled and should_log_every(next_global_step, wandb_cfg.log_every_n_steps)
                )
                should_log_diagnostics = (
                    wandb_enabled
                    and should_log_every(next_global_step, wandb_cfg.diagnostics_every_n_steps)
                )
                should_log_attention_entropy = (
                    wandb_enabled
                    and wandb_cfg.enable_attention_entropy
                    and should_log_every(next_global_step, wandb_cfg.attention_entropy_every_n_steps)
                )
                capture_activation_norms = (
                    should_log_diagnostics
                    and wandb_cfg.enable_activation_norms
                    and bool(layer_labels)
                )
                capture_attention_entropy = should_log_attention_entropy and bool(layer_labels)
                should_log_this_step = (
                    should_log_step_metrics or should_log_diagnostics or should_log_attention_entropy
                )

                metric_collector.begin_step(
                    capture_activation_norms=capture_activation_norms,
                    capture_attention_entropy=capture_attention_entropy,
                )

                step_start_time: float | None = None
                if should_log_step_metrics and wandb_cfg.enable_step_time:
                    step_start_time = time.perf_counter()
                if (
                    should_log_step_metrics
                    and wandb_cfg.enable_peak_memory
                    and device.type == "cuda"
                ):
                    torch.cuda.reset_peak_memory_stats(device)

                input_seq = input_seq.to(device, non_blocking=non_blocking)
                target_seq = target_seq.to(device, non_blocking=non_blocking)
                key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

                optimizer.zero_grad()
                with get_autocast_context(device=device, use_bf16=use_bf16):
                    output = model(input_seq, key_padding_mask=key_padding_mask)
                    loss_sum = loss_fn(
                        output.reshape(-1, output.size(-1)),
                        target_seq.reshape(-1),
                    )
                valid_tokens = int((target_seq != pad_id).sum().item())
                if valid_tokens == 0:
                    continue

                loss = loss_sum / valid_tokens
                loss.backward()

                step_metrics: dict[str, float] = {}
                if should_log_diagnostics and wandb_cfg.enable_global_grad_norm:
                    global_grad_norm = compute_global_grad_norm(checkpoint_model)
                    if global_grad_norm is not None:
                        step_metrics["global_grad_norm"] = global_grad_norm
                if should_log_diagnostics and wandb_cfg.enable_ln_grad_norms:
                    step_metrics.update(
                        compute_layernorm_grad_norms(checkpoint_model, layer_labels)
                    )

                optimizer.step()
                global_step = next_global_step
                tokens_seen_train += valid_tokens

                epoch_token_count += valid_tokens
                epoch_train_loss_sum += loss_sum.item()

                if should_log_this_step:
                    step_loss = float(loss.item())
                    step_metrics["epoch"] = float(
                        epoch + (batch_idx + 1) / max(len(train_loader), 1)
                    )

                    if wandb_cfg.enable_train_loss_vs_tokens:
                        step_metrics["train_loss"] = step_loss
                        step_metrics["train_loss_step"] = step_loss
                        step_metrics["tokens_seen_train"] = float(tokens_seen_train)

                    if wandb_cfg.enable_perplexity:
                        bounded = min(step_loss, 60.0)
                        step_metrics["train_perplexity"] = float(math.exp(bounded))
                        step_metrics.setdefault("tokens_seen_train", float(tokens_seen_train))

                    if device.type == "cuda" and should_log_step_metrics and (
                        wandb_cfg.enable_step_time or wandb_cfg.enable_peak_memory
                    ):
                        torch.cuda.synchronize(device)

                    if (
                        should_log_step_metrics
                        and wandb_cfg.enable_step_time
                        and step_start_time is not None
                    ):
                        elapsed_ms = (time.perf_counter() - step_start_time) * 1000.0
                        step_metrics["step_time_ms"] = float(elapsed_ms)
                    if (
                        should_log_step_metrics
                        and wandb_cfg.enable_peak_memory
                        and device.type == "cuda"
                    ):
                        peak_mem_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
                        step_metrics["peak_memory_gib"] = float(peak_mem_gib)

                    if capture_activation_norms or capture_attention_entropy:
                        step_metrics.update(metric_collector.take_metrics())

                    logger.log(step_metrics, step=global_step)

                if (
                    periodic_val_enabled
                    and should_log_every(global_step, wandb_cfg.val_every_n_steps)
                ):
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        loss_fn,
                        device,
                        use_bf16=use_bf16,
                    )
                    model.train()
                    last_val_metrics = val_metrics

                    val_log_metrics = {
                        "epoch": float(epoch + (batch_idx + 1) / max(len(train_loader), 1)),
                    }
                    if wandb_cfg.enable_val_loss_vs_tokens:
                        val_log_metrics["val_loss"] = float(val_metrics["val_loss"])
                        val_log_metrics["tokens_seen_train"] = float(tokens_seen_train)
                    if wandb_cfg.enable_perplexity:
                        val_log_metrics["val_perplexity"] = float(val_metrics["val_perplexity"])
                        val_log_metrics.setdefault("tokens_seen_train", float(tokens_seen_train))
                    logger.log(val_log_metrics, step=global_step)

                if (
                    config.run.checkpoint_every_n_steps > 0
                    and global_step % config.run.checkpoint_every_n_steps == 0
                ):
                    next_epoch = epoch
                    next_batch_idx = batch_idx + 1
                    if next_batch_idx >= len(train_loader):
                        next_epoch += 1
                        next_batch_idx = 0

                    save_checkpoint(
                        next_epoch,
                        next_batch_idx,
                        global_step,
                        aliases=("latest", f"step-{global_step}"),
                    )

            avg_train_loss = epoch_train_loss_sum / max(epoch_token_count, 1)
            val_metrics = evaluate(model, val_loader, loss_fn, device, use_bf16=use_bf16)
            model.train()

            epoch_metrics: dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss_epoch": float(avg_train_loss),
            }
            if (not wandb_enabled) or wandb_cfg.enable_val_loss_vs_tokens:
                epoch_metrics["val_loss"] = float(val_metrics["val_loss"])
            if (not wandb_enabled) or wandb_cfg.enable_perplexity:
                epoch_metrics["val_perplexity"] = float(val_metrics["val_perplexity"])
            if wandb_enabled and wandb_cfg.enable_train_loss_vs_tokens:
                epoch_metrics["tokens_seen_train"] = float(tokens_seen_train)
            if wandb_enabled and wandb_cfg.enable_perplexity:
                bounded_epoch_loss = min(float(avg_train_loss), 60.0)
                epoch_metrics["train_perplexity_epoch"] = float(math.exp(bounded_epoch_loss))
                epoch_metrics.setdefault("tokens_seen_train", float(tokens_seen_train))

            logger.log(epoch_metrics, step=global_step)

            print(
                f"Epoch {epoch + 1}/{config.train.epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_perplexity={val_metrics['val_perplexity']:.4f}"
            )

            last_avg_train_loss = float(avg_train_loss)
            last_val_metrics = val_metrics
    finally:
        for handle in hook_handles:
            handle.remove()

    save_checkpoint(
        config.train.epochs,
        0,
        global_step,
        aliases=("latest", "final", f"step-{global_step}"),
    )

    torch.save(checkpoint_model.state_dict(), run_paths["final_model_path"])
    logger.save(
        str(run_paths["final_model_path"]),
        artifact_name=final_model_artifact_name,
        artifact_type="model",
        aliases=("latest", "final"),
        metadata={
            "global_step": int(global_step),
            "final_train_loss": float(last_avg_train_loss),
            "final_val_loss": float(last_val_metrics["val_loss"]),
            "final_val_perplexity": float(last_val_metrics["val_perplexity"]),
            "run_name": run_label,
            "group_name": config.run.group_name,
        },
    )
    return global_step, last_avg_train_loss, last_val_metrics


def model_pipeline(config: ExperimentConfig) -> RunResult:
    validate_experiment_config(config)
    set_seed(config.run.seed)
    print(f"Using seed: {config.run.seed}")

    device = get_best_device()
    print(f"Using device: {device}")
    use_bf16 = should_enable_bf16_autocast(device)
    print(f"bf16 autocast: {'enabled' if use_bf16 else 'disabled'}")

    run_paths = prepare_run_artifact_paths(config)

    dataset_adapter = get_dataset_adapter(config.dataset.name)
    corpus = dataset_adapter.load(config.dataset)

    tokenizer_adapter = get_tokenizer_adapter(config.tokenizer.name)
    tokenized = tokenizer_adapter.build(corpus=corpus, cfg=config.tokenizer)

    write_run_metadata(config=config, tokenized=tokenized, run_paths=run_paths)

    train_loader, val_loader = build_data_loaders(
        config=config,
        tokenized=tokenized,
        pin_memory=device.type == "cuda",
    )

    model_adapter = get_model_adapter(config.model.name)
    model = model_adapter.build(
        cfg=config.model,
        vocab=tokenized.vocab,
        special=tokenized.vocab.special,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    loss_fn = CrossEntropyLoss(ignore_index=tokenized.vocab.special.pad_id, reduction="sum")

    logger_adapter = get_logger_adapter(config.logging.provider)
    logger = logger_adapter.start(
        cfg=config.logging,
        project_name=config.run.project_name,
        run_name=config.run.run_name,
        group_name=config.run.group_name,
        config_payload=asdict(config),
    )

    try:
        logger.save(
            str(run_paths["run_config_path"]),
            artifact_name=f"{config.run.run_name or run_paths['run_artifact_dir'].name}-run-config",
            artifact_type="metadata",
            aliases=("latest",),
            metadata={
                "run_name": config.run.run_name,
                "group_name": config.run.group_name,
            },
        )
        logger.save(
            str(run_paths["inference_config_path"]),
            artifact_name=(
                f"{config.run.run_name or run_paths['run_artifact_dir'].name}-"
                "inference-config"
            ),
            artifact_type="metadata",
            aliases=("latest",),
            metadata={
                "run_name": config.run.run_name,
                "group_name": config.run.group_name,
            },
        )
        model, compile_enabled, compile_status = maybe_compile_model(model, device, config)
        logger.log(
            {
                "torch_compile_enabled": float(1 if compile_enabled else 0),
                "bf16_autocast_enabled": float(1 if use_bf16 else 0),
            },
            step=0,
        )
        print(f"torch.compile: {compile_status}")

        if config.logging.provider == "wandb" and config.logging.wandb.watch_model:
            logger.watch(get_uncompiled_model(model), loss_fn)

        global_step, final_train_loss, final_val_metrics = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            logger=logger,
            device=device,
            use_bf16=use_bf16,
            run_paths=run_paths,
        )
    finally:
        logger.close()

    return RunResult(
        model=model,
        device=device,
        run_artifact_dir=str(run_paths["run_artifact_dir"]),
        checkpoint_path=str(run_paths["checkpoint_path"]),
        final_model_path=str(run_paths["final_model_path"]),
        global_step=global_step,
        final_train_loss=final_train_loss,
        final_val_loss=float(final_val_metrics["val_loss"]),
        final_val_perplexity=float(final_val_metrics["val_perplexity"]),
    )
