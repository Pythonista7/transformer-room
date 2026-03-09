from __future__ import annotations

import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .adapters import register_builtin_adapters
from .adapters.loggers import sanitize_wandb_name
from .core.config import ExperimentConfig, validate_experiment_config
from .core.registry import (
    get_dataset_adapter,
    get_logger_adapter,
    get_model_adapter,
    get_split_adapter,
    get_tokenizer_adapter,
)
from .core.types import RunResult, TokenizedCorpus
from .training.metrics import (
    EpochMetricsContext,
    MetricsEngine,
    PeriodicValMetricsContext,
    StepMetricsContext,
    build_default_metric_plugins,
    build_metric_schedule,
    get_decoder_layer_labels,
)

if TYPE_CHECKING:
    from .training.metrics import MetricPlugin


@dataclass(slots=True)
class TrainLoopResult:
    global_step: int
    final_train_loss: float
    final_val_metrics: dict[str, float]
    checkpoint_artifact_ref: str | None
    final_model_artifact_ref: str | None


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


def _set_activation_memory_budget_if_configured(config: ExperimentConfig) -> None:
    budget = config.run.activation_memory_budget
    if budget is None:
        return

    dynamo_module = getattr(torch, "_dynamo", None)
    if dynamo_module is None:
        raise RuntimeError(
            "run.activation_memory_budget is set, but torch._dynamo is unavailable "
            "on this PyTorch build."
        )
    _ = dynamo_module

    functorch_module = getattr(torch, "_functorch", None)
    functorch_config = getattr(functorch_module, "config", None)
    if functorch_config is None or not hasattr(
        functorch_config,
        "activation_memory_budget",
    ):
        raise RuntimeError(
            "run.activation_memory_budget is set, but "
            "torch._functorch.config.activation_memory_budget is unavailable on this "
            "PyTorch build."
        )
    functorch_config.activation_memory_budget = float(budget)


def maybe_compile_model(
    model: torch.nn.Module, device: torch.device, config: ExperimentConfig
) -> tuple[torch.nn.Module, bool, str]:
    if not config.run.use_torch_compile:
        return model, False, "disabled"
    if not hasattr(torch, "compile"):
        return model, False, "torch.compile unavailable"
    if device.type != "cuda":
        return model, False, f"skipped on {device.type}"

    _set_activation_memory_budget_if_configured(config)

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


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_autocast_context(device: torch.device, use_bf16: bool):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(
    model: torch.nn.Module,
    config: ExperimentConfig,
) -> optim.Optimizer:
    optimizer_cfg = config.train.optimizer
    optimizer_kwargs = {
        "lr": optimizer_cfg.learning_rate,
        "weight_decay": optimizer_cfg.weight_decay,
    }
    if optimizer_cfg.name == "adam":
        return optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_cfg.name == "adamw":
        return optim.AdamW(model.parameters(), **optimizer_kwargs)
    if optimizer_cfg.name == "sgd":
        return optim.SGD(model.parameters(), **optimizer_kwargs)
    raise ValueError(
        f"Unsupported train.optimizer.name '{optimizer_cfg.name}'. "
        "Expected one of: adam, adamw, sgd."
    )


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


def build_checkpoint_artifact_name(run_name: str) -> str:
    return f"{run_name}-checkpoint"


def build_final_model_artifact_name(run_name: str) -> str:
    return f"{run_name}-model"


def clone_config_with_run_settings(
    config: ExperimentConfig,
    *,
    run_name: str,
    resume_from_checkpoint: bool,
) -> ExperimentConfig:
    return replace(
        config,
        run=replace(
            config.run,
            run_name=run_name,
            resume_from_checkpoint=resume_from_checkpoint,
        ),
    )


def stdin_is_interactive() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def resolve_wandb_lineage(
    config: ExperimentConfig,
    logger_adapter,
    *,
    input_fn: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> ExperimentConfig:
    if config.logging.provider != "wandb":
        return config

    has_remote_artifact = getattr(logger_adapter, "has_remote_artifact", None)
    if not callable(has_remote_artifact):
        return config

    base_run_name = config.run.run_name
    if base_run_name is None:
        return config

    def remote_checkpoint_exists(run_name: str) -> bool:
        return bool(
            has_remote_artifact(
                project_name=config.run.project_name,
                artifact_name=build_checkpoint_artifact_name(run_name),
                alias="latest",
            )
        )

    if not remote_checkpoint_exists(base_run_name):
        return config

    interactive_mode = stdin_is_interactive() if interactive is None else interactive
    if not interactive_mode:
        if config.run.resume_from_checkpoint:
            print(
                f"Remote checkpoint already exists for run_name={base_run_name}; "
                "resuming latest lineage."
            )
            return clone_config_with_run_settings(
                config,
                run_name=base_run_name,
                resume_from_checkpoint=True,
            )
        raise ValueError(
            f"Remote checkpoint already exists for run_name={base_run_name}. "
            "Re-run interactively to resume or provide a distinct run.run_name."
        )

    print(f"Remote checkpoint already exists for run_name={base_run_name}.")
    while True:
        print("1. Resume from the existing latest remote checkpoint.")
        print("2. Start a new lineage with a manual suffix.")
        choice = input_fn("Select 1 or 2: ").strip()
        if choice == "1":
            return clone_config_with_run_settings(
                config,
                run_name=base_run_name,
                resume_from_checkpoint=True,
            )
        if choice != "2":
            print("Please enter 1 or 2.")
            continue

        while True:
            raw_suffix = input_fn("Enter a new lineage suffix: ").strip()
            if not raw_suffix:
                print("Suffix must be non-empty.")
                continue

            suffix = sanitize_wandb_name(raw_suffix)
            candidate_run_name = f"{base_run_name}-{suffix}"
            if remote_checkpoint_exists(candidate_run_name):
                print(
                    f"Remote checkpoint already exists for run_name={candidate_run_name}. "
                    "Enter a different suffix."
                )
                continue

            return clone_config_with_run_settings(
                config,
                run_name=candidate_run_name,
                resume_from_checkpoint=False,
            )


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
    compile_enabled: bool,
    run_paths: dict[str, Path],
    *,
    extra_metric_plugins: Sequence[MetricPlugin] | None = None,
    metrics_debug_timing: bool = False,
) -> TrainLoopResult:
    checkpoint_model = get_uncompiled_model(model)
    pad_id = int(loss_fn.ignore_index)
    wandb_cfg = config.logging.wandb
    wandb_enabled = config.logging.provider == "wandb"
    layer_labels = get_decoder_layer_labels(checkpoint_model)
    metrics_engine = MetricsEngine(
        build_default_metric_plugins(
            config=config,
            checkpoint_model=checkpoint_model,
            optimizer=optimizer,
            device=device,
            layer_labels=layer_labels,
            wandb_enabled=wandb_enabled,
            extra_plugins=extra_metric_plugins,
        ),
        enable_timing_debug=metrics_debug_timing,
    )
    tokens_seen_train = 0
    run_label = config.run.run_name or Path(run_paths["run_artifact_dir"]).name
    checkpoint_artifact_name = build_checkpoint_artifact_name(run_label)
    final_model_artifact_name = build_final_model_artifact_name(run_label)
    last_checkpoint_artifact_ref: str | None = None
    final_model_artifact_ref: str | None = None

    def save_checkpoint(
        epoch: int,
        next_batch_idx: int,
        global_step: int,
        *,
        aliases: tuple[str, ...] = ("latest",),
    ) -> str | None:
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
        return logger.save(
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
        restored_from_remote = False
        if not checkpoint_path.exists():
            restored_from_remote = logger.restore(
                str(checkpoint_path),
                artifact_name=checkpoint_artifact_name,
                artifact_type="checkpoint",
                alias="latest",
            )
            if not restored_from_remote:
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
        if restored_from_remote and checkpoint_path.exists():
            checkpoint_path.unlink()
        return start_epoch, start_batch_idx, global_step, tokens_seen

    model.train()
    non_blocking = device.type == "cuda"

    start_epoch, start_batch_idx, global_step, tokens_seen_train = load_checkpoint_if_available()

    last_avg_train_loss = 0.0
    last_val_metrics = {"val_loss": 0.0, "val_perplexity": 0.0}

    try:
        metrics_engine.on_train_start()
        for epoch in tqdm(range(start_epoch, config.train.epochs), desc="Epochs"):
            epoch_wall_start = time.perf_counter()
            epoch_train_loss_sum = 0.0
            epoch_token_count = 0

            for batch_idx, (input_seq, target_seq, key_padding_mask) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue

                next_global_step = global_step + 1
                schedule = build_metric_schedule(
                    next_global_step=next_global_step,
                    wandb_enabled=wandb_enabled,
                    wandb_cfg=wandb_cfg,
                    layer_labels_available=bool(layer_labels),
                )
                include_in_perf_aggregates = not (
                    compile_enabled
                    and next_global_step <= config.run.compile_warmup_steps
                )
                step_ctx = StepMetricsContext(
                    schedule=schedule,
                    global_step=global_step,
                    next_global_step=next_global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    train_loader_len=len(train_loader),
                    tokens_seen_train=tokens_seen_train,
                    step_loss=None,
                    include_in_perf_aggregates=include_in_perf_aggregates,
                )
                metrics_engine.on_step_start(step_ctx)

                input_seq = input_seq.to(device, non_blocking=non_blocking)
                target_seq = target_seq.to(device, non_blocking=non_blocking)
                key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

                optimizer.zero_grad()
                synchronize_if_cuda(device)
                step_start = time.perf_counter()

                synchronize_if_cuda(device)
                forward_start = time.perf_counter()
                with get_autocast_context(device=device, use_bf16=use_bf16):
                    output = model(input_seq, key_padding_mask=key_padding_mask)
                    loss_sum = loss_fn(
                        output.reshape(-1, output.size(-1)),
                        target_seq.reshape(-1),
                    )
                synchronize_if_cuda(device)
                forward_pass_time_ms = (time.perf_counter() - forward_start) * 1000.0

                valid_tokens = int((target_seq != pad_id).sum().item())
                if valid_tokens == 0:
                    continue

                loss = loss_sum / valid_tokens
                synchronize_if_cuda(device)
                backward_start = time.perf_counter()
                loss.backward()
                synchronize_if_cuda(device)
                backward_pass_time_ms = (time.perf_counter() - backward_start) * 1000.0

                step_ctx = replace(
                    step_ctx,
                    step_loss=float(loss.item()),
                    forward_pass_time_ms=float(forward_pass_time_ms),
                    backward_pass_time_ms=float(backward_pass_time_ms),
                )
                metrics_engine.after_backward(step_ctx)

                synchronize_if_cuda(device)
                optim_start = time.perf_counter()
                optimizer.step()
                synchronize_if_cuda(device)
                optim_step_time_ms = (time.perf_counter() - optim_start) * 1000.0
                step_time_ms = (time.perf_counter() - step_start) * 1000.0

                peak_memory_gib: float | None = None
                peak_reserved_memory_gib: float | None = None
                if device.type == "cuda":
                    peak_memory_gib = float(
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                    )
                    peak_reserved_memory_gib = float(
                        torch.cuda.max_memory_reserved(device) / (1024**3)
                    )

                global_step = next_global_step
                tokens_seen_train += valid_tokens
                step_ctx = replace(
                    step_ctx,
                    global_step=global_step,
                    tokens_seen_train=tokens_seen_train,
                    optim_step_time_ms=float(optim_step_time_ms),
                    step_time_ms=float(step_time_ms),
                    peak_memory_gib=peak_memory_gib,
                    peak_reserved_memory_gib=peak_reserved_memory_gib,
                )
                metrics_engine.after_optimizer_step(step_ctx)

                epoch_token_count += valid_tokens
                epoch_train_loss_sum += loss_sum.item()

                if schedule.should_log_this_step:
                    step_metrics = metrics_engine.collect_step_metrics(step_ctx)
                    logger.log(step_metrics, step=global_step)

                if schedule.periodic_val_due:
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        loss_fn,
                        device,
                        use_bf16=use_bf16,
                    )
                    model.train()
                    last_val_metrics = val_metrics
                    val_log_metrics = metrics_engine.collect_periodic_val_metrics(
                        PeriodicValMetricsContext(
                            schedule=schedule,
                            global_step=global_step,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            train_loader_len=len(train_loader),
                            tokens_seen_train=tokens_seen_train,
                            val_metrics=val_metrics,
                        )
                    )
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

                    last_checkpoint_artifact_ref = save_checkpoint(
                        next_epoch,
                        next_batch_idx,
                        global_step,
                        aliases=("latest",),
                    )

            avg_train_loss = epoch_train_loss_sum / max(epoch_token_count, 1)
            val_metrics = evaluate(model, val_loader, loss_fn, device, use_bf16=use_bf16)
            model.train()
            epoch_time_s = time.perf_counter() - epoch_wall_start
            epoch_metrics = metrics_engine.collect_epoch_metrics(
                EpochMetricsContext(
                    global_step=global_step,
                    epoch=epoch,
                    avg_train_loss=float(avg_train_loss),
                    tokens_seen_train=tokens_seen_train,
                    val_metrics=val_metrics,
                    epoch_time_s=float(epoch_time_s),
                )
            )
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
        metrics_engine.on_train_end()

    last_checkpoint_artifact_ref = save_checkpoint(
        config.train.epochs,
        0,
        global_step,
        aliases=("latest", "final"),
    )

    torch.save(checkpoint_model.state_dict(), run_paths["final_model_path"])
    final_model_artifact_ref = logger.save(
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
    return TrainLoopResult(
        global_step=global_step,
        final_train_loss=last_avg_train_loss,
        final_val_metrics=last_val_metrics,
        checkpoint_artifact_ref=last_checkpoint_artifact_ref,
        final_model_artifact_ref=final_model_artifact_ref,
    )


def model_pipeline(
    config: ExperimentConfig,
    *,
    extra_metric_plugins: Sequence[MetricPlugin] | None = None,
    metrics_debug_timing: bool = False,
) -> RunResult:
    register_builtin_adapters()
    validate_experiment_config(config)
    logger_adapter = get_logger_adapter(config.logging.provider)
    config = resolve_wandb_lineage(config, logger_adapter)
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
    optimizer = build_optimizer(model, config)
    loss_fn = CrossEntropyLoss(ignore_index=tokenized.vocab.special.pad_id, reduction="sum")

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

        train_result = train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            logger=logger,
            device=device,
            use_bf16=use_bf16,
            compile_enabled=compile_enabled,
            run_paths=run_paths,
            extra_metric_plugins=extra_metric_plugins,
            metrics_debug_timing=metrics_debug_timing,
        )
    finally:
        logger.close()

    return RunResult(
        model=model,
        device=device,
        run_artifact_dir=str(run_paths["run_artifact_dir"]),
        checkpoint_path=str(run_paths["checkpoint_path"]),
        final_model_path=str(run_paths["final_model_path"]),
        checkpoint_artifact_ref=train_result.checkpoint_artifact_ref,
        final_model_artifact_ref=train_result.final_model_artifact_ref,
        global_step=train_result.global_step,
        final_train_loss=train_result.final_train_loss,
        final_val_loss=float(train_result.final_val_metrics["val_loss"]),
        final_val_perplexity=float(train_result.final_val_metrics["val_perplexity"]),
    )
