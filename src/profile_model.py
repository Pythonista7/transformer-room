from __future__ import annotations

import gc
import math
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss

from .adapters import register_builtin_adapters
from .adapters.tokenizers import build_hf_pretrained_tokenizer_bundle
from .core.config import (
    ExperimentConfig,
    resolve_train_batching,
    resolve_train_learning_rate,
    validate_experiment_config,
)
from .core.registry import get_dataset_adapter, get_model_adapter, get_tokenizer_adapter
from .core.types import ProfileResult
from .train import _build_token_byte_lengths, _unpack_batch_tensors, _safe_len
from .training.artifacts import prepare_run_artifact_paths, write_run_metadata
from .training.data import build_data_loaders, build_streaming_data_loaders
from .training.optimizer import (
    build_lr_scheduler,
    build_optimizer,
    scale_gradients_by_token_count,
)
from .training.runtime import (
    get_autocast_context,
    get_best_device,
    maybe_compile_model,
    set_seed,
    should_enable_bf16_autocast,
    synchronize_if_cuda,
)


def _profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profile_model(
    config: ExperimentConfig,
    *,
    num_steps: int = 3,
    trace_path: str | Path | None = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
) -> ProfileResult:
    if int(num_steps) <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    register_builtin_adapters()
    validate_experiment_config(config)
    set_seed(config.run.seed)
    print(f"Using seed: {config.run.seed}")

    device = get_best_device()
    print(f"Using device: {device}")
    use_bf16 = should_enable_bf16_autocast(device)
    print(f"bf16 autocast: {'enabled' if use_bf16 else 'disabled'}")
    batching = resolve_train_batching(config.train)
    learning_rate_cfg = resolve_train_learning_rate(config.train)
    print(
        "Batching config: "
        f"effective_batch_size={batching.effective_batch_size} | "
        f"micro_batch_size={batching.micro_batch_size} | "
        f"accumulation_steps={batching.accumulation_steps}"
    )
    print(
        "Learning-rate config: "
        f"base_learning_rate={learning_rate_cfg.base_learning_rate:g} | "
        f"lr_scaling={learning_rate_cfg.scaling_mode} | "
        f"scale_factor={learning_rate_cfg.scale_factor:g} | "
        f"applied_learning_rate={learning_rate_cfg.applied_learning_rate:g} | "
        f"scaling_active={learning_rate_cfg.scaling_active}"
    )

    run_paths = prepare_run_artifact_paths(config)
    if config.train.data_mode == "streaming":
        if config.tokenizer.name != "hf_pretrained":
            raise ValueError(
                "Streaming mode requires tokenizer.name='hf_pretrained'."
            )
        tokenized = build_hf_pretrained_tokenizer_bundle(
            config.tokenizer,
            bpb_metrics_enabled=False,
        )
        write_run_metadata(config=config, tokenized=tokenized, run_paths=run_paths)
        train_loader, val_loader = build_streaming_data_loaders(
            config=config,
            tokenized=tokenized,
            pin_memory=device.type == "cuda",
        )
    else:
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

    token_byte_lengths = tokenized.token_byte_lengths
    if token_byte_lengths is None and config.tokenizer.name == "bpe":
        base_vocab_size = tokenized.vocab.special.base_vocab_size
        if base_vocab_size is None:
            raise ValueError("BPE tokenization expects base_vocab_size to be set.")
        token_byte_lengths = _build_token_byte_lengths(
            id_to_token=tokenized.vocab.id_to_token,
            base_vocab_size=base_vocab_size,
        )
    _ = token_byte_lengths

    model_adapter = get_model_adapter(config.model.name)
    model = model_adapter.build(
        cfg=config.model,
        vocab=tokenized.vocab,
        special=tokenized.vocab.special,
    )
    checkpoint_model = model
    model = model.to(device)

    param_count = sum(p.numel() for p in checkpoint_model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = build_optimizer(
        model,
        config,
        learning_rate=learning_rate_cfg.applied_learning_rate,
    )
    train_loader_len_for_scheduler = _safe_len(train_loader)
    total_optimizer_steps: int | None
    if config.train.max_steps is not None:
        total_optimizer_steps = int(config.train.max_steps)
    elif train_loader_len_for_scheduler is not None:
        total_optimizer_steps = int(
            config.train.epochs
            * math.ceil(
                float(train_loader_len_for_scheduler)
                / float(max(1, batching.accumulation_steps))
            )
        )
    else:
        total_optimizer_steps = None

    scheduler = build_lr_scheduler(
        optimizer,
        config,
        total_optimizer_steps=total_optimizer_steps,
    )
    loss_fn = CrossEntropyLoss(ignore_index=tokenized.vocab.special.pad_id, reduction="sum")

    model, compile_enabled, compile_status = maybe_compile_model(model, device, config)
    print(f"torch.compile: {compile_status}")

    trace_path_obj = (
        Path(trace_path)
        if trace_path is not None
        else Path(run_paths["run_artifact_dir"]) / "profile_trace.json"
    )
    trace_path_obj.parent.mkdir(parents=True, exist_ok=True)

    non_blocking = device.type == "cuda"
    pad_id = int(loss_fn.ignore_index)
    accumulation_steps = int(batching.accumulation_steps)
    steps_profiled = 0

    try:
        model.train()
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(0)
        train_iter = iter(train_loader)

        profiler_kwargs = {
            "activities": _profiler_activities(device),
            "record_shapes": bool(record_shapes),
            "profile_memory": bool(profile_memory),
            "with_stack": bool(with_stack),
            "acc_events": True,
        }
        with torch.profiler.profile(**profiler_kwargs) as prof:
            while steps_profiled < int(num_steps):
                optimizer.zero_grad()
                step_token_count = 0
                micro_batches_in_step = 0
                saw_batch = False

                while micro_batches_in_step < accumulation_steps:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        break

                    saw_batch = True
                    (
                        input_seq,
                        target_seq,
                        key_padding_mask,
                        _target_byte_lengths,
                    ) = _unpack_batch_tensors(batch)
                    input_seq = input_seq.to(device, non_blocking=non_blocking)
                    target_seq = target_seq.to(device, non_blocking=non_blocking)
                    key_padding_mask = key_padding_mask.to(
                        device,
                        non_blocking=non_blocking,
                    )

                    with get_autocast_context(device=device, use_bf16=use_bf16):
                        output = model(input_seq, key_padding_mask=key_padding_mask)
                        loss_sum = loss_fn(
                            output.reshape(-1, output.size(-1)),
                            target_seq.reshape(-1),
                        )

                    valid_tokens = int((target_seq != pad_id).sum().item())
                    if valid_tokens > 0:
                        loss_sum.backward()
                        step_token_count += valid_tokens

                    micro_batches_in_step += 1

                if not saw_batch:
                    break
                if step_token_count <= 0:
                    continue

                scale_gradients_by_token_count(checkpoint_model, step_token_count)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                synchronize_if_cuda(device)
                prof.step()
                steps_profiled += 1

        if steps_profiled <= 0:
            raise RuntimeError("No optimizer steps were profiled; train loader may be empty.")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(trace_path_obj))
        print(f"Saved profiler trace to: {trace_path_obj}")
    finally:
        if train_loader.multiprocessing_context is not None:
            for achild in train_loader.multiprocessing_context.active_children():
                print(
                    f"waiting for child process: {achild.name} | {achild.pid} to finish..."
                )
                achild.join()
                print(
                    f"child process: {achild.name} | {achild.pid} "
                    f"finished with exit code {achild.exitcode}"
                )
        else:
            print("No child processes to wait for in `train_loader.multiprocessing_context`")

        if val_loader is not None:
            del val_loader

        gc.collect()

    return ProfileResult(
        model=model,
        device=device,
        run_artifact_dir=str(run_paths["run_artifact_dir"]),
        trace_path=str(trace_path_obj),
        steps_profiled=steps_profiled,
        compile_enabled=compile_enabled,
        bf16_autocast_enabled=use_bf16,
    )
