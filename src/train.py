from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapters import register_builtin_adapters
from .core.config import (
    ExperimentConfig,
    ResolvedTrainBatchingConfig,
    resolve_train_batching,
    resolve_train_learning_rate,
    validate_experiment_config,
)
from .core.registry import (
    get_dataset_adapter,
    get_logger_adapter,
    get_model_adapter,
    get_tokenizer_adapter,
)
from .core.types import RunResult
from .training.artifacts import (
    build_checkpoint_artifact_name,
    build_final_model_artifact_name,
    prepare_run_artifact_paths,
    resolve_wandb_lineage,
    write_run_metadata,
)
from .training.data import build_data_loaders
from .training.evaluate import evaluate
from .training.metrics import (
    EpochMetricsContext,
    MicroBatchMetricsContext,
    MetricsEngine,
    PeriodicValMetricsContext,
    StepMetricsContext,
    build_default_metric_plugins,
    build_metric_schedule,
    get_decoder_layer_labels,
)
from .training.optimizer import (
    build_optimizer,
    move_optimizer_state_to_device,
    scale_gradients_by_token_count,
)
from .training.runtime import (
    get_autocast_context,
    get_best_device,
    get_uncompiled_model,
    maybe_compile_model,
    set_seed,
    should_enable_bf16_autocast,
    synchronize_if_cuda,
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
    completed_epochs: int
    epoch_end_validation_ran: bool


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
    persist_local_artifacts = bool(config.run.persist_local_artifacts)
    artifact_io_enabled = bool(config.logging.enable_artifact_io)
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
        if not persist_local_artifacts:
            return None

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
        if not artifact_io_enabled:
            return None
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
            if artifact_io_enabled:
                restored_from_remote = logger.restore(
                    str(checkpoint_path),
                    artifact_name=checkpoint_artifact_name,
                    artifact_type="checkpoint",
                    alias="latest",
                )
            else:
                print(
                    "Artifact I/O is disabled; skipping remote checkpoint restore."
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
    batching: ResolvedTrainBatchingConfig = resolve_train_batching(config.train)
    accumulation_steps = int(batching.accumulation_steps)
    run_validation = bool(config.train.run_validation)
    train_loader_len = len(train_loader)

    start_epoch, start_batch_idx, global_step, tokens_seen_train = load_checkpoint_if_available()

    last_avg_train_loss = 0.0
    last_val_metrics = {"val_loss": float("nan"), "val_perplexity": float("nan")}
    completed_epochs = int(start_epoch)
    epoch_end_validation_ran = False

    try:
        metrics_engine.on_train_start()
        for epoch in tqdm(range(start_epoch, config.train.epochs), desc="Epochs"):
            epoch_wall_start = time.perf_counter()
            epoch_train_loss_sum = 0.0
            epoch_token_count = 0

            micro_batches_in_step = 0
            step_ctx: StepMetricsContext | None = None
            step_start = 0.0
            step_forward_pass_time_ms = 0.0
            step_backward_pass_time_ms = 0.0
            should_measure_step_timing = False
            step_loss_sum = 0.0
            step_token_count = 0
            step_last_batch_idx = 0

            for batch_idx, (input_seq, target_seq, key_padding_mask) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue

                if micro_batches_in_step == 0:
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
                        train_loader_len=train_loader_len,
                        tokens_seen_train=tokens_seen_train,
                        step_loss=None,
                        include_in_perf_aggregates=include_in_perf_aggregates,
                        model=checkpoint_model,
                        optimizer=optimizer,
                    )
                    metrics_engine.on_step_start(step_ctx)

                    optimizer.zero_grad()
                    should_measure_step_timing = (
                        wandb_enabled
                        and wandb_cfg.enable_step_time
                        and step_ctx.include_in_perf_aggregates
                    )
                    if should_measure_step_timing:
                        synchronize_if_cuda(device)
                        step_start = time.perf_counter()
                    step_forward_pass_time_ms = 0.0
                    step_backward_pass_time_ms = 0.0
                    step_loss_sum = 0.0
                    step_token_count = 0

                if step_ctx is None:
                    raise RuntimeError("Step metrics context was not initialized.")

                step_last_batch_idx = batch_idx
                input_seq = input_seq.to(device, non_blocking=non_blocking)
                target_seq = target_seq.to(device, non_blocking=non_blocking)
                key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

                forward_start = 0.0
                if should_measure_step_timing:
                    synchronize_if_cuda(device)
                    forward_start = time.perf_counter()
                with get_autocast_context(device=device, use_bf16=use_bf16):
                    # FORWARD PASS
                    output = model(input_seq, key_padding_mask=key_padding_mask)
                    # COMPUTE LOSS
                    loss_sum = loss_fn(
                        output.reshape(-1, output.size(-1)),
                        target_seq.reshape(-1),
                    )
                if should_measure_step_timing:
                    synchronize_if_cuda(device)
                    step_forward_pass_time_ms += (
                        time.perf_counter() - forward_start
                    ) * 1000.0

                valid_tokens = int((target_seq != pad_id).sum().item())
                if valid_tokens > 0:
                    step_token_count += valid_tokens
                    step_loss_sum += loss_sum.item()
                    epoch_token_count += valid_tokens
                    epoch_train_loss_sum += loss_sum.item()
                    micro_batch_in_step = micro_batches_in_step + 1

                    backward_start = 0.0
                    if should_measure_step_timing:
                        synchronize_if_cuda(device)
                        backward_start = time.perf_counter()
                    # CALCULATE GRADIENTS
                    loss_sum.backward()
                    
                    if should_measure_step_timing:
                        synchronize_if_cuda(device)
                        step_backward_pass_time_ms += (
                            (time.perf_counter() - backward_start) * 1000.0
                        )
                    metrics_engine.after_microbatch_backward(
                        MicroBatchMetricsContext(
                            step_ctx=step_ctx,
                            micro_batch_in_step=micro_batch_in_step,
                            accumulation_steps=accumulation_steps,
                            valid_tokens=valid_tokens,
                        )
                    )

                micro_batches_in_step += 1
                is_last_batch_in_epoch = batch_idx + 1 >= train_loader_len
                if (
                    micro_batches_in_step < accumulation_steps
                    and not is_last_batch_in_epoch
                ):
                    continue

                if step_token_count <= 0:
                    micro_batches_in_step = 0
                    continue
                
                # SCALE GRADs BY TOKEN COUNT
                scale_gradients_by_token_count(checkpoint_model, step_token_count)
                step_loss = step_loss_sum / step_token_count
                step_ctx = replace(
                    step_ctx,
                    batch_idx=step_last_batch_idx,
                    step_loss=float(step_loss),
                    forward_pass_time_ms=float(step_forward_pass_time_ms),
                    backward_pass_time_ms=float(step_backward_pass_time_ms),
                )
                metrics_engine.after_backward(step_ctx)

                optim_start = 0.0
                if should_measure_step_timing:
                    synchronize_if_cuda(device)
                    optim_start = time.perf_counter()
                    
                # UPDATE WEIGHTS BASED ON CALCULATED GRADIENTS    
                optimizer.step()
                
                optim_step_time_ms: float | None = None
                step_time_ms: float | None = None
                if should_measure_step_timing:
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

                global_step = int(step_ctx.next_global_step)
                tokens_seen_train += step_token_count
                step_ctx = replace(
                    step_ctx,
                    batch_idx=step_last_batch_idx,
                    global_step=global_step,
                    tokens_seen_train=tokens_seen_train,
                    optim_step_time_ms=optim_step_time_ms,
                    step_time_ms=step_time_ms,
                    peak_memory_gib=peak_memory_gib,
                    peak_reserved_memory_gib=peak_reserved_memory_gib,
                )
                metrics_engine.after_optimizer_step(step_ctx)

                if step_ctx.schedule.should_log_this_step:
                    step_metrics = metrics_engine.collect_step_metrics(step_ctx)
                    logger.log(step_metrics, step=global_step)

                if run_validation and step_ctx.schedule.periodic_val_due:
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
                            schedule=step_ctx.schedule,
                            global_step=global_step,
                            epoch=epoch,
                            batch_idx=step_last_batch_idx,
                            train_loader_len=train_loader_len,
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
                    next_batch_idx = step_last_batch_idx + 1
                    if next_batch_idx >= train_loader_len:
                        next_epoch += 1
                        next_batch_idx = 0

                    last_checkpoint_artifact_ref = save_checkpoint(
                        next_epoch,
                        next_batch_idx,
                        global_step,
                        aliases=("latest",),
                    )

                micro_batches_in_step = 0

            avg_train_loss = epoch_train_loss_sum / max(epoch_token_count, 1)
            completed_epochs = int(epoch + 1)
            epoch_time_s = time.perf_counter() - epoch_wall_start
            val_metrics = last_val_metrics
            if run_validation:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                    use_bf16=use_bf16,
                )
                model.train()
                epoch_end_validation_ran = True
                last_val_metrics = val_metrics
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

            if run_validation:
                print(
                    f"Epoch {epoch + 1}/{config.train.epochs} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"val_loss={val_metrics['val_loss']:.4f} | "
                    f"val_perplexity={val_metrics['val_perplexity']:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{config.train.epochs} | "
                    f"train_loss={avg_train_loss:.4f}"
                )

            last_avg_train_loss = float(avg_train_loss)
    finally:
        metrics_engine.on_train_end()

    last_checkpoint_artifact_ref = save_checkpoint(
        config.train.epochs,
        0,
        global_step,
        aliases=("latest", "final"),
    )

    if persist_local_artifacts:
        torch.save(checkpoint_model.state_dict(), run_paths["final_model_path"])
    if artifact_io_enabled and persist_local_artifacts:
        final_model_metadata = {
            "global_step": int(global_step),
            "final_train_loss": float(last_avg_train_loss),
            "run_name": run_label,
            "group_name": config.run.group_name,
        }
        if epoch_end_validation_ran:
            final_model_metadata["final_val_loss"] = float(last_val_metrics["val_loss"])
            final_model_metadata["final_val_perplexity"] = float(
                last_val_metrics["val_perplexity"]
            )
        final_model_artifact_ref = logger.save(
            str(run_paths["final_model_path"]),
            artifact_name=final_model_artifact_name,
            artifact_type="model",
            aliases=("latest", "final"),
            metadata=final_model_metadata,
        )
    return TrainLoopResult(
        global_step=global_step,
        final_train_loss=last_avg_train_loss,
        final_val_metrics=last_val_metrics,
        checkpoint_artifact_ref=last_checkpoint_artifact_ref,
        final_model_artifact_ref=final_model_artifact_ref,
        completed_epochs=completed_epochs,
        epoch_end_validation_ran=epoch_end_validation_ran,
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
    optimizer = build_optimizer(
        model,
        config,
        learning_rate=learning_rate_cfg.applied_learning_rate,
    )
    # We need to do reduction="sum" because we have grad-acc, if we set it to "mean" then
    # at the end of effective batch we will have (avg_loss_mb_1 + avg_loss_mb_2 ...)/num_of_mb which i wrong,
    # what we want is (loss_mb_1 + loss_mb_2 + ...)/num_of_mb hence we use reduction="sum"
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
                "lr_base": float(learning_rate_cfg.base_learning_rate),
                "lr_scale_factor": float(learning_rate_cfg.scale_factor),
                "lr_applied": float(learning_rate_cfg.applied_learning_rate),
                "lr_scaling_active": float(1 if learning_rate_cfg.scaling_active else 0),
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
        completed_epochs=train_result.completed_epochs,
        epoch_end_validation_ran=train_result.epoch_end_validation_ran,
    )
