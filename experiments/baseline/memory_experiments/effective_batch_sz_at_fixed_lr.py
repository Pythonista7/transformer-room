"""
Note: this experiment was run before I introduced gated check for lr-scaling when accumulating, 
but the results of this expriment are designed with the goal of accessing gradient quality metrics (coherence, global grad norm, adam snr norm) at various effective batch sizes with a fixed base learning rate, 
to see how the gradient quality changes as we increase the effective batch size via accumulation, specifically looking as coherence as a core metric here, should not change with lr-scaling on reruns.
"""
from __future__ import annotations

import gc
import importlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HFTextDatasetConfig,
    HoldoutSplitConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from src.train import model_pipeline
from src.training.metrics import (
    BaseMetricPlugin,
    MicroBatchMetricsContext,
    MetricPayload,
    StepMetricsContext,
)
from src.training.metrics.plugins.global_grad_norm import compute_global_grad_norm
from src.training.metrics.plugins.parameter_optimizer_norms import compute_adam_state_norms

PROJECT_NAME = "transformer-room-baseline"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"
SEQ_LEN = 1024
STRIDE = 1024

MICRO_BATCH_SIZE = 28

MEASURE_TRAINING_ONLY = True
EPOCHS = 1
DATA_FRACTION = 0.5
LEARNING_RATE = 3e-5
SEED = 42


@dataclass(frozen=True, slots=True)
class LoggedStepSummary:
    max_peak_memory_gib: float | None
    max_peak_reserved_memory_gib: float | None
    avg_step_time_ms: float | None
    avg_tokens_per_sec: float | None
    avg_global_grad_norm: float | None
    avg_adam_elemwise_snr_norm: float | None
    avg_gradient_coherence: float | None
    min_gradient_coherence: float | None
    max_gradient_coherence: float | None
    avg_gradient_coherence_pairs: float | None


@dataclass(frozen=True, slots=True)
class TrialResult:
    accumulation_steps: int
    effective_batch_size: int
    run_name: str
    status: str
    global_step: int | None = None
    run_artifact_dir: str | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    max_peak_memory_gib: float | None = None
    max_peak_reserved_memory_gib: float | None = None
    avg_step_time_ms: float | None = None
    avg_tokens_per_sec: float | None = None
    avg_global_grad_norm: float | None = None
    avg_adam_elemwise_snr_norm: float | None = None
    avg_gradient_coherence: float | None = None
    min_gradient_coherence: float | None = None
    max_gradient_coherence: float | None = None
    avg_gradient_coherence_pairs: float | None = None
    error_type: str | None = None
    error_message: str | None = None


def _resolve_hf_load_dataset():
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset support requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc

    load_dataset = getattr(datasets_module, "load_dataset", None)
    if not callable(load_dataset):
        raise ImportError(
            "Resolved `datasets` module does not expose `load_dataset`. "
            "A local `datasets/` directory may be shadowing the Hugging Face package."
        )
    return load_dataset


def _iter_wikitext_tokens(text: str) -> Iterable[str]:
    for token in text.strip().split():
        if token:
            yield token


def ensure_wikitext_vocab_file(
    dataset_name: str,
    dataset_config: str,
    vocab_path: Path,
) -> int:
    if vocab_path.exists():
        size = 0
        with vocab_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    size += 1
        if size <= 0:
            raise ValueError(f"Existing vocab file is empty: {vocab_path}")
        print(f"Using existing Wikitext vocab file: {vocab_path} | size={size:,}")
        return size

    load_dataset = _resolve_hf_load_dataset()
    splits = ("train", "validation", "test")
    token_set: set[str] = {" ", "\n", "\t"}

    for split in splits:
        dataset = load_dataset(dataset_name, name=dataset_config, split=split)
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            token_set.update(_iter_wikitext_tokens(text))

    ordered_tokens = sorted(token_set)
    byte_tokens = [tuple(token.encode("utf-8")) for token in ordered_tokens]

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as handle:
        for token in byte_tokens:
            handle.write(f"{token}\n")

    print(
        f"Created Wikitext vocab file: {vocab_path} | "
        f"tokens={len(byte_tokens):,} | splits={','.join(splits)}"
    )
    return len(byte_tokens)


def classify_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
    )


def clear_runtime_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak_memory_stats = getattr(torch.cuda, "reset_peak_memory_stats", None)
        if callable(reset_peak_memory_stats):
            try:
                reset_peak_memory_stats()
            except Exception:
                pass
    reset_compiler = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(reset_compiler):
        reset_compiler()


def merge_logged_metrics_by_step(
    logged_entries: list[tuple[int | None, dict[str, float]]],
) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for step, payload in logged_entries:
        if step is None:
            continue
        step_metrics = merged.setdefault(int(step), {})
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                step_metrics[key] = float(value)
    return merged


def _mean_from_step_metrics(
    step_metrics: dict[int, dict[str, float]],
    key: str,
) -> float | None:
    values = [
        float(metrics[key])
        for metrics in step_metrics.values()
        if key in metrics
    ]
    if not values:
        return None
    return float(mean(values))


def _min_from_step_metrics(
    step_metrics: dict[int, dict[str, float]],
    key: str,
) -> float | None:
    values = [
        float(metrics[key])
        for metrics in step_metrics.values()
        if key in metrics
    ]
    if not values:
        return None
    return float(min(values))


def _max_from_step_metrics(
    step_metrics: dict[int, dict[str, float]],
    key: str,
) -> float | None:
    values = [
        float(metrics[key])
        for metrics in step_metrics.values()
        if key in metrics
    ]
    if not values:
        return None
    return float(max(values))


def compute_avg_tokens_per_sec(
    logged_entries: list[tuple[int | None, dict[str, float]]],
) -> float | None:
    step_metrics = merge_logged_metrics_by_step(logged_entries)
    if not step_metrics:
        return None

    rates: list[float] = []
    prev_tokens_seen: float | None = None
    for step in sorted(step_metrics):
        metrics = step_metrics[step]
        tokens_seen = metrics.get("tokens_seen_train")
        step_time_ms = metrics.get("step_time_ms")
        if tokens_seen is None:
            continue
        if step_time_ms is None or step_time_ms <= 0:
            prev_tokens_seen = tokens_seen
            continue

        delta_tokens = (
            tokens_seen if prev_tokens_seen is None else tokens_seen - prev_tokens_seen
        )
        prev_tokens_seen = tokens_seen
        if delta_tokens <= 0:
            continue
        rates.append(delta_tokens / (step_time_ms / 1000.0))

    if not rates:
        return None
    return float(mean(rates))


def summarize_logged_steps(
    logged_entries: list[tuple[int | None, dict[str, float]]],
) -> LoggedStepSummary:
    step_metrics = merge_logged_metrics_by_step(logged_entries)
    if not step_metrics:
        return LoggedStepSummary(
            max_peak_memory_gib=None,
            max_peak_reserved_memory_gib=None,
            avg_step_time_ms=None,
            avg_tokens_per_sec=None,
            avg_global_grad_norm=None,
            avg_adam_elemwise_snr_norm=None,
            avg_gradient_coherence=None,
            min_gradient_coherence=None,
            max_gradient_coherence=None,
            avg_gradient_coherence_pairs=None,
        )

    return LoggedStepSummary(
        max_peak_memory_gib=_max_from_step_metrics(step_metrics, "peak_memory_gib"),
        max_peak_reserved_memory_gib=_max_from_step_metrics(
            step_metrics,
            "peak_reserved_memory_gib",
        ),
        avg_step_time_ms=_mean_from_step_metrics(step_metrics, "step_time_ms"),
        avg_tokens_per_sec=compute_avg_tokens_per_sec(logged_entries),
        avg_global_grad_norm=_mean_from_step_metrics(step_metrics, "global_grad_norm"),
        avg_adam_elemwise_snr_norm=_mean_from_step_metrics(
            step_metrics,
            "adam_elemwise_snr_norm",
        ),
        avg_gradient_coherence=_mean_from_step_metrics(step_metrics, "gradient_coherence"),
        min_gradient_coherence=_min_from_step_metrics(step_metrics, "gradient_coherence"),
        max_gradient_coherence=_max_from_step_metrics(step_metrics, "gradient_coherence"),
        avg_gradient_coherence_pairs=_mean_from_step_metrics(
            step_metrics,
            "gradient_coherence_pairs",
        ),
    )


class GradientQualitySummaryPlugin(BaseMetricPlugin):
    name = "gradient_quality_summary"

    def __init__(self) -> None:
        self.logged_entries: list[tuple[int | None, dict[str, float]]] = []
        self._prev_cumulative_grads: dict[int, torch.Tensor] = {}
        self._prev_normalized_micro_grads: dict[int, torch.Tensor] = {}
        self._step_coherence_values: list[float] = []
        self._step_gradient_coherence: float | None = None
        self._step_gradient_coherence_pairs: int = 0
        self._step_global_grad_norm: float | None = None
        self._step_microbatch_count: int = 0
        self._step_expected_accumulation_steps: int = 1

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self._prev_cumulative_grads = {}
        self._prev_normalized_micro_grads = {}
        self._step_coherence_values = []
        self._step_gradient_coherence = None
        self._step_gradient_coherence_pairs = 0
        self._step_global_grad_norm = None
        self._step_microbatch_count = 0
        self._step_expected_accumulation_steps = 1

    def after_microbatch_backward(self, ctx: MicroBatchMetricsContext) -> None:
        self._step_microbatch_count += 1
        self._step_expected_accumulation_steps = int(ctx.accumulation_steps)
        if ctx.valid_tokens <= 0:
            return

        model = ctx.step_ctx.model
        if not isinstance(model, torch.nn.Module):
            return

        # We are basically calculating the required data to find the 
        # cosine similarity between the current and the previous micro-batch gradients,
        # which is a measure of their coherence. 
        # The gradients are normalized by the number of valid tokens to account for varying micro-batch sizes. 
        # We keep track of the cumulative gradients to compute the micro-batch gradients, and we store the previous normalized micro-batch gradients to compute the dot product and norms needed for cosine similarity. 
        # The coherence values are accumulated across micro-batches in a step and averaged at the end of the step.
        inv_token_count = 1.0 / float(ctx.valid_tokens)
        dot_total: torch.Tensor | None = None
        prev_sq_total: torch.Tensor | None = None
        curr_sq_total: torch.Tensor | None = None
        curr_normalized_micro_grads: dict[int, torch.Tensor] = {}

        with torch.no_grad():
            for param in model.parameters():
                grad = param.grad
                if grad is None:
                    continue
                
                grad_cpu = grad.detach().to(device="cpu", dtype=torch.float32)
                prev_cumulative = self._prev_cumulative_grads.get(id(param))
                micro_grad = grad_cpu if prev_cumulative is None else grad_cpu - prev_cumulative
                normalized_micro_grad = micro_grad.mul(inv_token_count)

                curr_normalized_micro_grads[id(param)] = normalized_micro_grad
                self._prev_cumulative_grads[id(param)] = grad_cpu

                prev_normalized_micro_grad = self._prev_normalized_micro_grads.get(id(param))
                if prev_normalized_micro_grad is None:
                    continue

                dot_value = (prev_normalized_micro_grad * normalized_micro_grad).sum()
                prev_sq_value = prev_normalized_micro_grad.pow(2).sum()
                curr_sq_value = normalized_micro_grad.pow(2).sum()

                dot_total = dot_value if dot_total is None else dot_total + dot_value
                prev_sq_total = (
                    prev_sq_value if prev_sq_total is None else prev_sq_total + prev_sq_value
                )
                curr_sq_total = (
                    curr_sq_value if curr_sq_total is None else curr_sq_total + curr_sq_value
                )

        if (
            self._prev_normalized_micro_grads
            and dot_total is not None
            and prev_sq_total is not None
            and curr_sq_total is not None
        ):
            denom = prev_sq_total.sqrt() * curr_sq_total.sqrt()
            denom_scalar = float(denom.item())
            if denom_scalar > 0.0:
                coherence = float((dot_total / denom).item())
                self._step_coherence_values.append(coherence)

        self._prev_normalized_micro_grads = curr_normalized_micro_grads

    def after_backward(self, ctx: StepMetricsContext) -> None:
        if isinstance(ctx.model, torch.nn.Module):
            self._step_global_grad_norm = compute_global_grad_norm(ctx.model)

        if self._step_coherence_values:
            self._step_gradient_coherence = float(mean(self._step_coherence_values))
            self._step_gradient_coherence_pairs = len(self._step_coherence_values)
        elif (
            self._step_microbatch_count == 1
            and self._step_expected_accumulation_steps == 1
        ):
            self._step_gradient_coherence = 1.0
            self._step_gradient_coherence_pairs = 0
        else:
            self._step_gradient_coherence = None
            self._step_gradient_coherence_pairs = 0

        self._prev_cumulative_grads = {}
        self._prev_normalized_micro_grads = {}

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        payload: MetricPayload = {}
        payload["tokens_seen_train"] = float(ctx.tokens_seen_train)
        if ctx.step_time_ms is not None:
            payload["step_time_ms"] = float(ctx.step_time_ms)
        if ctx.peak_memory_gib is not None:
            payload["peak_memory_gib"] = float(ctx.peak_memory_gib)
        if ctx.peak_reserved_memory_gib is not None:
            payload["peak_reserved_memory_gib"] = float(ctx.peak_reserved_memory_gib)
        if self._step_global_grad_norm is not None:
            payload["global_grad_norm"] = float(self._step_global_grad_norm)

        optimizer_obj = ctx.optimizer
        if isinstance(optimizer_obj, optim.Optimizer):
            adam_metrics = compute_adam_state_norms(optimizer_obj)
            snr_norm = adam_metrics.get("adam_elemwise_snr_norm")
            if isinstance(snr_norm, (int, float)):
                payload["adam_elemwise_snr_norm"] = float(snr_norm)

        if self._step_gradient_coherence is not None:
            payload["gradient_coherence"] = float(self._step_gradient_coherence)
        payload["gradient_coherence_pairs"] = float(self._step_gradient_coherence_pairs)

        if ctx.include_in_perf_aggregates:
            self.logged_entries.append((ctx.global_step, payload))

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if not ctx.schedule.should_log_this_step:
            return {}

        metrics: MetricPayload = {}
        if self._step_gradient_coherence is not None:
            metrics["gradient_coherence"] = float(self._step_gradient_coherence)
        metrics["gradient_coherence_pairs"] = float(self._step_gradient_coherence_pairs)
        return metrics

    @property
    def summary(self) -> LoggedStepSummary:
        return summarize_logged_steps(self.logged_entries)

    def on_train_end(self) -> None:
        self._prev_cumulative_grads = {}
        self._prev_normalized_micro_grads = {}
        self._step_coherence_values = []
        self._step_microbatch_count = 0
        self._step_expected_accumulation_steps = 1


def build_config(
    *,
    run_name: str,
    sweep_group: str,
    base_vocab_size: int,
    accumulation_steps: int,
) -> ExperimentConfig:
    
    effective_batch_size = MICRO_BATCH_SIZE * accumulation_steps
    vocab_path = (
        PROJECT_ROOT
        / "src"
        / "vocabs"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )

    return ExperimentConfig(
        run=RunConfig(
            project_name=PROJECT_NAME,
            run_name=run_name,
            group_name=sweep_group,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            persist_local_artifacts=False,
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=0,
            use_torch_compile=True,
            torch_compile_mode="default",
            torch_compile_fullgraph=False,
            torch_compile_dynamic=False,
            compile_warmup_steps=3,
            seed=SEED,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            split="train",
            text_field="text",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=base_vocab_size,
            num_special_tokens=3,
            vocab_path=str(vocab_path),
        ),
        model=BaselineDecoderConfig(
            d_model=768,
            n_heads=8,
            layers=12,
            dropout=0.1,
        ),
        train=TrainConfig(
            epochs=EPOCHS,
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=LEARNING_RATE,
                weight_decay=0.0,
            ),
            effective_batch_size=effective_batch_size,
            micro_batch_size=MICRO_BATCH_SIZE,
            accumulation_steps=accumulation_steps,
            lr_scaling="sqrt" if accumulation_steps > 1 else "none",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_fraction=DATA_FRACTION,
            run_validation=not MEASURE_TRAINING_ONLY,
        ),
        split=HoldoutSplitConfig(
            train_fraction=0.9,
            seed=SEED,
            shuffle=False,
        ),
        logging=LoggingConfig(
            provider="wandb",
            enable_artifact_io=False,
            wandb=WandbMetricsConfig(
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=not MEASURE_TRAINING_ONLY,
                enable_perplexity=False,
                enable_step_time=True,
                enable_peak_memory=True,
                enable_global_grad_norm=True,
                enable_global_param_norm=False,
                enable_layer_param_norms=False,
                enable_param_update_norm=False,
                enable_update_to_weight_ratio=False,
                enable_optimizer_state_norms=True,
                enable_activation_norms=False,
                enable_ln_grad_norms=False,
                enable_attention_entropy=False,
                watch_model=False,
                log_every_n_steps=1,
                diagnostics_every_n_steps=1,
                parameter_optimizer_norms_every_n_steps=1,
                val_every_n_steps=0,
                attention_entropy_every_n_steps=10_000,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=64,
            ),
        ),
    )


def _build_sweep_group() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"4-effective-batch-scaled-lr-{timestamp}"


def trial_specs() -> list[tuple[int, int]]:
    accumulation_steps_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    return [
        (acc, MICRO_BATCH_SIZE * acc)
        for acc in accumulation_steps_list
    ]

def run_trial(
    *,
    sweep_group: str,
    accumulation_steps: int,
    base_vocab_size: int,
) -> TrialResult:
    effective_batch_size = MICRO_BATCH_SIZE * accumulation_steps
    run_name = (
        f"{sweep_group}-eff_{effective_batch_size:04d}"
        f"-acc_{accumulation_steps:02d}"
    )
    config = build_config(
        run_name=run_name,
        sweep_group=sweep_group,
        base_vocab_size=base_vocab_size,
        accumulation_steps=accumulation_steps,
    )
    summary_plugin = GradientQualitySummaryPlugin()
    requires_validation = bool(config.train.run_validation)

    try:
        run_result = model_pipeline(config, extra_metric_plugins=[summary_plugin])
        status = "success"
        error_type = None
        error_message = None
        global_step = int(run_result.global_step)
        run_artifact_dir = run_result.run_artifact_dir
        final_train_loss = float(run_result.final_train_loss)
        final_val_loss = (
            float(run_result.final_val_loss) if requires_validation else None
        )
        completed_epochs = int(run_result.completed_epochs)
        epoch_end_validation_ran = bool(run_result.epoch_end_validation_ran)
        if completed_epochs < int(config.train.epochs) or (
            requires_validation and not epoch_end_validation_ran
        ):
            status = "error"
            error_type = "IncompleteEpochOrValidation"
            error_message = (
                "Run did not complete the required workload. "
                f"completed_epochs={completed_epochs}, "
                f"required_epochs={int(config.train.epochs)}, "
                f"requires_validation={requires_validation}, "
                f"epoch_end_validation_ran={epoch_end_validation_ran}"
            )
    except Exception as exc:
        status = "oom" if classify_oom_exception(exc) else "error"
        error_type = exc.__class__.__name__
        error_message = str(exc)
        global_step = None
        run_artifact_dir = None
        final_train_loss = None
        final_val_loss = None
    finally:
        logged_summary = summary_plugin.summary
        clear_runtime_state()

    result = TrialResult(
        accumulation_steps=accumulation_steps,
        effective_batch_size=effective_batch_size,
        run_name=run_name,
        status=status,
        global_step=global_step,
        run_artifact_dir=run_artifact_dir,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        max_peak_memory_gib=logged_summary.max_peak_memory_gib,
        max_peak_reserved_memory_gib=logged_summary.max_peak_reserved_memory_gib,
        avg_step_time_ms=logged_summary.avg_step_time_ms,
        avg_tokens_per_sec=logged_summary.avg_tokens_per_sec,
        avg_global_grad_norm=logged_summary.avg_global_grad_norm,
        avg_adam_elemwise_snr_norm=logged_summary.avg_adam_elemwise_snr_norm,
        avg_gradient_coherence=logged_summary.avg_gradient_coherence,
        min_gradient_coherence=logged_summary.min_gradient_coherence,
        max_gradient_coherence=logged_summary.max_gradient_coherence,
        avg_gradient_coherence_pairs=logged_summary.avg_gradient_coherence_pairs,
        error_type=error_type,
        error_message=error_message,
    )

    print(
        "Trial complete | "
        f"accumulation_steps={result.accumulation_steps} | "
        f"effective_batch_size={result.effective_batch_size} | "
        f"status={result.status} | "
        f"avg_gradient_coherence={result.avg_gradient_coherence}"
    )
    return result


def _to_row_value(value: float | int | str | None) -> float | int | str | None:
    return value


def log_wandb_summary_tables(
    *,
    sweep_group: str,
    trial_results: list[TrialResult],
    summary_stage: str = "final",
    snapshot_id: str | None = None,
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B summary table logging requires the `wandb` package."
        ) from exc

    if summary_stage not in {"partial", "final"}:
        raise ValueError("summary_stage must be one of: partial, final")

    summary_run_name = f"{sweep_group}-summary-{summary_stage}"
    if summary_stage == "partial":
        if snapshot_id is None or not snapshot_id.strip():
            raise ValueError("snapshot_id is required for partial summary stage.")
        summary_run_name = f"{summary_run_name}-{snapshot_id}"

    run = wandb.init(
        project=PROJECT_NAME,
        name=summary_run_name,
        group=sweep_group,
        config={
            "micro_batch_size": MICRO_BATCH_SIZE,
            "measure_training_only": float(1 if MEASURE_TRAINING_ONLY else 0),
            "epochs": EPOCHS,
            "data_fraction": DATA_FRACTION,
            "base_learning_rate": LEARNING_RATE,
            "lr_scaling_mode": "sqrt_if_accumulating",
            "summary_stage": summary_stage,
            "summary_snapshot_id": snapshot_id,
        },
    )
    try:
        trial_columns = [
            "accumulation_steps",
            "effective_batch_size",
            "status",
            "global_step",
            "final_train_loss",
            "final_val_loss",
            "avg_global_grad_norm",
            "avg_adam_elemwise_snr_norm",
            "avg_gradient_coherence",
            "min_gradient_coherence",
            "max_gradient_coherence",
            "avg_gradient_coherence_pairs",
            "max_peak_memory_gib",
            "max_peak_reserved_memory_gib",
            "avg_step_time_ms",
            "avg_tokens_per_sec",
            "run_name",
            "run_artifact_dir",
            "error_type",
        ]
        trial_rows = [
            [
                trial.accumulation_steps,
                trial.effective_batch_size,
                trial.status,
                _to_row_value(trial.global_step),
                _to_row_value(trial.final_train_loss),
                _to_row_value(trial.final_val_loss),
                _to_row_value(trial.avg_global_grad_norm),
                _to_row_value(trial.avg_adam_elemwise_snr_norm),
                _to_row_value(trial.avg_gradient_coherence),
                _to_row_value(trial.min_gradient_coherence),
                _to_row_value(trial.max_gradient_coherence),
                _to_row_value(trial.avg_gradient_coherence_pairs),
                _to_row_value(trial.max_peak_memory_gib),
                _to_row_value(trial.max_peak_reserved_memory_gib),
                _to_row_value(trial.avg_step_time_ms),
                _to_row_value(trial.avg_tokens_per_sec),
                trial.run_name,
                _to_row_value(trial.run_artifact_dir),
                _to_row_value(trial.error_type),
            ]
            for trial in sorted(
                trial_results,
                key=lambda trial: trial.accumulation_steps,
            )
        ]

        run.log(
            {
                "effective_batch_scaled_lr_trials": wandb.Table(
                    columns=trial_columns,
                    data=trial_rows,
                ),
            }
        )
    finally:
        run.finish()


def _print_sweep_summary(trials: list[TrialResult]) -> None:
    success_trials = [trial for trial in trials if trial.status == "success"]
    print(
        "Sweep summary | "
        f"trials={len(trials)} | "
        f"successful_trials={len(success_trials)} | "
        f"micro_batch_size={MICRO_BATCH_SIZE}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires CUDA because memory metrics "
            "(peak_memory_gib / peak_reserved_memory_gib) are CUDA-only."
        )

    vocab_path = (
        PROJECT_ROOT
        / "src"
        / "vocabs"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )
    base_vocab_size = ensure_wikitext_vocab_file(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        vocab_path=vocab_path,
    )

    sweep_group = _build_sweep_group()
    specs = trial_specs()
    print(f"Starting LR-scaled effective-batch sweep group: {sweep_group}")
    print(
        "Effective-batch LR-scaled sweep config | "
        f"micro_batch_size={MICRO_BATCH_SIZE} | "
        f"accumulation_steps={specs[0][0]}-{specs[-1][0]} | "
        f"effective_batch_size=[{MICRO_BATCH_SIZE * specs[0][0]},{MICRO_BATCH_SIZE * specs[-1][0]}] | "
        f"base_learning_rate={LEARNING_RATE} | "
        "lr_scaling=sqrt_if_accumulating"
    )

    trial_results: list[TrialResult] = []

    for accumulation_steps, effective_batch_size in specs:
        print(
            "Starting trial | "
            f"accumulation_steps={accumulation_steps} | "
            f"effective_batch_size={effective_batch_size}"
        )

        result = run_trial(
            sweep_group=sweep_group,
            accumulation_steps=accumulation_steps,
            base_vocab_size=base_vocab_size,
        )
        trial_results.append(result)

        snapshot_id = (
            f"trial_{len(trial_results):03d}-"
            f"acc_{accumulation_steps:02d}"
        )
        try:
            log_wandb_summary_tables(
                sweep_group=sweep_group,
                trial_results=trial_results,
                summary_stage="partial",
                snapshot_id=snapshot_id,
            )
        except Exception as exc:
            print(
                "Warning: failed to persist partial summary checkpoint | "
                f"snapshot_id={snapshot_id} | error={exc}"
            )

    _print_sweep_summary(trial_results)
    log_wandb_summary_tables(
        sweep_group=sweep_group,
        trial_results=trial_results,
        summary_stage="final",
    )

    print("Sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
