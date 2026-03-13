from __future__ import annotations

import gc
import importlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable

import torch

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
from src.training.metrics import BaseMetricPlugin, MetricPayload, StepMetricsContext

PROJECT_NAME = "transformer-room-baseline"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"
SEQ_LEN = 1024
STRIDE = 1024
BUDGETS: tuple[float | None, ...] = (None, 0.4, 0.8, 1.0)
MIN_MICRO_BATCH = 16
MAX_MICRO_BATCH = 256
EPOCHS = 1
DATA_FRACTION = 0.1
SEED = 42


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    activation_memory_budget: float | None

    @property
    def slug(self) -> str:
        if self.activation_memory_budget is None:
            return "none"
        text = f"{self.activation_memory_budget:.3f}".rstrip("0").rstrip(".")
        if "." not in text:
            text = f"{text}.0"
        return text.replace(".", "p")

    @property
    def label(self) -> str:
        if self.activation_memory_budget is None:
            return "none"
        return f"{self.activation_memory_budget:g}"


@dataclass(frozen=True, slots=True)
class LoggedStepSummary:
    max_peak_memory_gib: float | None
    max_peak_reserved_memory_gib: float | None
    avg_step_time_ms: float | None
    avg_tokens_per_sec: float | None


@dataclass(frozen=True, slots=True)
class TrialResult:
    budget_label: str
    activation_memory_budget: float | None
    micro_batch_size: int
    run_name: str
    status: str
    global_step: int | None = None
    run_artifact_dir: str | None = None
    max_peak_memory_gib: float | None = None
    max_peak_reserved_memory_gib: float | None = None
    avg_step_time_ms: float | None = None
    avg_tokens_per_sec: float | None = None
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
        )

    peak_memory_values = [
        float(metrics["peak_memory_gib"])
        for metrics in step_metrics.values()
        if "peak_memory_gib" in metrics
    ]
    peak_reserved_values = [
        float(metrics["peak_reserved_memory_gib"])
        for metrics in step_metrics.values()
        if "peak_reserved_memory_gib" in metrics
    ]
    step_times = [
        float(metrics["step_time_ms"])
        for metrics in step_metrics.values()
        if "step_time_ms" in metrics
    ]

    return LoggedStepSummary(
        max_peak_memory_gib=max(peak_memory_values) if peak_memory_values else None,
        max_peak_reserved_memory_gib=max(peak_reserved_values)
        if peak_reserved_values
        else None,
        avg_step_time_ms=float(mean(step_times)) if step_times else None,
        avg_tokens_per_sec=compute_avg_tokens_per_sec(logged_entries),
    )


class MemoryFrontierSummaryPlugin(BaseMetricPlugin):
    name = "memory_frontier_summary"

    def __init__(self) -> None:
        self.logged_entries: list[tuple[int | None, dict[str, float]]] = []

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        if not ctx.include_in_perf_aggregates:
            return
        payload: MetricPayload = {}
        payload["tokens_seen_train"] = float(ctx.tokens_seen_train)
        if ctx.step_time_ms is not None:
            payload["step_time_ms"] = float(ctx.step_time_ms)
        if ctx.peak_memory_gib is not None:
            payload["peak_memory_gib"] = float(ctx.peak_memory_gib)
        if ctx.peak_reserved_memory_gib is not None:
            payload["peak_reserved_memory_gib"] = float(ctx.peak_reserved_memory_gib)
        self.logged_entries.append((ctx.global_step, payload))

    @property
    def summary(self) -> LoggedStepSummary:
        return summarize_logged_steps(self.logged_entries)


def build_config(
    *,
    run_name: str,
    sweep_group: str,
    base_vocab_size: int,
    micro_batch_size: int,
    activation_memory_budget: float | None,
) -> ExperimentConfig:
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
            activation_memory_budget=activation_memory_budget,
            compile_warmup_steps=10,
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
                learning_rate=3e-5,
                weight_decay=0.0,
            ),
            effective_batch_size=micro_batch_size,
            micro_batch_size=micro_batch_size,
            accumulation_steps=1,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_fraction=DATA_FRACTION,
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
                enable_val_loss_vs_tokens=False,
                enable_perplexity=False,
                enable_step_time=True,
                enable_peak_memory=True,
                enable_global_grad_norm=False,
                enable_global_param_norm=False,
                enable_layer_param_norms=False,
                enable_param_update_norm=False,
                enable_update_to_weight_ratio=False,
                enable_optimizer_state_norms=False,
                enable_activation_norms=False,
                enable_ln_grad_norms=False,
                enable_attention_entropy=False,
                watch_model=False,
                log_every_n_steps=1,
                diagnostics_every_n_steps=10_000,
                parameter_optimizer_norms_every_n_steps=10_000,
                val_every_n_steps=0,
                attention_entropy_every_n_steps=10_000,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=64,
            ),
        ),
    )


def preflight_activation_memory_budget_api(budgets: tuple[float | None, ...]) -> None:
    if all(budget is None for budget in budgets):
        return

    dynamo_module = getattr(torch, "_dynamo", None)
    if dynamo_module is None:
        raise RuntimeError(
            "Budgeted variants were requested, but torch._dynamo is unavailable."
        )
    _ = dynamo_module

    functorch_module = getattr(torch, "_functorch", None)
    functorch_config = getattr(functorch_module, "config", None)
    if functorch_config is None or not hasattr(functorch_config, "activation_memory_budget"):
        raise RuntimeError(
            "Budgeted variants were requested, but "
            "torch._functorch.config.activation_memory_budget is unavailable."
        )


def _build_sweep_group() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"memory-frontier-step1-compile-budget-{timestamp}"


def run_trial(
    *,
    sweep_group: str,
    budget_spec: BudgetSpec,
    micro_batch_size: int,
    base_vocab_size: int,
) -> TrialResult:
    run_name = (
        f"{sweep_group}-budget_{budget_spec.slug}-mb_{micro_batch_size}"
    )
    config = build_config(
        run_name=run_name,
        sweep_group=sweep_group,
        base_vocab_size=base_vocab_size,
        micro_batch_size=micro_batch_size,
        activation_memory_budget=budget_spec.activation_memory_budget,
    )
    summary_plugin = MemoryFrontierSummaryPlugin()

    try:
        run_result = model_pipeline(config, extra_metric_plugins=[summary_plugin])
        status = "success"
        error_type = None
        error_message = None
        global_step = int(run_result.global_step)
        run_artifact_dir = run_result.run_artifact_dir
        completed_epochs = int(run_result.completed_epochs)
        epoch_end_validation_ran = bool(run_result.epoch_end_validation_ran)
        if (
            completed_epochs < int(config.train.epochs)
            or not epoch_end_validation_ran
        ):
            status = "error"
            error_type = "IncompleteEpochOrValidation"
            error_message = (
                "Run did not complete all epochs and epoch-end validation. "
                f"completed_epochs={completed_epochs}, "
                f"required_epochs={int(config.train.epochs)}, "
                f"epoch_end_validation_ran={epoch_end_validation_ran}"
            )
    except Exception as exc:
        status = "oom" if classify_oom_exception(exc) else "error"
        error_type = exc.__class__.__name__
        error_message = str(exc)
        global_step = None
        run_artifact_dir = None
    finally:
        logged_summary = summary_plugin.summary
        clear_runtime_state()

    result = TrialResult(
        budget_label=budget_spec.label,
        activation_memory_budget=budget_spec.activation_memory_budget,
        micro_batch_size=micro_batch_size,
        run_name=run_name,
        status=status,
        global_step=global_step,
        run_artifact_dir=run_artifact_dir,
        max_peak_memory_gib=logged_summary.max_peak_memory_gib,
        max_peak_reserved_memory_gib=logged_summary.max_peak_reserved_memory_gib,
        avg_step_time_ms=logged_summary.avg_step_time_ms,
        avg_tokens_per_sec=logged_summary.avg_tokens_per_sec,
        error_type=error_type,
        error_message=error_message,
    )

    print(
        "Trial complete | "
        f"budget={result.budget_label} | "
        f"micro_batch_size={result.micro_batch_size} | "
        f"status={result.status} | "
        f"max_peak_reserved_memory_gib={result.max_peak_reserved_memory_gib} | "
        f"avg_tokens_per_sec={result.avg_tokens_per_sec}"
    )
    return result


def adaptive_find_frontier(
    *,
    min_micro_batch: int,
    max_micro_batch: int,
    trial_runner: Callable[[int], TrialResult],
) -> tuple[int | None, list[TrialResult]]:
    if min_micro_batch <= 0:
        raise ValueError("min_micro_batch must be > 0.")
    if max_micro_batch < min_micro_batch:
        raise ValueError("max_micro_batch must be >= min_micro_batch.")

    attempts_by_mb: dict[int, TrialResult] = {}

    def run_once(micro_batch_size: int) -> TrialResult:
        cached = attempts_by_mb.get(micro_batch_size)
        if cached is not None:
            return cached
        result = trial_runner(micro_batch_size)
        attempts_by_mb[micro_batch_size] = result
        return result

    first = run_once(min_micro_batch)
    if first.status == "oom":
        return None, list(attempts_by_mb.values())
    if first.status != "success":
        raise RuntimeError(
            "Search aborted because minimum micro-batch trial failed with "
            f"status={first.status}: {first.error_type}: {first.error_message}"
        )

    low = min_micro_batch
    high: int | None = None
    probe = min_micro_batch * 2

    while probe <= max_micro_batch:
        trial = run_once(probe)
        # Frontier classification is strictly status-based; memory metrics are diagnostic.
        if trial.status == "success":
            low = probe
            probe *= 2
            continue
        if trial.status == "oom":
            high = probe
            break
        raise RuntimeError(
            "Search aborted because trial failed with "
            f"status={trial.status}: {trial.error_type}: {trial.error_message}"
        )

    if high is None:
        if low == max_micro_batch:
            return low, list(attempts_by_mb.values())
        max_trial = run_once(max_micro_batch)
        if max_trial.status == "success":
            return max_micro_batch, list(attempts_by_mb.values())
        if max_trial.status == "oom":
            high = max_micro_batch
        else:
            raise RuntimeError(
                "Search aborted because max-boundary trial failed with "
                f"status={max_trial.status}: {max_trial.error_type}: {max_trial.error_message}"
            )

    while high is not None and low + 1 < high:
        mid = (low + high) // 2
        trial = run_once(mid)
        if trial.status == "success":
            low = mid
            continue
        if trial.status == "oom":
            high = mid
            continue
        raise RuntimeError(
            "Search aborted during binary search because trial failed with "
            f"status={trial.status}: {trial.error_type}: {trial.error_message}"
        )

    return low, list(attempts_by_mb.values())


def _to_row_value(value: float | int | str | None) -> float | int | str | None:
    return value


def log_wandb_summary_tables(
    *,
    sweep_group: str,
    trial_results: list[TrialResult],
    frontiers: dict[str, int | None],
) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B summary table logging requires the `wandb` package."
        ) from exc

    summary_run_name = f"{sweep_group}-summary"
    run = wandb.init(
        project=PROJECT_NAME,
        name=summary_run_name,
        group=sweep_group,
        config={
            "budgets": [spec.label for spec in (BudgetSpec(b) for b in BUDGETS)],
            "min_micro_batch": MIN_MICRO_BATCH,
            "max_micro_batch": MAX_MICRO_BATCH,
            "epochs": EPOCHS,
            "data_fraction": DATA_FRACTION,
            "step": "memory_frontier",
        },
    )
    try:
        trial_columns = [
            "budget",
            "micro_batch_size",
            "status",
            "global_step",
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
                trial.budget_label,
                trial.micro_batch_size,
                trial.status,
                _to_row_value(trial.global_step),
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
                key=lambda trial: (trial.budget_label, trial.micro_batch_size),
            )
        ]
        frontier_columns = [
            "budget",
            "frontier_micro_batch_size",
        ]
        frontier_rows = [
            [budget_label, _to_row_value(frontier_mb)]
            for budget_label, frontier_mb in frontiers.items()
        ]

        run.log(
            {
                "memory_frontier_trials": wandb.Table(
                    columns=trial_columns,
                    data=trial_rows,
                ),
                "memory_frontier_frontiers": wandb.Table(
                    columns=frontier_columns,
                    data=frontier_rows,
                ),
            }
        )
    finally:
        run.finish()


def _print_frontier_summary(
    budget_spec: BudgetSpec,
    frontier: int | None,
    trials: list[TrialResult],
) -> None:
    success_trials = [trial for trial in trials if trial.status == "success"]
    best_success = (
        max(success_trials, key=lambda trial: trial.micro_batch_size)
        if success_trials
        else None
    )
    print(
        "Budget summary | "
        f"budget={budget_spec.label} | "
        f"frontier_micro_batch={frontier} | "
        "frontier_rule=status_only | "
        f"trials={len(trials)} | "
        f"best_peak_reserved_memory_gib={None if best_success is None else best_success.max_peak_reserved_memory_gib} | "
        f"best_avg_tokens_per_sec={None if best_success is None else best_success.avg_tokens_per_sec}"
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires CUDA because memory frontier metrics "
            "(peak_memory_gib / peak_reserved_memory_gib) are CUDA-only."
        )

    preflight_activation_memory_budget_api(BUDGETS)

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
    print(f"Starting memory-frontier sweep group: {sweep_group}")

    all_trial_results: list[TrialResult] = []
    frontiers: dict[str, int | None] = {}

    for budget_value in BUDGETS:
        budget_spec = BudgetSpec(activation_memory_budget=budget_value)
        print(
            "Starting budget sweep | "
            f"budget={budget_spec.label} | "
            f"min_micro_batch={MIN_MICRO_BATCH} | "
            f"max_micro_batch={MAX_MICRO_BATCH}"
        )

        frontier, trials = adaptive_find_frontier(
            min_micro_batch=MIN_MICRO_BATCH,
            max_micro_batch=MAX_MICRO_BATCH,
            trial_runner=lambda micro_batch_size: run_trial(
                sweep_group=sweep_group,
                budget_spec=budget_spec,
                micro_batch_size=micro_batch_size,
                base_vocab_size=base_vocab_size,
            ),
        )

        all_trial_results.extend(trials)
        frontiers[budget_spec.label] = frontier
        _print_frontier_summary(
            budget_spec=budget_spec,
            frontier=frontier,
            trials=trials,
        )

    log_wandb_summary_tables(
        sweep_group=sweep_group,
        trial_results=all_trial_results,
        frontiers=frontiers,
    )

    print("Sweep completed.")
    for budget_label, frontier in frontiers.items():
        print(f"Frontier | budget={budget_label} | micro_batch={frontier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
