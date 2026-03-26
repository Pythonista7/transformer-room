from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from src.training import runtime as training_runtime
from src.training import wikitext as training_wikitext
from src.training.metrics import BaseMetricPlugin, StepMetricsContext
from src.training.metrics.plugins.layer_grad_norm import (
    compute_layer_grad_norms,
    get_sampled_decoder_layer_indices,
)

PROJECT_NAME = "transformer-room-baseline"
RUN_GROUP = "pre-vs-post-layer-norm"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"
SUMMARY_ROOT = PROJECT_ROOT / "artifacts" / "plots" / "pre_vs_post_layer_norm"

D_MODEL = 128
DEEP_D_MODEL = 512
N_HEADS = 8
SHALLOW_LAYERS = 4
DEEP_LAYERS = 12
EFFECTIVE_BATCH_SIZE = 32
SEQ_LEN = 128
STRIDE = 128
EPOCHS = 1
DATA_FRACTION = 1.0
TRAIN_FRACTION = 0.9
SEED = 42

LEARNING_RATE = 1e-3
LR_WARMUP_STEPS = 500
LR_WARMUP_START_FACTOR = 1e-4
LAYER_GRAD_STRIDE = 1
LAYER_GRAD_EVERY_N_STEPS = 10


@dataclass(frozen=True, slots=True)
class VariantSpec:
    config_name: str
    run_name: str
    n_layers: int
    d_model: int
    norm_position: str
    warmup_steps: int


@dataclass(frozen=True, slots=True)
class LayerGradRecord:
    step: int
    layer_index: int
    grad_norm: float


@dataclass(slots=True)
class TrialResult:
    spec: VariantSpec
    run_artifact_dir: str
    global_step: int
    final_train_loss: float
    final_val_loss: float
    final_val_perplexity: float
    layer_grad_records: list[LayerGradRecord]


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        config_name="post-ln-shallow",
        run_name="post-ln-shallow-warmup0",
        n_layers=SHALLOW_LAYERS,
        d_model=D_MODEL,
        norm_position="post",
        warmup_steps=0,
    ),
    VariantSpec(
        config_name="post-ln-shallow",
        run_name=f"post-ln-shallow-warmup{LR_WARMUP_STEPS}",
        n_layers=SHALLOW_LAYERS,
        d_model=D_MODEL,
        norm_position="post",
        warmup_steps=LR_WARMUP_STEPS,
    ),
    VariantSpec(
        config_name="post-ln-deep",
        run_name="post-ln-deep-warmup0",
        n_layers=DEEP_LAYERS,
        d_model=DEEP_D_MODEL,
        norm_position="post",
        warmup_steps=0,
    ),
    VariantSpec(
        config_name="post-ln-deep",
        run_name=f"post-ln-deep-warmup{LR_WARMUP_STEPS}",
        n_layers=DEEP_LAYERS,
        d_model=DEEP_D_MODEL,
        norm_position="post",
        warmup_steps=LR_WARMUP_STEPS,
    ),
    VariantSpec(
        config_name="pre-ln-shallow",
        run_name="pre-ln-shallow-warmup0",
        n_layers=SHALLOW_LAYERS,
        d_model=D_MODEL,
        norm_position="pre",
        warmup_steps=0,
    ),
    VariantSpec(
        config_name="pre-ln-deep",
        run_name="pre-ln-deep-warmup0",
        n_layers=DEEP_LAYERS,
        d_model=DEEP_D_MODEL,
        norm_position="pre",
        warmup_steps=0,
    ),
)


class LayerGradNormCollector(BaseMetricPlugin):
    name = "layer_grad_norm_collector"

    def __init__(self, *, stride: int) -> None:
        self._stride = int(stride)
        self._layer_indices: tuple[int, ...] = ()
        self.records: list[LayerGradRecord] = []

    def after_backward(self, ctx: StepMetricsContext) -> None:
        if not ctx.schedule.should_log_layer_grad_norms:
            return
        model = ctx.model
        if not isinstance(model, torch.nn.Module):
            return
        if not self._layer_indices:
            self._layer_indices = get_sampled_decoder_layer_indices(
                model,
                stride=self._stride,
            )
        metrics = compute_layer_grad_norms(model, self._layer_indices)
        for metric_key, grad_norm in metrics.items():
            layer_index = int(metric_key.rsplit("_", maxsplit=1)[-1])
            self.records.append(
                LayerGradRecord(
                    step=int(ctx.next_global_step),
                    layer_index=layer_index,
                    grad_norm=float(grad_norm),
                )
            )


def _build_variant_config(
    *,
    base_vocab_size: int,
    vocab_path: Path,
    spec: VariantSpec,
) -> ExperimentConfig:
    warmup_start_factor = (
        LR_WARMUP_START_FACTOR if spec.warmup_steps > 0 else 0.0
    )
    return ExperimentConfig(
        run=RunConfig(
            project_name=PROJECT_NAME,
            run_name=spec.run_name,
            group_name=RUN_GROUP,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            persist_local_artifacts=False,
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=0,
            seed=SEED,
            use_torch_compile=True
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
            d_model=spec.d_model,
            n_heads=N_HEADS,
            layers=spec.n_layers,
            norm_placement=spec.norm_position,
        ),
        train=TrainConfig(
            epochs=EPOCHS,
            optimizer=OptimizerConfig(name="adam", learning_rate=LEARNING_RATE),
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            accumulation_steps=1,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_fraction=DATA_FRACTION,
            lr_warmup_steps=spec.warmup_steps,
            lr_warmup_start_factor=warmup_start_factor,
        ),
        split=HoldoutSplitConfig(
            train_fraction=TRAIN_FRACTION,
            seed=SEED,
            shuffle=False,
        ),
        logging=LoggingConfig(
            provider="wandb",
            enable_artifact_io=False,
            wandb=WandbMetricsConfig(
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=True,
                enable_perplexity=True,
                enable_bits_per_byte=True,
                enable_step_time=True,
                enable_peak_memory=True,
                enable_global_grad_norm=True,
                enable_layer_grad_norms=True,
                enable_global_param_norm=False,
                enable_layer_param_norms=False,
                enable_param_update_norm=False,
                enable_update_to_weight_ratio=False,
                enable_optimizer_state_norms=False,
                enable_activation_norms=False,
                enable_ln_grad_norms=False,
                enable_attention_entropy=False,
                watch_model=False,
                log_every_n_steps=10,
                diagnostics_every_n_steps=10,
                layer_grad_norm_stride=LAYER_GRAD_STRIDE,
                layer_grad_norms_every_n_steps=LAYER_GRAD_EVERY_N_STEPS,
                val_every_n_steps=250,
                attention_entropy_every_n_steps=250,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=SEQ_LEN,
            ),
        ),
    )


def build_variant_configs() -> list[tuple[VariantSpec, ExperimentConfig]]:
    vocab_path = PROJECT_ROOT / "src" / "vocabs" / "wikitext2_v1_hf_vocab_bpe.txt"
    base_vocab_size = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        vocab_path=vocab_path,
    )
    return [
        (
            spec,
            _build_variant_config(
                base_vocab_size=base_vocab_size,
                vocab_path=vocab_path,
                spec=spec,
            ),
        )
        for spec in VARIANT_SPECS
    ]


def _variant_label(spec: VariantSpec) -> str:
    return (
        f"{spec.config_name} | warmup={spec.warmup_steps} | "
        f"layers={spec.n_layers} | d_model={spec.d_model} | "
        f"norm={spec.norm_position}"
    )


def _summarize_layer_grad_records(
    records: list[LayerGradRecord],
) -> list[dict[str, float | int]]:
    by_layer: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_layer[record.layer_index].append(float(record.grad_norm))

    summary_rows: list[dict[str, float | int]] = []
    for layer_index in sorted(by_layer):
        values = by_layer[layer_index]
        summary_rows.append(
            {
                "layer_index": int(layer_index),
                "mean_layer_grad_norm": float(mean(values)),
                "min_layer_grad_norm": float(min(values)),
                "max_layer_grad_norm": float(max(values)),
                "logged_steps": int(len(values)),
            }
        )
    return summary_rows


def _write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_trial_rows(trial_results: list[TrialResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trial in trial_results:
        logged_steps = len({record.step for record in trial.layer_grad_records})
        rows.append(
            {
                "config": trial.spec.config_name,
                "run_name": trial.spec.run_name,
                "n_layers": trial.spec.n_layers,
                "d_model": trial.spec.d_model,
                "norm_position": trial.spec.norm_position,
                "warmup_steps": trial.spec.warmup_steps,
                "global_step": trial.global_step,
                "final_train_loss": trial.final_train_loss,
                "final_val_loss": trial.final_val_loss,
                "final_val_perplexity": trial.final_val_perplexity,
                "logged_layer_grad_steps": logged_steps,
                "run_artifact_dir": trial.run_artifact_dir,
            }
        )
    return rows


def _build_layer_grad_rows(trial_results: list[TrialResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trial in trial_results:
        for record in trial.layer_grad_records:
            rows.append(
                {
                    "config": trial.spec.config_name,
                    "run_name": trial.spec.run_name,
                    "n_layers": trial.spec.n_layers,
                    "d_model": trial.spec.d_model,
                    "norm_position": trial.spec.norm_position,
                    "warmup_steps": trial.spec.warmup_steps,
                    "step": record.step,
                    "layer_index": record.layer_index,
                    "layer_grad_norm": record.grad_norm,
                }
            )
    return rows


def _build_layer_grad_summary_rows(
    trial_results: list[TrialResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trial in trial_results:
        for summary_row in _summarize_layer_grad_records(trial.layer_grad_records):
            rows.append(
                {
                    "config": trial.spec.config_name,
                    "run_name": trial.spec.run_name,
                    "n_layers": trial.spec.n_layers,
                    "d_model": trial.spec.d_model,
                    "norm_position": trial.spec.norm_position,
                    "warmup_steps": trial.spec.warmup_steps,
                    **summary_row,
                }
            )
    return rows


def _plot_layer_grad_norms(
    trial_results: list[TrialResult],
    *,
    output_path: Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Skipping layer grad norm plot: matplotlib is not installed. "
            "CSV summaries will still be written."
        )
        return None

    figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    axes_by_depth = {
        SHALLOW_LAYERS: axes[0],
        DEEP_LAYERS: axes[1],
    }

    for n_layers, axis in axes_by_depth.items():
        axis.set_title(f"{n_layers} layers")
        axis.set_xlabel("Layer index")
        axis.set_ylabel("Average layer grad norm")
        axis.grid(alpha=0.3)
        plotted_any = False

        for trial in trial_results:
            if trial.spec.n_layers != n_layers:
                continue

            summary_rows = _summarize_layer_grad_records(trial.layer_grad_records)
            if not summary_rows:
                continue

            layer_indices = [
                int(row["layer_index"])
                for row in summary_rows
            ]
            grad_norms = [
                float(row["mean_layer_grad_norm"])
                for row in summary_rows
            ]
            axis.plot(
                layer_indices,
                grad_norms,
                marker="o",
                linewidth=2.0,
                label=f"{trial.spec.norm_position}-ln, warmup={trial.spec.warmup_steps}",
            )
            plotted_any = True

        if plotted_any:
            axis.legend()
        else:
            axis.text(
                0.5,
                0.5,
                "No layer grad norms logged",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    figure.suptitle("Average layer grad norm vs layer index")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _log_summary_to_wandb(
    *,
    summary_id: str,
    trial_rows: list[dict[str, object]],
    layer_grad_summary_rows: list[dict[str, object]],
    plot_paths: list[Path],
) -> None:
    try:
        import wandb
    except ImportError:
        print("Skipping W&B summary logging: wandb is not installed.")
        return

    run = wandb.init(
        project=PROJECT_NAME,
        name=f"{RUN_GROUP}-summary-{summary_id}",
        group=RUN_GROUP,
        config={
            "summary_id": summary_id,
            "variant_count": len(trial_rows),
            "layer_grad_stride": LAYER_GRAD_STRIDE,
            "layer_grad_every_n_steps": LAYER_GRAD_EVERY_N_STEPS,
            "step": "layer_grad_norm_summary",
        },
    )
    try:
        payload: dict[str, object] = {
            "pre_post_layer_norm_trials": wandb.Table(
                columns=[
                    "config",
                    "run_name",
                    "n_layers",
                    "d_model",
                    "norm_position",
                    "warmup_steps",
                    "global_step",
                    "final_train_loss",
                    "final_val_loss",
                    "final_val_perplexity",
                    "logged_layer_grad_steps",
                    "run_artifact_dir",
                ],
                data=[
                    [row[column] for column in (
                        "config",
                        "run_name",
                        "n_layers",
                        "d_model",
                        "norm_position",
                        "warmup_steps",
                        "global_step",
                        "final_train_loss",
                        "final_val_loss",
                        "final_val_perplexity",
                        "logged_layer_grad_steps",
                        "run_artifact_dir",
                    )]
                    for row in trial_rows
                ],
            ),
            "pre_post_layer_norm_grad_summary": wandb.Table(
                columns=[
                    "config",
                    "run_name",
                    "n_layers",
                    "d_model",
                    "norm_position",
                    "warmup_steps",
                    "layer_index",
                    "mean_layer_grad_norm",
                    "min_layer_grad_norm",
                    "max_layer_grad_norm",
                    "logged_steps",
                ],
                data=[
                    [row[column] for column in (
                        "config",
                        "run_name",
                        "n_layers",
                        "d_model",
                        "norm_position",
                        "warmup_steps",
                        "layer_index",
                        "mean_layer_grad_norm",
                        "min_layer_grad_norm",
                        "max_layer_grad_norm",
                        "logged_steps",
                    )]
                    for row in layer_grad_summary_rows
                ],
            ),
        }
        for plot_path in plot_paths:
            payload[plot_path.stem] = wandb.Image(str(plot_path))
        run.log(payload)
    finally:
        run.finish()


def _write_summary_artifacts(trial_results: list[TrialResult]) -> Path:
    summary_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    summary_dir = SUMMARY_ROOT / summary_id
    trial_rows = _build_trial_rows(trial_results)
    layer_grad_rows = _build_layer_grad_rows(trial_results)
    layer_grad_summary_rows = _build_layer_grad_summary_rows(trial_results)

    _write_csv(
        summary_dir / "trial_results.csv",
        fieldnames=[
            "config",
            "run_name",
            "n_layers",
            "d_model",
            "norm_position",
            "warmup_steps",
            "global_step",
            "final_train_loss",
            "final_val_loss",
            "final_val_perplexity",
            "logged_layer_grad_steps",
            "run_artifact_dir",
        ],
        rows=trial_rows,
    )
    _write_csv(
        summary_dir / "layer_grad_norm_points.csv",
        fieldnames=[
            "config",
            "run_name",
            "n_layers",
            "d_model",
            "norm_position",
            "warmup_steps",
            "step",
            "layer_index",
            "layer_grad_norm",
        ],
        rows=layer_grad_rows,
    )
    _write_csv(
        summary_dir / "layer_grad_norm_summary.csv",
        fieldnames=[
            "config",
            "run_name",
            "n_layers",
            "d_model",
            "norm_position",
            "warmup_steps",
            "layer_index",
            "mean_layer_grad_norm",
            "min_layer_grad_norm",
            "max_layer_grad_norm",
            "logged_steps",
        ],
        rows=layer_grad_summary_rows,
    )

    plot_paths: list[Path] = []
    plot_path = _plot_layer_grad_norms(
        trial_results,
        output_path=summary_dir / "avg_layer_grad_norm_vs_layer_index.png",
    )
    if plot_path is not None:
        plot_paths.append(plot_path)

    try:
        _log_summary_to_wandb(
            summary_id=summary_id,
            trial_rows=trial_rows,
            layer_grad_summary_rows=layer_grad_summary_rows,
            plot_paths=plot_paths,
        )
    except Exception as exc:
        print(f"Warning: failed to log W&B summary artifacts: {exc}")

    return summary_dir


def main() -> int:
    trial_results: list[TrialResult] = []

    for spec, config in build_variant_configs():
        print(f"Running variant: {_variant_label(spec)}")
        collector = LayerGradNormCollector(stride=LAYER_GRAD_STRIDE)
        try:
            result = model_pipeline(
                config,
                extra_metric_plugins=(collector,),
            )
            trial_results.append(
                TrialResult(
                    spec=spec,
                    run_artifact_dir=result.run_artifact_dir,
                    global_step=result.global_step,
                    final_train_loss=result.final_train_loss,
                    final_val_loss=result.final_val_loss,
                    final_val_perplexity=result.final_val_perplexity,
                    layer_grad_records=list(collector.records),
                )
            )
            print(
                "Completed variant | "
                f"run_name={spec.run_name} | "
                f"run_dir={result.run_artifact_dir} | "
                f"final_val_loss={result.final_val_loss:.6f} | "
                f"final_val_ppl={result.final_val_perplexity:.6f} | "
                f"layer_grad_points={len(collector.records)}"
            )
        finally:
            training_runtime.clear_runtime_state()

    summary_dir = _write_summary_artifacts(trial_results)
    print(f"Summary artifacts written to: {summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
