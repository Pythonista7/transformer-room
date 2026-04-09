# Baseline Quickstart

Fast path to create and run a new experiment with the modular baseline pipeline.

## 1) Create an experiment file

Create `experiments/baseline/hyperparam_sweeps/my_experiment.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ExperimentConfig,
    RunConfig,
    LocalTextDatasetConfig,
    BPETokenizerConfig,
    BaselineDecoderConfig,
    LRSchedulerChainConfig,
    LRSchedulerStageConfig,
    OptimizerConfig,
    TrainConfig,
    HoldoutSplitConfig,
    LoggingConfig,
    WandbMetricsConfig,
)
from src.train import model_pipeline


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(
            project_name="my-project",
            artifacts_root=str(PROJECT_ROOT / "src" / "models"),
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=250,
        ),
        dataset=LocalTextDatasetConfig(
            path=str(PROJECT_ROOT / "datasets" / "tiny_shakespeare.txt"),
            segment_delimiter="\n\n",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=10_000,
            num_special_tokens=3,
            vocab_path=str(PROJECT_ROOT / "src" / "vocabs" / "my_vocab.txt"),
        ),
        model=BaselineDecoderConfig(
            d_model=128,
            n_heads=8,
            layers=2,
        ),
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=1e-3,
                weight_decay=1e-2,
            ),
            effective_batch_size=128,
            seq_len=128,
            stride=128,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(
            train_fraction=0.9,
            seed=42,
            shuffle=False,
        ),
        logging=LoggingConfig(provider="console"),
    )


def main() -> int:
    result = model_pipeline(build_config())
    print(result.run_artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Batching Semantics

`TrainConfig` now uses explicit batch math with:

- `effective_batch_size`: logical optimizer-step batch size.
- `micro_batch_size`: physical DataLoader batch size.
- `accumulation_steps`: number of micro-batches per optimizer step.

Required equation everywhere:

`effective_batch_size = micro_batch_size * accumulation_steps`

Defaults (non-accumulation behavior):

- `micro_batch_size` defaults to `effective_batch_size`.
- `accumulation_steps` defaults to `1`.

Non-accumulation example:

```python
TrainConfig(
    effective_batch_size=128,
    micro_batch_size=128,
    accumulation_steps=1,
    ...
)
```

Accumulation example:

```python
TrainConfig(
    effective_batch_size=128,
    micro_batch_size=32,
    accumulation_steps=4,
    lr_scaling="sqrt",
    ...
)
```

`train.batch_size` was removed. Any legacy `batch_size=...` usage now fails at constructor time.
When accumulation is active (`effective_batch_size > micro_batch_size`), `lr_scaling="sqrt"` is required.

## LR Scheduler Chains

`TrainConfig.lr_scheduler` supports chained stage schedules with optimizer-step cadence.

```python
TrainConfig(
    ...,
    lr_scheduler=LRSchedulerChainConfig(
        stages=[
            LRSchedulerStageConfig(
                type="linear",
                start_factor=0.1,
                end_factor=1.0,
                steps=500,
            ),
            LRSchedulerStageConfig(
                type="cosine",
                end_factor=0.05,
                steps=None,  # only valid for final stage with known total steps
            ),
        ]
    ),
)
```

Notes:

- Schedulers advance once per optimizer step (never per micro-batch).
- `start_factor` and `end_factor` are multipliers, not absolute LRs.
  Effective LR at a step is `optimizer_lr * factor` (and `optimizer_lr` already includes `lr_scaling` when enabled).
- For unknown total-step runs (for example streaming with `max_steps=None`), every stage must set explicit `steps`.
- If training outlives explicit stages, the final stage LR factor is held constant.

## 2) Run it

Script path style (from repo root):

```bash
.venv/bin/python experiments/baseline/hyperparam_sweeps/my_experiment.py
```

Module style (from repo root):

```bash
.venv/bin/python -m experiments.baseline.hyperparam_sweeps.my_experiment
```

## 3) Inspect outputs

Run artifacts are written under `run.artifacts_root`, usually:

- `src/models/<run_name>/baseline_checkpoint.pt`
- `src/models/<run_name>/baseline_model.pt`
- `src/models/<run_name>/run_config.json`
- `src/models/<run_name>/inference_config.json`
- `src/models/<run_name>/tokenizer/` for Hugging Face tokenizer runs

For `LoggingConfig(provider="wandb", ...)` runs:

- `run.run_name` is required and should be deterministic.
- `run.group_name` may include timestamps for launch grouping, but must not be used to derive `run_name`.
- Checkpoint/model `.pt` files are treated as upload transport files and may be deleted locally after a successful W&B upload.
- `run_config.json` and `inference_config.json` remain on disk.

## Quick knobs to change

- Dataset source:
  - Local text: `LocalTextDatasetConfig(...)`
  - Hugging Face text: `HFTextDatasetConfig(...)`
- Data access mode:
  - `TrainConfig(data_mode="materialized")`
  - `TrainConfig(data_mode="streaming", max_steps=...)`
- Logging:
  - `LoggingConfig(provider="console")` for local iteration
  - `LoggingConfig(provider="local", wandb=WandbMetricsConfig(...))` for local rich metrics in `metrics.jsonl`
  - `LoggingConfig(provider="wandb", wandb=WandbMetricsConfig(...))` for experiment tracking
  - For W&B runs, set a stable `run.run_name` such as `wikitext2-gpt2-lr1e-4-bs20`
- Model size:
  - `d_model`, `n_heads`, `layers`
- Optimizer:
  - `OptimizerConfig(name="adam" | "adamw" | "sgd", learning_rate=..., weight_decay=...)`
  - When using accumulation (`effective_batch_size > micro_batch_size`), set `TrainConfig(lr_scaling="sqrt")` to enable required LR scaling.
- LR scheduler:
  - `TrainConfig(lr_scheduler=LRSchedulerChainConfig(stages=[...]))`
  - Stage types: `"linear"` and `"cosine"`
- Tokenizer size:
  - `base_vocab_size`

## Streaming HF path

Streaming is supported only with:

- `HFTextDatasetConfig(...)`
- `HFPretrainedTokenizerConfig(...)`
- `PreSplitConfig()`
- `TrainConfig(data_mode="streaming", max_steps=...)` or epoch-based streaming with `max_steps=None`

Example:

```python
dataset=HFTextDatasetConfig(
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-2-v1",
    split="train",
    validation_split="validation",
    text_field="text",
),
tokenizer=HFPretrainedTokenizerConfig(
    pretrained_name_or_path="gpt2",
),
train=TrainConfig(
    effective_batch_size=64,
    seq_len=1024,
    stride=1024,
    data_mode="streaming",
    max_steps=2000,
),
split=PreSplitConfig(),
```

## If something fails

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

For full architecture and adapter extension guide, see `src/docs/Readme.md`.

## W&B metrics quick example

```python
run=RunConfig(
    project_name="my-project",
    run_name="my-dataset-baseline-v1",
),
logging=LoggingConfig(
    provider="wandb",
    wandb=WandbMetricsConfig(
        enable_train_loss_vs_tokens=True,
        enable_val_loss_vs_tokens=True,
        enable_perplexity=True,
        enable_bits_per_byte=True,
        enable_step_time=True,
        enable_peak_memory=True,
        enable_global_grad_norm=True,
        enable_activation_norms=True,
        enable_ln_grad_norms=True,
        enable_attention_entropy=True,
        watch_model=False,
        log_every_n_steps=10,
        diagnostics_every_n_steps=50,
        parameter_optimizer_norms_every_n_steps=None,  # defaults to diagnostics cadence
        val_every_n_steps=250,
        attention_entropy_every_n_steps=200,
        attention_entropy_head_cap=2,
        attention_entropy_token_cap=128,
    ),
)
```

If a W&B run starts with a `run_name` that already has a remote checkpoint, the CLI will either:

1. Resume the existing lineage from the latest remote checkpoint.
2. Ask for a manual suffix and start a new lineage such as `my-dataset-baseline-v1-rerun1`.

When disk space is tight, set `WANDB_DATA_DIR` to a workspace-backed directory so W&B staging does not hit a small root volume.
