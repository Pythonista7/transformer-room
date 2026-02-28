# Baseline Quickstart

Fast path to create and run a new experiment with the modular baseline pipeline.

## 1) Create an experiment file

Create `baseline/experiments/my_experiment.py`:

```python
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.config import (
    ExperimentConfig,
    RunConfig,
    LocalTextDatasetConfig,
    BPETokenizerConfig,
    BaselineDecoderConfig,
    TrainConfig,
    HoldoutSplitConfig,
    LoggingConfig,
    WandbMetricsConfig,
)
from baseline.train import model_pipeline


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(
            project_name="my-project",
            artifacts_root=str(PROJECT_ROOT / "baseline" / "models"),
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
            vocab_path=str(PROJECT_ROOT / "baseline" / "tokenizers" / "my_vocab.txt"),
        ),
        model=BaselineDecoderConfig(
            d_model=128,
            n_heads=8,
            layers=2,
        ),
        train=TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            batch_size=128,
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

## 2) Run it

From repo root:

```bash
.venv/bin/python baseline/experiments/my_experiment.py
```

From inside `baseline/`:

```bash
../.venv/bin/python experiments/my_experiment.py
```

## 3) Inspect outputs

Run artifacts are written under `run.artifacts_root`, usually:

- `baseline/models/<run_name>/baseline_checkpoint.pt`
- `baseline/models/<run_name>/baseline_model.pt`
- `baseline/models/<run_name>/run_config.json`
- `baseline/models/<run_name>/inference_config.json`

## Quick knobs to change

- Dataset source:
  - Local text: `LocalTextDatasetConfig(...)`
  - Hugging Face text: `HFTextDatasetConfig(...)`
- Logging:
  - `LoggingConfig(provider="console")` for local iteration
  - `LoggingConfig(provider="wandb", wandb=WandbMetricsConfig(...))` for experiment tracking
- Model size:
  - `d_model`, `n_heads`, `layers`
- Tokenizer size:
  - `base_vocab_size`

## If something fails

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

For full architecture and adapter extension guide, see `baseline/docs/Readme.md`.

## W&B metrics quick example

```python
logging=LoggingConfig(
    provider="wandb",
    wandb=WandbMetricsConfig(
        enable_train_loss_vs_tokens=True,
        enable_val_loss_vs_tokens=True,
        enable_perplexity=True,
        enable_step_time=True,
        enable_peak_memory=True,
        enable_global_grad_norm=True,
        enable_activation_norms=True,
        enable_ln_grad_norms=True,
        enable_attention_entropy=True,
        watch_model=False,
        log_every_n_steps=10,
        diagnostics_every_n_steps=50,
        val_every_n_steps=250,
        attention_entropy_every_n_steps=200,
        attention_entropy_head_cap=2,
        attention_entropy_token_cap=128,
    ),
)
```
