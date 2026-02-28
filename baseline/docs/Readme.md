# Baseline Modular Training: Overview and Experiment Guide

## What this baseline is trying to do

The baseline is organized so you can run:

```python
model_pipeline(config: ExperimentConfig)
```

and swap datasets, tokenizers, model variants, split strategy, and logging by changing config selectors (for example `dataset.name`, `model.name`) instead of rewriting trainer code.

The main idea is:

1. Keep training orchestration in one place.
2. Keep component-specific logic in adapters.
3. Keep runtime behavior explicit through typed dataclass config.

---

## Current architecture

### Core pipeline

- `baseline/train.py`
  - Validates config
  - Resolves adapters from registries
  - Builds token stream + dataloaders + model
  - Runs train/eval/checkpoint loop
  - Writes run artifacts (`run_config.json`, `inference_config.json`)

### Typed config

- `baseline/core/config.py`
  - `ExperimentConfig` and section configs:
    - `RunConfig`
    - `DatasetConfig` (`LocalTextDatasetConfig`, `HFTextDatasetConfig`)
    - `TokenizerConfig` (`BPETokenizerConfig`)
    - `ModelConfig` (`BaselineDecoderConfig`)
    - `TrainConfig`
    - `SplitConfig` (`HoldoutSplitConfig`)
    - `LoggingConfig` (`WandbMetricsConfig`)
  - `validate_experiment_config(...)`

### Adapter interfaces

- `baseline/core/types.py`
  - Protocols:
    - `DatasetAdapter`
    - `TokenizerAdapter`
    - `ModelAdapter`
    - `SplitAdapter`
    - `LoggerAdapter` + `LoggerSession`

### Adapter registry

- `baseline/core/registry.py`
  - Global registries and lookup helpers with clear errors.
- `baseline/adapters/__init__.py`
  - Imports built-ins so registration happens on import.

### Built-in adapters

- Dataset: `local_text`, `hf_text` in `baseline/adapters/datasets.py`
- Tokenizer: `bpe` in `baseline/adapters/tokenizers.py`
- Model: `baseline_decoder` in `baseline/adapters/models.py`
- Split: `holdout` in `baseline/adapters/splits.py`
- Logging: `console`, `wandb` in `baseline/adapters/loggers.py`

---

## Directory layout (relevant)

- `baseline/train.py` - orchestrator
- `baseline/core/` - config/types/registry
- `baseline/adapters/` - pluggable implementations
- `baseline/experiments/` - experiment config modules
- `baseline/models/<run_name>/` - run outputs
- `baseline/tokenizers/` - persisted tokenizer vocab files

---

## Running the canonical experiment

From repo root:

```bash
.venv/bin/python baseline/experiments/tiny_shakespeare.py
```

From inside `baseline/`:

```bash
../.venv/bin/python experiments/tiny_shakespeare.py
```

`tiny_shakespeare.py` already bootstraps `sys.path` so package imports work from either location.

---

## Creating a new experiment

1. Add a new file in `baseline/experiments/`, for example `my_experiment.py`.
2. Build and return an `ExperimentConfig`.
3. Call `model_pipeline(config)` in `main()`.

Example skeleton:

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
            path=str(PROJECT_ROOT / "datasets" / "my_dataset.txt"),
            segment_delimiter="\\n\\n",
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
            epochs=3,
            learning_rate=1e-3,
            batch_size=256,
            seq_len=128,
            stride=128,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(
            train_fraction=0.9,
            seed=42,
            shuffle=False,
        ),
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
        ),
    )


def main() -> int:
    model_pipeline(build_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Switching datasets/models quickly

Most experiment changes should be config-only:

- Change dataset type:
  - `LocalTextDatasetConfig(...)` or `HFTextDatasetConfig(...)`
- Change tokenizer params:
  - `base_vocab_size`, `vocab_path`, `num_special_tokens`
- Change model params:
  - `d_model`, `n_heads`, `layers`
- Change logging:
  - `LoggingConfig(provider="console")`
  - `LoggingConfig(provider="wandb", wandb=WandbMetricsConfig(...))`
- Change split behavior:
  - `HoldoutSplitConfig(train_fraction=..., seed=..., shuffle=...)`

---

## Adding a new adapter type

### Add a dataset adapter

1. Implement a class with:
   - `load(self, cfg) -> TextCorpus`
2. Register it:
   - `register_dataset_adapter("your_name", YourAdapter())`
3. Add a config type in `baseline/core/config.py` if needed.
4. Extend `DatasetConfig` union and validation logic.

### Add a tokenizer/model/split/logger adapter

Use the same pattern:

1. Implement the matching protocol from `baseline/core/types.py`
2. Register in `baseline/core/registry.py` via helper
3. Ensure config has a selector name and validation path
4. Make sure `baseline/adapters/__init__.py` imports the module so registration runs

---

## Run artifacts and outputs

Each run writes to a run directory inside `run.artifacts_root`:

- `<run_dir>/<checkpoint_filename>` (default: `baseline_checkpoint.pt`)
- `<run_dir>/<final_model_filename>` (default: `baseline_model.pt`)
- `<run_dir>/run_config.json`
- `<run_dir>/inference_config.json`

`run_config.json` stores the full normalized experiment config.  
`inference_config.json` stores minimal model/tokenizer fields for downstream inference utilities.

---

## W&B metrics configuration

`LoggingConfig` now has a nested `wandb` section:

```python
LoggingConfig(
    provider="wandb",
    wandb=WandbMetricsConfig(...),
)
```

### Metric toggles

- `enable_train_loss_vs_tokens`: logs `train_loss` and `tokens_seen_train`
- `enable_val_loss_vs_tokens`: logs `val_loss` and `tokens_seen_train`
- `enable_perplexity`: logs `train_perplexity`, `train_perplexity_epoch`, `val_perplexity`
- `enable_step_time`: logs `step_time_ms`
- `enable_peak_memory`: logs `peak_memory_gib` (CUDA only)
- `enable_global_grad_norm`: logs `global_grad_norm`
- `enable_activation_norms`: logs `activation_norm_first|middle|last`
- `enable_ln_grad_norms`: logs `ln_weight_grad_norm_first|middle|last` and `ln_bias_grad_norm_first|middle|last`
- `enable_attention_entropy`: logs `attention_entropy_first|middle|last` (sampled)
- `watch_model`: gates `wandb.watch(...)`

### Cadence and sampling controls

- `log_every_n_steps`: cadence for step metrics (loss/tokens/perplexity/time/memory)
- `diagnostics_every_n_steps`: cadence for grad/activation/LN diagnostics
- `val_every_n_steps`: periodic validation cadence (`0` disables periodic val; epoch-end val remains)
- `attention_entropy_every_n_steps`: cadence for entropy collection
- `attention_entropy_head_cap`: number of heads sampled for entropy
- `attention_entropy_token_cap`: number of tokens sampled per axis for entropy

### Overhead guidance

- Keep `watch_model=False` unless you specifically need full parameter/gradient tracking.
- Attention entropy is sampled by `head_cap` and `token_cap` to avoid full `T x T` cost.
- `peak_memory_gib` is only emitted on CUDA runs.

---

## Validation and test notes

- Config validation occurs at the start of `model_pipeline(...)`.
- Registry lookup failures list available adapter names.
- Existing tests (under `tests/`) include:
  - config and registry failures
  - split determinism
  - local dataset empty-file handling
  - CPU smoke training run + artifact checks

Run tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

---

## Practical conventions for new experiments

- Keep experiment files in `baseline/experiments/`.
- Keep vocab files under `baseline/tokenizers/`.
- Use `console` logging while iterating quickly; switch to `wandb` when needed.
- Prefer changing config first; only add new adapters when config-only changes cannot express what you need.
