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
    - `LoggingConfig`
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
        logging=LoggingConfig(provider="console"),
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
  - `LoggingConfig(provider="console")` or `LoggingConfig(provider="wandb")`
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
