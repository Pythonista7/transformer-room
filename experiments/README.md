# Experiments Guide

This directory contains runnable experiment entrypoints for the baseline training pipeline.

## Directory layout

- `experiments/baseline/basic/`: single-run baselines
- `experiments/baseline/hyperparam_sweeps/`: grid/sweep scripts
- `experiments/baseline/memory_experiments/`: memory and throughput studies

## Prerequisites

From repo root:

```bash
python3 -m pip install -r requirements.txt
```

If you use a virtual environment, activate it first.

## Run an existing experiment

From repo root, use either style.

Script path style:

```bash
python3 experiments/baseline/basic/wikitext2_GPT2_v1.py
```

Module style:

```bash
python3 -m experiments.baseline.basic.wikitext2_GPT2_v1
```

## Create a new experiment

1. Add a new file under the relevant folder (for example `experiments/baseline/hyperparam_sweeps/my_experiment.py`).
2. Define `build_config() -> ExperimentConfig`.
3. Call `model_pipeline(build_config())` in `main()`.

Minimal template:

```python
from __future__ import annotations
import sys
from pathlib import Path

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
)
from src.train import model_pipeline
from src.training import wikitext as training_wikitext


def build_config() -> ExperimentConfig:
    vocab_path = PROJECT_ROOT / "src" / "vocabs" / "wikitext2_v1_hf_vocab_bpe.txt"
    base_vocab_size = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-v1",
        vocab_path=vocab_path,
    )

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            run_name="my_experiment_v1",
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            resume_from_checkpoint=False,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name="Salesforce/wikitext",
            dataset_config="wikitext-2-v1",
            split="train",
            text_field="text",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=base_vocab_size,
            num_special_tokens=3,
            vocab_path=str(vocab_path),
        ),
        model=BaselineDecoderConfig(d_model=128, n_heads=8, layers=2),
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            effective_batch_size=64,
            seq_len=128,
            stride=128,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(train_fraction=0.9, seed=42, shuffle=False),
        logging=LoggingConfig(provider="console"),
    )


def main() -> int:
    result = model_pipeline(build_config())
    print(result.run_artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Common knobs to change

- Dataset: `HFTextDatasetConfig` or `LocalTextDatasetConfig`
- Model scale: `d_model`, `n_heads`, `layers`, `dropout`
- Optimizer: `name`, `learning_rate`, `weight_decay`
- Batching: `effective_batch_size`, `micro_batch_size`, `accumulation_steps`, `lr_scaling`
- Logging: `LoggingConfig(provider="console" | "wandb")`

## Notes

- Keep `run.run_name` stable and deterministic for repeatable runs.
- Use `run.group_name` for timestamped grouping when running sweeps.
- Artifacts are written under `run.artifacts_root`.
- For deeper architecture details, see `src/docs/Readme.md` and `src/docs/README_QUICKSTART.md`.
