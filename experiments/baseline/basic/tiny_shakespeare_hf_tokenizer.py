from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BaselineDecoderConfig,
    ExperimentConfig,
    HFPretrainedTokenizerConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from src.train import model_pipeline


def build_config() -> ExperimentConfig:
    project_root = PROJECT_ROOT

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            artifacts_root=str(project_root / "artifacts" / "models"),
            run_name="tiny_shakespeare_hf_tokenizer_v1",
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=250,
        ),
        dataset=LocalTextDatasetConfig(
            path=str(project_root / "datasets" / "tiny_shakespeare.txt"),
            segment_delimiter="\n\n",
        ),
        tokenizer=HFPretrainedTokenizerConfig(
            pretrained_name_or_path="gpt2",
            use_fast=True,
        ),
        model=BaselineDecoderConfig(
            d_model=128,
            n_heads=8,
            layers=2,
        ),
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
            effective_batch_size=128,
            seq_len=128,
            stride=128,
            data_mode="materialized",
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
    print(
        "Training complete | "
        f"run_dir={result.run_artifact_dir} | "
        f"checkpoint={result.checkpoint_path} | "
        f"final_model={result.final_model_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
