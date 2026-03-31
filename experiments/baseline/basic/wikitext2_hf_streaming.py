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
    HFTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    PreSplitConfig,
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
            run_name="wikitext2_hf_streaming_v1",
            resume_from_checkpoint=True,
            checkpoint_every_n_steps=None,
            use_torch_compile=True
        ),
        dataset=HFTextDatasetConfig(
            dataset_name="epfml/FineWeb-HQ",
            dataset_config=None, # Same as huggingface dataset-subset
            split="train",
            # validation_split="validation",
            text_field="text",
            shuffle_buffer_size=30,
        ),
        tokenizer=HFPretrainedTokenizerConfig(
            pretrained_name_or_path="gpt2",
            use_fast=True,
        ),
        model=BaselineDecoderConfig(
            d_model=128,
            n_heads=4,
            layers=2,
        ),
        train=TrainConfig(
            epochs=3,
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.01),
            effective_batch_size=32,
            micro_batch_size=16,
            accumulation_steps=2,
            lr_scaling='sqrt',
            seq_len=512,
            stride=512,
            data_mode="streaming",
            max_steps=60,
            run_validation=False
        ),
        split=PreSplitConfig(),
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
    result = main()
    import os
    os._exit(result)
