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
    HoldoutSplitConfig,
    HFTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from src.train import model_pipeline
from src.training import wikitext as training_wikitext


def build_config() -> ExperimentConfig:
    project_root = PROJECT_ROOT
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-v1"
    vocab_path = (
        project_root
        / "src"
        / "vocabs"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )
    base_vocab_size = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        vocab_path=vocab_path,
    )

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            artifacts_root=str(project_root / "artifacts" / "models"),
            run_name="wikitext2_gpt2_v1",
            resume_from_checkpoint=True,
            checkpoint_every_n_steps=1000,
            use_torch_compile=False,
            torch_compile_mode="default",
            torch_compile_fullgraph=False,
            torch_compile_dynamic=False,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
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
        ),
        train=TrainConfig(
            epochs=3,
            optimizer=OptimizerConfig(learning_rate=0.001, weight_decay=0.0),
            effective_batch_size=64,
            seq_len=1024,
            stride=1024,
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
    config = build_config()
    result = model_pipeline(config)
    print(
        "Training complete | "
        f"run_dir={result.run_artifact_dir} | "
        f"checkpoint={result.checkpoint_path} | "
        f"final_model={result.final_model_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
