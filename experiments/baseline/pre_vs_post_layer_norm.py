from __future__ import annotations
import sys
from pathlib import Path

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

    pre_norm_cfg = ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            run_name="pre_ln_norm",
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
        model=BaselineDecoderConfig(d_model=128, n_heads=8, layers=20,norm_placement="pre"),
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
    
    post_norm_cfg = ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            run_name="post_ln_norm",
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
        model=BaselineDecoderConfig(d_model=128, n_heads=8, layers=20,norm_placement="post"),
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
    
    return {
        "pre": pre_norm_cfg,
        "post": post_norm_cfg
    }


def main() -> int:
    cfgs = build_config() # keys: "pre" and "post" 
    result = model_pipeline()
    print(result.run_artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())