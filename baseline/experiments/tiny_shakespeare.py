from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from baseline.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    RunConfig,
    TrainConfig,
)
from baseline.train import model_pipeline


def build_config() -> ExperimentConfig:
    project_root = PROJECT_ROOT

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            artifacts_root=str(project_root / "baseline" / "models"),
            resume_from_checkpoint=True,
            checkpoint_every_n_steps=250,
            use_torch_compile=True,
            torch_compile_mode="default",
            torch_compile_fullgraph=False,
            torch_compile_dynamic=False,
        ),
        dataset=LocalTextDatasetConfig(
            path=str(project_root / "datasets" / "tiny_shakespeare.txt"),
            segment_delimiter="\n\n",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=100,
            num_special_tokens=3,
            vocab_path=str(
                project_root
                / "baseline"
                / "tokenizers"
                / "tiny_shakespeare_bpe_vocab.txt"
            ),
        ),
        model=BaselineDecoderConfig(
            d_model=128,
            n_heads=8,
            layers=2,
        ),
        train=TrainConfig(
            epochs=3,
            learning_rate=0.001,
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
