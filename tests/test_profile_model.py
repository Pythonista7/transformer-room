from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from src.profile_model import profile_model


class ProfileModelTests(unittest.TestCase):
    def test_profile_model_exports_trace_for_single_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "tiny.txt"
            vocab_path = tmp_path / "tiny_vocab.txt"
            trace_path = tmp_path / "trace.json"
            artifacts_root = tmp_path / "artifacts"

            dataset_text = (
                "To be, or not to be: that is the question.\n\n"
                "Whether 'tis nobler in the mind to suffer.\n\n"
                "The slings and arrows of outrageous fortune.\n\n"
                "Or to take arms against a sea of troubles.\n\n"
            ) * 4
            dataset_path.write_text(dataset_text, encoding="utf-8")

            config = ExperimentConfig(
                run=RunConfig(
                    project_name="profile-test",
                    artifacts_root=str(artifacts_root),
                    resume_from_checkpoint=False,
                    checkpoint_every_n_steps=0,
                    use_torch_compile=False,
                ),
                dataset=LocalTextDatasetConfig(path=str(dataset_path)),
                tokenizer=BPETokenizerConfig(
                    base_vocab_size=64,
                    num_special_tokens=3,
                    vocab_path=str(vocab_path),
                ),
                model=BaselineDecoderConfig(d_model=32, n_heads=4, layers=1),
                train=TrainConfig(
                    epochs=1,
                    optimizer=OptimizerConfig(
                        name="adamw",
                        learning_rate=1e-3,
                        weight_decay=1e-2,
                    ),
                    effective_batch_size=8,
                    seq_len=16,
                    stride=16,
                    data_fraction=1.0,
                ),
                split=HoldoutSplitConfig(train_fraction=0.8, seed=123, shuffle=True),
                logging=LoggingConfig(provider="console"),
            )

            result = profile_model(config, num_steps=1, trace_path=trace_path)

            self.assertTrue(Path(result.run_artifact_dir).exists())
            self.assertTrue((Path(result.run_artifact_dir) / "run_config.json").exists())
            self.assertTrue((Path(result.run_artifact_dir) / "inference_config.json").exists())
            self.assertEqual(result.steps_profiled, 1)
            self.assertTrue(Path(result.trace_path).exists())


if __name__ == "__main__":
    unittest.main()
