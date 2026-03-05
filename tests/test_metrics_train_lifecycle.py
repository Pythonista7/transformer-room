from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from baseline.config import (
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
from baseline.train import model_pipeline
from baseline.training.metrics import BaseMetricPlugin, StepMetricsContext


class _FailingPlugin(BaseMetricPlugin):
    name = "failing_plugin"

    def __init__(self) -> None:
        self.ended = False

    def after_backward(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        raise RuntimeError("failing plugin")

    def on_train_end(self) -> None:
        self.ended = True


def _make_console_config(tmp_path: Path) -> ExperimentConfig:
    dataset_path = tmp_path / "tiny.txt"
    vocab_path = tmp_path / "tiny_vocab.txt"
    artifacts_root = tmp_path / "artifacts"

    dataset_text = (
        "To be, or not to be: that is the question.\n\n"
        "Whether 'tis nobler in the mind to suffer.\n\n"
        "The slings and arrows of outrageous fortune.\n\n"
        "Or to take arms against a sea of troubles.\n\n"
    ) * 4
    dataset_path.write_text(dataset_text, encoding="utf-8")

    return ExperimentConfig(
        run=RunConfig(
            project_name="metrics-train-lifecycle-test",
            run_name="metrics-train-lifecycle-run",
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
        model=BaselineDecoderConfig(d_model=32, n_heads=4, layers=3),
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
            batch_size=4,
            seq_len=16,
            stride=16,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(train_fraction=0.8, seed=123, shuffle=True),
        logging=LoggingConfig(provider="console"),
    )


class MetricsTrainLifecycleTests(unittest.TestCase):
    def test_on_train_end_runs_when_plugin_raises(self) -> None:
        plugin = _FailingPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_console_config(Path(tmpdir))
            with self.assertRaisesRegex(RuntimeError, "failing plugin"):
                model_pipeline(cfg, extra_metric_plugins=[plugin])

        self.assertTrue(plugin.ended)


if __name__ == "__main__":
    unittest.main()
