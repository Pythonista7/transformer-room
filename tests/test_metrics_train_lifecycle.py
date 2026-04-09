from __future__ import annotations

import tempfile
import unittest
import json
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
from src.train import model_pipeline
from src.training.metrics import BaseMetricPlugin, StepMetricsContext


class _FailingPlugin(BaseMetricPlugin):
    name = "failing_plugin"

    def __init__(self) -> None:
        self.ended = False

    def after_backward(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        raise RuntimeError("failing plugin")

    def on_train_end(self) -> None:
        self.ended = True


def _make_config(tmp_path: Path, *, provider: str) -> ExperimentConfig:
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
            effective_batch_size=4,
            seq_len=16,
            stride=16,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(train_fraction=0.8, seed=123, shuffle=True),
        logging=LoggingConfig(provider=provider),
    )


class MetricsTrainLifecycleTests(unittest.TestCase):
    def test_on_train_end_runs_when_plugin_raises(self) -> None:
        plugin = _FailingPlugin()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(Path(tmpdir), provider="console")
            with self.assertRaisesRegex(RuntimeError, "failing plugin"):
                model_pipeline(cfg, extra_metric_plugins=[plugin])

        self.assertTrue(plugin.ended)

    def test_local_provider_writes_metrics_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(Path(tmpdir), provider="local")
            model_pipeline(cfg)

            run_dir = Path(cfg.run.artifacts_root) / cfg.run.run_name
            metrics_path = run_dir / "metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertGreater(len(records), 0)
            self.assertIn("metrics", records[0])
            self.assertIn("logged_at", records[0])

    def test_console_provider_does_not_write_metrics_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(Path(tmpdir), provider="console")
            model_pipeline(cfg)

            run_dir = Path(cfg.run.artifacts_root) / cfg.run.run_name
            self.assertFalse((run_dir / "metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
