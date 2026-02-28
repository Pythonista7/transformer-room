from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from baseline.adapters.loggers import WandbLoggerAdapter
from baseline.config import LoggingConfig


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata=None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str | None]] = []

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append((path, name))


class FakeRun:
    def __init__(self) -> None:
        self.id = "run-123"
        self.name = "lr-vs-batch-run"
        self.logged_artifacts: list[tuple[FakeArtifact, list[str]]] = []
        self.finished = False

    def log(self, metrics, step=None) -> None:
        _ = metrics
        _ = step

    def log_artifact(self, artifact: FakeArtifact, aliases=None) -> None:
        self.logged_artifacts.append((artifact, list(aliases or [])))

    def watch(self, model, loss_fn, log="all", log_freq=10) -> None:
        _ = model
        _ = loss_fn
        _ = log
        _ = log_freq

    def finish(self) -> None:
        self.finished = True


class WandbLoggerAdapterTests(unittest.TestCase):
    def test_start_passes_group_and_save_logs_artifact(self) -> None:
        init_calls: list[dict[str, object]] = []
        fake_run = FakeRun()

        def fake_init(**kwargs):
            init_calls.append(kwargs)
            return fake_run

        fake_wandb = types.SimpleNamespace(init=fake_init, Artifact=FakeArtifact)
        adapter = WandbLoggerAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "checkpoint.pt"
            artifact_path.write_text("checkpoint-bytes", encoding="utf-8")

            original_module = sys.modules.get("wandb")
            sys.modules["wandb"] = fake_wandb
            try:
                session = adapter.start(
                    cfg=LoggingConfig(provider="wandb"),
                    project_name="transformer-room-baseline",
                    run_name="lr-run",
                    group_name="sweep-20260228-120000",
                    config_payload={"train": {"learning_rate": 1e-4}},
                )
                session.save(
                    str(artifact_path),
                    artifact_name="lr-run-checkpoint",
                    artifact_type="checkpoint",
                    aliases=("latest", "step-10"),
                    metadata={"global_step": 10},
                )
                session.close()
            finally:
                if original_module is None:
                    sys.modules.pop("wandb", None)
                else:
                    sys.modules["wandb"] = original_module

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(init_calls[0]["project"], "transformer-room-baseline")
        self.assertEqual(init_calls[0]["name"], "lr-run")
        self.assertEqual(init_calls[0]["group"], "sweep-20260228-120000")
        self.assertEqual(len(fake_run.logged_artifacts), 1)

        artifact, aliases = fake_run.logged_artifacts[0]
        self.assertEqual(artifact.name, "lr-run-checkpoint")
        self.assertEqual(artifact.type, "checkpoint")
        self.assertEqual(artifact.metadata, {"global_step": 10})
        self.assertEqual(aliases, ["latest", "step-10"])
        self.assertEqual(artifact.files, [(str(artifact_path.resolve()), "checkpoint.pt")])
        self.assertTrue(fake_run.finished)


if __name__ == "__main__":
    unittest.main()
