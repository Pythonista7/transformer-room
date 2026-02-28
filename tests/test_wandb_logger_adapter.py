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
        self.files: list[tuple[str, str | None, dict[str, object]]] = []

    def add_file(self, path: str, name: str | None = None, **kwargs) -> None:
        self.files.append((path, name, dict(kwargs)))


class FakeLoggedArtifact:
    def __init__(self, version: "FakeArtifactVersion", source_path: str | None) -> None:
        self.version = version
        self.source_path = source_path
        self.wait_called = False
        self.path_exists_during_wait = False

    def wait(self) -> "FakeArtifactVersion":
        self.wait_called = True
        if self.source_path is not None:
            self.path_exists_during_wait = Path(self.source_path).exists()
        return self.version


class FakeArtifactVersion:
    def __init__(
        self,
        store: "FakeArtifactStore",
        *,
        collection_name: str,
        artifact_type: str,
        metadata,
        files: dict[str, bytes],
        aliases: list[str],
    ) -> None:
        self._store = store
        self.collection_name = collection_name
        self.type = artifact_type
        self.metadata = metadata
        self.files = dict(files)
        self.aliases = list(aliases)
        self.deleted = False

    def delete(self, delete_aliases: bool = True) -> None:
        _ = delete_aliases
        self.deleted = True
        self.aliases = []

    def download(self, root: str) -> str:
        download_dir = Path(root) / self.collection_name
        download_dir.mkdir(parents=True, exist_ok=True)
        for name, data in self.files.items():
            (download_dir / name).write_bytes(data)
        return str(download_dir)


class FakeArtifactStore:
    def __init__(self) -> None:
        self.collections: dict[str, list[FakeArtifactVersion]] = {}

    def publish(self, artifact: FakeArtifact, aliases: list[str]) -> FakeLoggedArtifact:
        versions = self.collections.setdefault(artifact.name, [])
        for version in versions:
            version.aliases = [alias for alias in version.aliases if alias not in aliases]

        files: dict[str, bytes] = {}
        source_path: str | None = None
        for path, name, _kwargs in artifact.files:
            file_path = Path(path)
            files[name or file_path.name] = file_path.read_bytes()
            source_path = path

        version = FakeArtifactVersion(
            self,
            collection_name=artifact.name,
            artifact_type=artifact.type,
            metadata=artifact.metadata,
            files=files,
            aliases=list(aliases),
        )
        versions.append(version)
        return FakeLoggedArtifact(version=version, source_path=source_path)

    def artifact(self, ref: str) -> FakeArtifactVersion:
        collection_name, alias = ref.split("/")[-1].split(":", 1)
        for version in reversed(self.collections.get(collection_name, [])):
            if version.deleted:
                continue
            if alias in version.aliases:
                return version
        raise FileNotFoundError(ref)

    def artifact_versions(self, artifact_type: str, artifact_name: str) -> list[FakeArtifactVersion]:
        return [
            version
            for version in self.collections.get(artifact_name, [])
            if version.type == artifact_type and not version.deleted
        ]


class FakeApi:
    def __init__(self, store: FakeArtifactStore) -> None:
        self._store = store

    def artifact(self, ref: str) -> FakeArtifactVersion:
        return self._store.artifact(ref)

    def artifact_versions(self, artifact_type: str, artifact_name: str) -> list[FakeArtifactVersion]:
        return self._store.artifact_versions(artifact_type, artifact_name)


class FakeRun:
    def __init__(self, store: FakeArtifactStore) -> None:
        self.id = "run-123"
        self.name = "lr-vs-batch-run"
        self.entity = "test-entity"
        self.project = "transformer-room-baseline"
        self._store = store
        self.logged_artifacts: list[tuple[FakeArtifact, list[str], FakeLoggedArtifact]] = []
        self.finished = False

    def log(self, metrics, step=None) -> None:
        _ = metrics
        _ = step

    def log_artifact(self, artifact: FakeArtifact, aliases=None) -> FakeLoggedArtifact:
        logged_artifact = self._store.publish(artifact, list(aliases or []))
        self.logged_artifacts.append((artifact, list(aliases or []), logged_artifact))
        return logged_artifact

    def use_artifact(self, ref: str) -> FakeArtifactVersion:
        return self._store.artifact(ref)

    def watch(self, model, loss_fn, log="all", log_freq=10) -> None:
        _ = model
        _ = loss_fn
        _ = log
        _ = log_freq

    def finish(self) -> None:
        self.finished = True


class WandbLoggerAdapterTests(unittest.TestCase):
    def _make_fake_wandb(self) -> tuple[types.SimpleNamespace, FakeRun, FakeArtifactStore, list[dict[str, object]]]:
        init_calls: list[dict[str, object]] = []
        store = FakeArtifactStore()
        fake_run = FakeRun(store)

        def fake_init(**kwargs):
            init_calls.append(kwargs)
            return fake_run

        fake_wandb = types.SimpleNamespace(
            init=fake_init,
            Artifact=FakeArtifact,
            Api=lambda: FakeApi(store),
        )
        return fake_wandb, fake_run, store, init_calls

    def test_start_passes_group_and_checkpoint_save_uses_ephemeral_upload_settings(self) -> None:
        fake_wandb, fake_run, _store, init_calls = self._make_fake_wandb()
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
                artifact_ref = session.save(
                    str(artifact_path),
                    artifact_name="lr-run-checkpoint",
                    artifact_type="checkpoint",
                    aliases=("latest",),
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

        artifact, aliases, logged_artifact = fake_run.logged_artifacts[0]
        self.assertEqual(artifact.name, "lr-run-checkpoint")
        self.assertEqual(artifact.type, "checkpoint")
        self.assertEqual(artifact.metadata, {"global_step": 10})
        self.assertEqual(aliases, ["latest"])
        self.assertEqual(
            artifact.files,
            [
                (
                    str(artifact_path.resolve()),
                    "checkpoint.pt",
                    {"policy": "immutable", "skip_cache": True},
                )
            ],
        )
        self.assertTrue(logged_artifact.wait_called)
        self.assertTrue(logged_artifact.path_exists_during_wait)
        self.assertFalse(artifact_path.exists())
        self.assertEqual(
            artifact_ref,
            "test-entity/transformer-room-baseline/lr-run-checkpoint:latest",
        )
        self.assertTrue(fake_run.finished)

    def test_checkpoint_save_prunes_older_versions(self) -> None:
        fake_wandb, fake_run, store, _init_calls = self._make_fake_wandb()
        adapter = WandbLoggerAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "checkpoint.pt"

            original_module = sys.modules.get("wandb")
            sys.modules["wandb"] = fake_wandb
            try:
                session = adapter.start(
                    cfg=LoggingConfig(provider="wandb"),
                    project_name="transformer-room-baseline",
                    run_name="lr-run",
                    group_name=None,
                    config_payload={},
                )
                artifact_path.write_text("checkpoint-v1", encoding="utf-8")
                session.save(
                    str(artifact_path),
                    artifact_name="lr-run-checkpoint",
                    artifact_type="checkpoint",
                    aliases=("latest",),
                )

                artifact_path.write_text("checkpoint-v2", encoding="utf-8")
                session.save(
                    str(artifact_path),
                    artifact_name="lr-run-checkpoint",
                    artifact_type="checkpoint",
                    aliases=("latest", "final"),
                )
                session.close()
            finally:
                if original_module is None:
                    sys.modules.pop("wandb", None)
                else:
                    sys.modules["wandb"] = original_module

        versions = store.collections["lr-run-checkpoint"]
        active_versions = [version for version in versions if not version.deleted]
        self.assertEqual(len(active_versions), 1)
        self.assertEqual(active_versions[0].aliases, ["latest", "final"])
        self.assertTrue(versions[0].deleted)

    def test_restore_downloads_latest_artifact(self) -> None:
        fake_wandb, _fake_run, _store, _init_calls = self._make_fake_wandb()
        adapter = WandbLoggerAdapter()

        with tempfile.TemporaryDirectory() as tmpdir:
            upload_path = Path(tmpdir) / "checkpoint.pt"
            restore_path = Path(tmpdir) / "restored.pt"
            upload_path.write_text("checkpoint-payload", encoding="utf-8")

            original_module = sys.modules.get("wandb")
            sys.modules["wandb"] = fake_wandb
            try:
                session = adapter.start(
                    cfg=LoggingConfig(provider="wandb"),
                    project_name="transformer-room-baseline",
                    run_name="lr-run",
                    group_name=None,
                    config_payload={},
                )
                session.save(
                    str(upload_path),
                    artifact_name="lr-run-checkpoint",
                    artifact_type="checkpoint",
                    aliases=("latest",),
                )
                restored = session.restore(
                    str(restore_path),
                    artifact_name="lr-run-checkpoint",
                    artifact_type="checkpoint",
                    alias="latest",
                )
                session.close()
            finally:
                if original_module is None:
                    sys.modules.pop("wandb", None)
                else:
                    sys.modules["wandb"] = original_module

            self.assertTrue(restored)
            self.assertTrue(restore_path.exists())
            self.assertEqual(restore_path.read_text(encoding="utf-8"), "checkpoint-payload")


if __name__ == "__main__":
    unittest.main()
