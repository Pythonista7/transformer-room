from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import baseline.adapters  # noqa: F401
from baseline.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from baseline.core.registry import LOGGER_ADAPTERS, get_model_adapter
from baseline.core.types import SpecialTokenIds, VocabInfo
from baseline.train import model_pipeline, resolve_wandb_lineage


class FakeRemoteLoggerSession:
    def __init__(self, remote_checkpoint_payload: dict[str, Any] | None) -> None:
        self.remote_checkpoint_payload = remote_checkpoint_payload
        self.saved: list[dict[str, Any]] = []
        self.restore_calls: list[dict[str, Any]] = []

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        _ = metrics
        _ = step

    def save(
        self,
        path: str,
        *,
        artifact_name: str | None = None,
        artifact_type: str | None = None,
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str | None:
        self.saved.append(
            {
                "path": path,
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "aliases": tuple(aliases) if aliases is not None else None,
                "metadata": dict(metadata) if metadata is not None else None,
            }
        )
        artifact_path = Path(path)
        if artifact_type in {"checkpoint", "model"} and artifact_path.exists():
            artifact_path.unlink()
        primary_alias = aliases[0] if aliases else "latest"
        return (
            f"wandb://{artifact_name}:{primary_alias}"
            if artifact_name is not None
            else None
        )

    def restore(
        self,
        path: str,
        *,
        artifact_name: str,
        artifact_type: str | None = None,
        alias: str = "latest",
    ) -> bool:
        self.restore_calls.append(
            {
                "path": path,
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "alias": alias,
            }
        )
        if self.remote_checkpoint_payload is None:
            return False
        torch.save(self.remote_checkpoint_payload, path)
        return True

    def watch(self, model, loss_fn) -> None:
        _ = model
        _ = loss_fn

    def close(self) -> None:
        return


class FakeRemoteLoggerAdapter:
    def __init__(
        self,
        *,
        remote_checkpoint_payload: dict[str, Any] | None = None,
        existing_run_names: set[str] | None = None,
    ) -> None:
        self.remote_checkpoint_payload = remote_checkpoint_payload
        self.existing_run_names = set(existing_run_names or set())
        self.sessions: list[FakeRemoteLoggerSession] = []

    def has_remote_artifact(
        self,
        *,
        project_name: str,
        artifact_name: str,
        alias: str = "latest",
    ) -> bool:
        _ = project_name
        _ = alias
        return artifact_name.removesuffix("-checkpoint") in self.existing_run_names

    def start(
        self,
        cfg: LoggingConfig,
        project_name: str,
        run_name: str | None,
        group_name: str | None,
        config_payload: dict[str, Any],
    ) -> FakeRemoteLoggerSession:
        _ = cfg
        _ = project_name
        _ = run_name
        _ = group_name
        _ = config_payload
        session = FakeRemoteLoggerSession(self.remote_checkpoint_payload)
        self.sessions.append(session)
        return session


def make_wandb_config(tmp_path: Path, *, run_name: str, resume_from_checkpoint: bool) -> ExperimentConfig:
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
            project_name="wandb-lineage-test",
            run_name=run_name,
            artifacts_root=str(artifacts_root),
            resume_from_checkpoint=resume_from_checkpoint,
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
            learning_rate=1e-3,
            batch_size=4,
            seq_len=16,
            stride=16,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(train_fraction=0.8, seed=123, shuffle=True),
        logging=LoggingConfig(
            provider="wandb",
            wandb=WandbMetricsConfig(
                watch_model=False,
                log_every_n_steps=1,
                diagnostics_every_n_steps=1,
                val_every_n_steps=1,
                attention_entropy_every_n_steps=1,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=8,
            ),
        ),
    )


def build_checkpoint_payload(config: ExperimentConfig) -> dict[str, Any]:
    special = SpecialTokenIds(
        base_vocab_size=64,
        num_special_tokens=3,
        eos_id=64,
        pad_id=65,
        unk_id=66,
    )
    vocab = VocabInfo(
        token_to_id={idx: idx for idx in range(special.vocab_size)},
        id_to_token=list(range(special.vocab_size)),
        special=special,
    )
    model = get_model_adapter(config.model.name).build(
        cfg=config.model,
        vocab=vocab,
        special=special,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    return {
        "epoch": 1,
        "batch_idx": 0,
        "global_step": 7,
        "tokens_seen_train": 123,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
    }


class WandbRemoteLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_wandb_adapter = LOGGER_ADAPTERS["wandb"]

    def tearDown(self) -> None:
        LOGGER_ADAPTERS["wandb"] = self.original_wandb_adapter

    def test_resolve_wandb_lineage_resume_choice_keeps_base_run_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_wandb_config(
                Path(tmpdir),
                run_name="deterministic-run",
                resume_from_checkpoint=False,
            )
            adapter = FakeRemoteLoggerAdapter(existing_run_names={"deterministic-run"})
            answers = iter(["1"])

            resolved = resolve_wandb_lineage(
                config,
                adapter,
                input_fn=lambda _prompt: next(answers),
                interactive=True,
            )

        self.assertEqual(resolved.run.run_name, "deterministic-run")
        self.assertTrue(resolved.run.resume_from_checkpoint)

    def test_resolve_wandb_lineage_new_suffix_choice_renames_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_wandb_config(
                Path(tmpdir),
                run_name="deterministic-run",
                resume_from_checkpoint=False,
            )
            adapter = FakeRemoteLoggerAdapter(existing_run_names={"deterministic-run"})
            answers = iter(["2", "fresh attempt"])

            resolved = resolve_wandb_lineage(
                config,
                adapter,
                input_fn=lambda _prompt: next(answers),
                interactive=True,
            )

        self.assertEqual(resolved.run.run_name, "deterministic-run-fresh-attempt")
        self.assertFalse(resolved.run.resume_from_checkpoint)

    def test_resolve_wandb_lineage_noninteractive_collision_without_resume_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_wandb_config(
                Path(tmpdir),
                run_name="deterministic-run",
                resume_from_checkpoint=False,
            )
            adapter = FakeRemoteLoggerAdapter(existing_run_names={"deterministic-run"})

            with self.assertRaisesRegex(ValueError, "Remote checkpoint already exists"):
                resolve_wandb_lineage(config, adapter, interactive=False)

    def test_model_pipeline_restores_remote_checkpoint_when_local_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = make_wandb_config(
                tmp_path,
                run_name="remote-resume-run",
                resume_from_checkpoint=True,
            )
            adapter = FakeRemoteLoggerAdapter(
                remote_checkpoint_payload=build_checkpoint_payload(config),
                existing_run_names={"remote-resume-run"},
            )
            LOGGER_ADAPTERS["wandb"] = adapter

            result = model_pipeline(config)

            session = adapter.sessions[-1]
            run_dir = Path(result.run_artifact_dir)
            self.assertEqual(result.global_step, 7)
            self.assertEqual(len(session.restore_calls), 1)
            self.assertEqual(
                session.restore_calls[0]["artifact_name"],
                "remote-resume-run-checkpoint",
            )
            self.assertFalse(Path(result.checkpoint_path).exists())
            self.assertFalse(Path(result.final_model_path).exists())
            self.assertTrue((run_dir / "run_config.json").exists())
            self.assertTrue((run_dir / "inference_config.json").exists())
            self.assertEqual(
                result.checkpoint_artifact_ref,
                "wandb://remote-resume-run-checkpoint:latest",
            )
            self.assertEqual(
                result.final_model_artifact_ref,
                "wandb://remote-resume-run-model:latest",
            )


if __name__ == "__main__":
    unittest.main()
