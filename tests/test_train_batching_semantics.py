from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    WandbMetricsConfig,
)
from src.core.registry import LOGGER_ADAPTERS
from src.train import model_pipeline


class RecordingLoggerSession:
    def __init__(self) -> None:
        self.logged: list[tuple[int | None, dict[str, float]]] = []
        self.saved: list[dict[str, Any]] = []
        self.restore_calls: list[dict[str, Any]] = []

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.logged.append((step, dict(metrics)))

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
        return f"wandb://{artifact_name}:{(aliases[0] if aliases else 'latest')}"

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
        return False

    def watch(self, model, loss_fn) -> None:
        _ = model
        _ = loss_fn

    def close(self) -> None:
        return


class RecordingLoggerAdapter:
    def __init__(self) -> None:
        self.sessions: list[RecordingLoggerSession] = []

    def start(
        self,
        cfg: LoggingConfig,
        project_name: str,
        run_name: str | None,
        group_name: str | None,
        config_payload: dict[str, Any],
    ) -> RecordingLoggerSession:
        _ = cfg
        _ = project_name
        _ = run_name
        _ = group_name
        _ = config_payload
        session = RecordingLoggerSession()
        self.sessions.append(session)
        return session


def _make_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_path = tmp_path / "tiny.txt"
    vocab_path = tmp_path / "tiny_vocab.txt"
    artifacts_root = tmp_path / "artifacts"
    dataset_path.write_text(("a b c d e f g h\n\n" * 40), encoding="utf-8")
    return dataset_path, vocab_path, artifacts_root


def _make_config(
    *,
    tmp_path: Path,
    run_name: str | None,
    provider: str,
    effective_batch_size: int,
    micro_batch_size: int | None = None,
    accumulation_steps: int | None = None,
    persist_local_artifacts: bool = True,
    enable_artifact_io: bool = True,
    resume_from_checkpoint: bool = False,
    checkpoint_every_n_steps: int = 0,
    wandb_cfg: WandbMetricsConfig | None = None,
) -> ExperimentConfig:
    dataset_path, vocab_path, artifacts_root = _make_dataset(tmp_path)
    return ExperimentConfig(
        run=RunConfig(
            project_name="batching-semantics-test",
            run_name=run_name,
            artifacts_root=str(artifacts_root),
            persist_local_artifacts=persist_local_artifacts,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            use_torch_compile=False,
        ),
        dataset=LocalTextDatasetConfig(path=str(dataset_path)),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=32,
            num_special_tokens=3,
            vocab_path=str(vocab_path),
        ),
        model=BaselineDecoderConfig(d_model=16, n_heads=4, layers=1),
        train=TrainConfig(
            effective_batch_size=effective_batch_size,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
            epochs=1,
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
            seq_len=8,
            stride=8,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(train_fraction=0.8, seed=1, shuffle=False),
        logging=LoggingConfig(
            provider=provider,
            enable_artifact_io=enable_artifact_io,
            wandb=wandb_cfg or WandbMetricsConfig(),
        ),
    )


class TrainBatchingSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_wandb_adapter = LOGGER_ADAPTERS["wandb"]
        self.recording_adapter = RecordingLoggerAdapter()
        LOGGER_ADAPTERS["wandb"] = self.recording_adapter

    def tearDown(self) -> None:
        LOGGER_ADAPTERS["wandb"] = self.original_wandb_adapter

    def test_non_accum_and_accum_have_same_optimizer_step_count_for_same_effective_batch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            non_accum_cfg = _make_config(
                tmp_path=Path(tmp_a),
                run_name=None,
                provider="console",
                effective_batch_size=4,
            )
            accum_cfg = _make_config(
                tmp_path=Path(tmp_b),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                micro_batch_size=2,
                accumulation_steps=2,
            )

            non_accum_result = model_pipeline(non_accum_cfg)
            accum_result = model_pipeline(accum_cfg)

            self.assertGreater(non_accum_result.global_step, 0)
            self.assertEqual(non_accum_result.global_step, accum_result.global_step)

    def test_logging_and_checkpoints_use_optimizer_steps_under_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="batching-cadence-run",
                provider="wandb",
                effective_batch_size=4,
                micro_batch_size=2,
                accumulation_steps=2,
                checkpoint_every_n_steps=1,
                wandb_cfg=WandbMetricsConfig(
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=1,
                    val_every_n_steps=1,
                    attention_entropy_every_n_steps=1,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
            )

            result = model_pipeline(cfg)
            session = self.recording_adapter.sessions[-1]

            logged_steps = [step for step, _ in session.logged if step is not None]
            self.assertTrue(logged_steps)
            self.assertEqual(max(logged_steps), result.global_step)
            self.assertTrue(all(step <= result.global_step for step in logged_steps))

            checkpoint_steps = [
                int(entry["metadata"]["global_step"])
                for entry in session.saved
                if entry["artifact_type"] == "checkpoint" and entry["metadata"] is not None
            ]
            self.assertTrue(checkpoint_steps)
            self.assertTrue(all(1 <= step <= result.global_step for step in checkpoint_steps))
            self.assertTrue(
                set(range(1, result.global_step + 1)).issubset(set(checkpoint_steps))
            )

    def test_artifact_writes_and_uploads_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="artifact-gating-run",
                provider="wandb",
                effective_batch_size=4,
                micro_batch_size=2,
                accumulation_steps=2,
                checkpoint_every_n_steps=1,
                persist_local_artifacts=False,
                enable_artifact_io=False,
                wandb_cfg=WandbMetricsConfig(
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=1,
                    val_every_n_steps=1,
                    attention_entropy_every_n_steps=1,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
            )

            result = model_pipeline(cfg)
            session = self.recording_adapter.sessions[-1]

            self.assertEqual(session.saved, [])
            self.assertEqual(session.restore_calls, [])
            self.assertIsNone(result.checkpoint_artifact_ref)
            self.assertIsNone(result.final_model_artifact_ref)

            run_dir = Path(result.run_artifact_dir)
            self.assertFalse((run_dir / cfg.run.checkpoint_filename).exists())
            self.assertFalse((run_dir / cfg.run.final_model_filename).exists())


if __name__ == "__main__":
    unittest.main()
