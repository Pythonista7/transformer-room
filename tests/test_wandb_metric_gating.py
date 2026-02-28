from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from baseline.core.registry import LOGGER_ADAPTERS
from baseline.train import model_pipeline


class RecordingLoggerSession:
    def __init__(self) -> None:
        self.logged: list[tuple[int | None, dict[str, float]]] = []
        self.saved: list[dict[str, Any]] = []
        self.watch_called = 0

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
        _ = path
        _ = artifact_name
        _ = artifact_type
        _ = alias
        return False

    def watch(self, model, loss_fn) -> None:
        _ = model
        _ = loss_fn
        self.watch_called += 1

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


def _flatten_logged_keys(session: RecordingLoggerSession) -> set[str]:
    keys: set[str] = set()
    for _, payload in session.logged:
        keys.update(payload.keys())
    return keys


def _make_config(
    *,
    tmp_path: Path,
    wandb_cfg: WandbMetricsConfig,
    watch_model: bool,
) -> ExperimentConfig:
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

    wandb_cfg.watch_model = watch_model
    return ExperimentConfig(
        run=RunConfig(
            project_name="wandb-gating-test",
            run_name="wandb-gating-test-run",
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
            learning_rate=1e-3,
            batch_size=4,
            seq_len=16,
            stride=16,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(train_fraction=0.8, seed=123, shuffle=True),
        logging=LoggingConfig(provider="wandb", wandb=wandb_cfg),
    )


class WandbMetricGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_wandb_adapter = LOGGER_ADAPTERS["wandb"]
        self.recording_adapter = RecordingLoggerAdapter()
        LOGGER_ADAPTERS["wandb"] = self.recording_adapter

    def tearDown(self) -> None:
        LOGGER_ADAPTERS["wandb"] = self.original_wandb_adapter

    def test_metrics_disabled_are_not_logged_and_watch_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                wandb_cfg=WandbMetricsConfig(
                    enable_train_loss_vs_tokens=False,
                    enable_val_loss_vs_tokens=False,
                    enable_perplexity=False,
                    enable_step_time=False,
                    enable_peak_memory=False,
                    enable_global_grad_norm=False,
                    enable_activation_norms=False,
                    enable_ln_grad_norms=False,
                    enable_attention_entropy=False,
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=1,
                    val_every_n_steps=1,
                    attention_entropy_every_n_steps=1,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
                watch_model=False,
            )

            model_pipeline(cfg)
            session = self.recording_adapter.sessions[-1]

            keys = _flatten_logged_keys(session)
            self.assertNotIn("train_loss", keys)
            self.assertNotIn("tokens_seen_train", keys)
            self.assertNotIn("val_loss", keys)
            self.assertNotIn("val_perplexity", keys)
            self.assertNotIn("global_grad_norm", keys)
            self.assertNotIn("activation_norm_first", keys)
            self.assertNotIn("ln_weight_grad_norm_first", keys)
            self.assertNotIn("attention_entropy_first", keys)
            self.assertEqual(session.watch_called, 0)

    def test_metrics_enabled_are_logged_and_watch_is_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                wandb_cfg=WandbMetricsConfig(
                    enable_train_loss_vs_tokens=True,
                    enable_val_loss_vs_tokens=True,
                    enable_perplexity=True,
                    enable_step_time=True,
                    enable_peak_memory=True,
                    enable_global_grad_norm=True,
                    enable_activation_norms=True,
                    enable_ln_grad_norms=True,
                    enable_attention_entropy=True,
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=1,
                    val_every_n_steps=1,
                    attention_entropy_every_n_steps=1,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
                watch_model=True,
            )

            model_pipeline(cfg)
            session = self.recording_adapter.sessions[-1]

            keys = _flatten_logged_keys(session)
            self.assertIn("train_loss", keys)
            self.assertIn("tokens_seen_train", keys)
            self.assertIn("val_loss", keys)
            self.assertIn("train_perplexity", keys)
            self.assertIn("val_perplexity", keys)
            self.assertIn("step_time_ms", keys)
            self.assertIn("global_grad_norm", keys)
            self.assertIn("activation_norm_first", keys)
            self.assertIn("ln_weight_grad_norm_first", keys)
            self.assertIn("ln_bias_grad_norm_first", keys)
            self.assertIn("attention_entropy_first", keys)

            if any("peak_memory_gib" in payload for _, payload in session.logged):
                self.assertIn("peak_memory_gib", keys)
            self.assertEqual(session.watch_called, 1)
            saved_types = {entry["artifact_type"] for entry in session.saved}
            self.assertIn("metadata", saved_types)
            self.assertIn("checkpoint", saved_types)
            self.assertIn("model", saved_types)
            self.assertFalse(Path(session.saved[-2]["path"]).exists())
            self.assertFalse(Path(session.saved[-1]["path"]).exists())
            self.assertTrue(Path(cfg.run.artifacts_root, "wandb-gating-test-run", "run_config.json").exists())
            self.assertTrue(
                Path(cfg.run.artifacts_root, "wandb-gating-test-run", "inference_config.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
