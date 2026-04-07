from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.adapters import register_builtin_adapters
from src.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LRSchedulerChainConfig,
    LRSchedulerStageConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
    resolve_train_learning_rate,
)
from src.core.registry import LOGGER_ADAPTERS
from src.train import model_pipeline
from src.training.optimizer import build_lr_scheduler, build_optimizer
from src.training.metrics import BaseMetricPlugin, MicroBatchMetricsContext, StepMetricsContext


class RecordingLoggerSession:
    def __init__(self) -> None:
        self.logged: list[tuple[int | None, dict[str, float]]] = []
        self.saved: list[dict[str, Any]] = []
        self.restore_calls: list[dict[str, Any]] = []
        self.uploaded_run_files: list[dict[str, Any]] = []

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

    def get_run_id(self) -> str | None:
        return "recording-run-id"

    def upload_run_files(
        self,
        paths: Sequence[str],
        *,
        base_path: str | None = None,
    ) -> None:
        self.uploaded_run_files.append(
            {
                "paths": list(paths),
                "base_path": base_path,
            }
        )

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


class _MicroBatchCallbackRecorder(BaseMetricPlugin):
    name = "microbatch_callback_recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, int, int]] = []
        self.microbatch_counts_by_step: dict[int, int] = {}

    def after_microbatch_backward(self, ctx: MicroBatchMetricsContext) -> None:
        step = int(ctx.step_ctx.next_global_step)
        micro_batch_in_step = int(ctx.micro_batch_in_step)
        self.events.append(("micro", step, micro_batch_in_step))
        self.microbatch_counts_by_step[step] = (
            self.microbatch_counts_by_step.get(step, 0) + 1
        )

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        self.events.append(("optim", int(ctx.global_step), 0))


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
    lr_scheduler: LRSchedulerChainConfig | None = None,
) -> ExperimentConfig:
    dataset_path, vocab_path, artifacts_root = _make_dataset(tmp_path)
    resolved_micro_batch_size = (
        effective_batch_size
        if micro_batch_size is None
        else int(micro_batch_size)
    )
    resolved_accumulation_steps = (
        1
        if accumulation_steps is None
        else int(accumulation_steps)
    )
    lr_scaling = (
        "sqrt"
        if effective_batch_size > resolved_micro_batch_size and resolved_accumulation_steps > 1
        else "none"
    )
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
            lr_scaling=lr_scaling,
            lr_scheduler=lr_scheduler,
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
        register_builtin_adapters()
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

            saved_types = [entry["artifact_type"] for entry in session.saved]
            self.assertEqual(saved_types, ["metadata", "metadata"])
            self.assertEqual(session.restore_calls, [])
            self.assertIsNone(result.checkpoint_artifact_ref)
            self.assertIsNone(result.final_model_artifact_ref)

            run_dir = Path(result.run_artifact_dir)
            self.assertFalse((run_dir / cfg.run.checkpoint_filename).exists())
            self.assertFalse((run_dir / cfg.run.final_model_filename).exists())

    def test_trace_run_files_upload_even_when_artifact_io_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="trace-upload-run",
                provider="wandb",
                effective_batch_size=4,
                persist_local_artifacts=False,
                enable_artifact_io=False,
            )
            cfg.run.use_torch_compile = True
            cfg.run.torch_compile_trace = True

            trace_dir = Path(tmpdir) / "torch-trace" / "recording-run-id"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_file = trace_dir / "dedicated_log_torch_trace_recording-run-id.log"
            trace_file.write_text("trace", encoding="utf-8")

            with (
                mock.patch("src.train.resolve_torch_compile_trace_dir", return_value=trace_dir),
                mock.patch(
                    "src.train.maybe_compile_model",
                    side_effect=lambda model, device, config, trace_dir=None: (
                        model,
                        True,
                        "enabled",
                    ),
                ),
            ):
                result = model_pipeline(cfg)

            session = self.recording_adapter.sessions[-1]
            saved_types = [entry["artifact_type"] for entry in session.saved]
            self.assertEqual(saved_types, ["metadata", "metadata"])
            self.assertEqual(session.restore_calls, [])
            self.assertEqual(
                session.uploaded_run_files,
                [
                    {
                        "paths": [str(trace_file)],
                        "base_path": str(trace_dir.parent),
                    }
                ],
            )
            self.assertIsNone(result.checkpoint_artifact_ref)
            self.assertIsNone(result.final_model_artifact_ref)

    def test_microbatch_backward_callback_runs_before_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                micro_batch_size=2,
                accumulation_steps=2,
            )
            recorder = _MicroBatchCallbackRecorder()

            result = model_pipeline(cfg, extra_metric_plugins=[recorder])
            self.assertGreater(result.global_step, 0)

            optimizer_events = 0
            pending_microbatches = 0
            for event_type, _, _ in recorder.events:
                if event_type == "micro":
                    pending_microbatches += 1
                    continue

                self.assertGreater(
                    pending_microbatches,
                    0,
                    "Optimizer step happened before any microbatch callback.",
                )
                pending_microbatches = 0
                optimizer_events += 1

            self.assertEqual(optimizer_events, result.global_step)
            self.assertEqual(
                pending_microbatches,
                0,
                "Found trailing microbatch callbacks without optimizer step.",
            )
            self.assertTrue(recorder.microbatch_counts_by_step)
            self.assertTrue(
                all(1 <= count <= 2 for count in recorder.microbatch_counts_by_step.values())
            )

    def test_optimizer_lr_matches_resolved_scaled_lr_under_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                micro_batch_size=2,
                accumulation_steps=2,
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )

            self.assertAlmostEqual(
                float(optimizer.param_groups[0]["lr"]),
                float(resolved.applied_learning_rate),
                places=12,
            )

    def test_scheduler_builder_disabled_when_scheduler_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )

            self.assertIsNone(
                build_lr_scheduler(
                    optimizer,
                    cfg,
                    total_optimizer_steps=100,
                )
            )

    def test_scheduler_builder_emits_lambda_scheduler_for_linear_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.25,
                            end_factor=1.0,
                            steps=4,
                        )
                    ]
                ),
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )

            scheduler = build_lr_scheduler(
                optimizer,
                cfg,
                total_optimizer_steps=100,
            )
            self.assertIsNotNone(scheduler)
            self.assertEqual(scheduler.__class__.__name__, "LambdaLR")

    def test_remainder_stage_requires_known_total_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.1,
                            end_factor=0.5,
                            steps=2,
                        ),
                        LRSchedulerStageConfig(
                            type="cosine",
                            end_factor=0.05,
                            steps=None,
                        ),
                    ]
                ),
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )

            with self.assertRaisesRegex(
                ValueError,
                "requires known total optimizer steps",
            ):
                build_lr_scheduler(optimizer, cfg, total_optimizer_steps=None)

    def test_known_total_allows_remainder_final_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.1,
                            end_factor=1.0,
                            steps=2,
                        ),
                        LRSchedulerStageConfig(
                            type="cosine",
                            end_factor=0.01,
                            steps=None,
                        ),
                    ]
                ),
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )

            scheduler = build_lr_scheduler(
                optimizer,
                cfg,
                total_optimizer_steps=6,
            )
            self.assertIsNotNone(scheduler)
            self.assertEqual(scheduler.__class__.__name__, "LambdaLR")

    def test_cosine_stage_progression_descends_to_end_factor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="cosine",
                            start_factor=1.0,
                            end_factor=0.0,
                            steps=5,
                        )
                    ]
                ),
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )
            scheduler = build_lr_scheduler(optimizer, cfg, total_optimizer_steps=None)
            self.assertIsNotNone(scheduler)

            lrs: list[float] = []
            for _ in range(5):
                lrs.append(float(optimizer.param_groups[0]["lr"]))
                optimizer.step()
                scheduler.step()

            self.assertAlmostEqual(lrs[0], 1e-3, places=12)
            self.assertAlmostEqual(lrs[-1], 0.0, places=12)
            self.assertTrue(all(curr <= prev for prev, curr in zip(lrs, lrs[1:])))

    def test_scheduler_holds_final_factor_after_explicit_stage_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.5,
                            end_factor=1.0,
                            steps=2,
                        )
                    ]
                ),
            )
            model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)
            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )
            scheduler = build_lr_scheduler(optimizer, cfg, total_optimizer_steps=None)
            self.assertIsNotNone(scheduler)

            observed: list[float] = []
            for _ in range(4):
                observed.append(float(optimizer.param_groups[0]["lr"]))
                optimizer.step()
                scheduler.step()

            self.assertEqual(
                [round(value, 12) for value in observed],
                [
                    round(0.0005, 12),
                    round(0.001, 12),
                    round(0.001, 12),
                    round(0.001, 12),
                ],
            )

    def test_step_zero_logs_lr_scaling_diagnostics_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="lr-step-zero-run",
                provider="wandb",
                effective_batch_size=4,
                micro_batch_size=2,
                accumulation_steps=2,
                wandb_cfg=WandbMetricsConfig(
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=1,
                    val_every_n_steps=1,
                    attention_entropy_every_n_steps=1,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
            )

            model_pipeline(cfg)
            session = self.recording_adapter.sessions[-1]
            step_zero_payloads = [payload for step, payload in session.logged if step == 0]
            self.assertEqual(len(step_zero_payloads), 1)

            payload = step_zero_payloads[0]
            self.assertIn("lr_base", payload)
            self.assertIn("lr_scale_factor", payload)
            self.assertIn("lr_applied", payload)
            self.assertIn("lr_scaling_active", payload)
            self.assertAlmostEqual(payload["lr_base"], 1e-3, places=12)
            self.assertAlmostEqual(payload["lr_scale_factor"], 2**0.5, places=9)
            self.assertAlmostEqual(payload["lr_applied"], 1e-3 * (2**0.5), places=9)
            self.assertEqual(payload["lr_scaling_active"], 1.0)
            self.assertIn("lr_scheduler_enabled", payload)
            self.assertIn("lr_scheduler_stage_count", payload)
            self.assertIn("lr_scheduler_total_optimizer_steps", payload)

    def test_logged_lr_current_follows_linear_scheduler_progression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="scheduler-progress-run",
                provider="wandb",
                effective_batch_size=2,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.25,
                            end_factor=1.0,
                            steps=4,
                        )
                    ]
                ),
                wandb_cfg=WandbMetricsConfig(
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=10,
                    val_every_n_steps=0,
                    attention_entropy_every_n_steps=10,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
            )

            model_pipeline(cfg)
            session = self.recording_adapter.sessions[-1]
            lr_values = [
                payload["lr_current"]
                for step, payload in session.logged
                if step is not None and "lr_current" in payload
            ]
            self.assertGreaterEqual(len(lr_values), 4)
            self.assertEqual(
                [round(value, 12) for value in lr_values[:4]],
                [
                    round(0.00025, 12),
                    round(0.0005, 12),
                    round(0.00075, 12),
                    round(0.001, 12),
                ],
            )

    def test_scheduler_state_restores_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name=None,
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.25,
                            end_factor=1.0,
                            steps=4,
                        )
                    ]
                ),
            )
            model = torch.nn.Linear(2, 1, bias=False)
            restored_model = torch.nn.Linear(2, 1, bias=False)
            resolved = resolve_train_learning_rate(cfg.train)

            optimizer = build_optimizer(
                model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )
            scheduler = build_lr_scheduler(
                optimizer,
                cfg,
                total_optimizer_steps=10,
            )
            self.assertIsNotNone(scheduler)
            optimizer.step()
            scheduler.step()
            optimizer.step()
            scheduler.step()
            saved_optimizer_state = optimizer.state_dict()
            saved_scheduler_state = scheduler.state_dict()

            restored_optimizer = build_optimizer(
                restored_model,
                cfg,
                learning_rate=resolved.applied_learning_rate,
            )
            restored_scheduler = build_lr_scheduler(
                restored_optimizer,
                cfg,
                total_optimizer_steps=10,
            )
            self.assertIsNotNone(restored_scheduler)
            restored_optimizer.load_state_dict(saved_optimizer_state)
            restored_scheduler.load_state_dict(saved_scheduler_state)

            self.assertAlmostEqual(
                float(restored_optimizer.param_groups[0]["lr"]),
                float(optimizer.param_groups[0]["lr"]),
                places=12,
            )
            restored_optimizer.step()
            restored_scheduler.step()
            optimizer.step()
            scheduler.step()
            self.assertAlmostEqual(
                float(restored_optimizer.param_groups[0]["lr"]),
                float(optimizer.param_groups[0]["lr"]),
                places=12,
            )

    def test_checkpoint_includes_scheduler_state_when_scheduler_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="scheduler-checkpoint-run",
                provider="console",
                effective_batch_size=4,
                lr_scheduler=LRSchedulerChainConfig(
                    stages=[
                        LRSchedulerStageConfig(
                            type="linear",
                            start_factor=0.25,
                            end_factor=1.0,
                            steps=4,
                        )
                    ]
                ),
                checkpoint_every_n_steps=1,
            )

            result = model_pipeline(cfg)
            checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
            self.assertIn("scheduler_state_dict", checkpoint)
            self.assertIsNotNone(checkpoint["scheduler_state_dict"])

    def test_disabling_step_timing_avoids_sync_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                tmp_path=Path(tmpdir),
                run_name="no-step-timing-run",
                provider="wandb",
                effective_batch_size=4,
                wandb_cfg=WandbMetricsConfig(
                    enable_step_time=False,
                    enable_peak_memory=False,
                    log_every_n_steps=1,
                    diagnostics_every_n_steps=1,
                    val_every_n_steps=0,
                    attention_entropy_every_n_steps=1,
                    attention_entropy_head_cap=1,
                    attention_entropy_token_cap=8,
                ),
            )

            sync_calls = 0

            def _record_sync(_device: torch.device) -> None:
                nonlocal sync_calls
                sync_calls += 1

            with mock.patch("src.train.synchronize_if_cuda", side_effect=_record_sync):
                model_pipeline(cfg)

            self.assertEqual(sync_calls, 0)


if __name__ == "__main__":
    unittest.main()
