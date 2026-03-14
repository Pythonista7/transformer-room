from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import experiments.baseline.memory_experiments.microbatch_memory_frontier_exp as frontier_exp
from experiments.baseline.memory_experiments.microbatch_memory_frontier_exp import (
    MemoryFrontierSummaryPlugin,
    TrialResult,
    _to_row_value,
    adaptive_find_frontier,
    classify_oom_exception,
    compute_avg_tokens_per_sec,
    run_trial,
)
from src.training.metrics import MetricSchedule, StepMetricsContext


def _trial(
    *,
    micro_batch_size: int,
    status: str,
) -> TrialResult:
    return TrialResult(
        micro_batch_size=micro_batch_size,
        run_name=f"run-mb-{micro_batch_size}",
        status=status,
    )


class AdaptiveFrontierSearchTests(unittest.TestCase):
    def test_default_initial_max_micro_batch_is_128(self) -> None:
        self.assertEqual(frontier_exp.INITIAL_MAX_MICRO_BATCH, 128)

    def test_adaptive_search_finds_exact_boundary(self) -> None:
        attempts: list[int] = []

        def runner(micro_batch_size: int) -> TrialResult:
            attempts.append(micro_batch_size)
            return _trial(
                micro_batch_size=micro_batch_size,
                status="success" if micro_batch_size <= 40 else "oom",
            )

        frontier, trials = adaptive_find_frontier(
            min_micro_batch=8,
            initial_max_micro_batch=64,
            trial_runner=runner,
        )

        self.assertEqual(frontier, 40)
        self.assertIn(40, attempts)
        self.assertEqual(len(attempts), len(set(attempts)))
        self.assertEqual({trial.micro_batch_size for trial in trials}, set(attempts))

    def test_adaptive_search_expands_past_initial_max_until_first_oom(self) -> None:
        attempts: list[int] = []

        def runner(micro_batch_size: int) -> TrialResult:
            attempts.append(micro_batch_size)
            return _trial(
                micro_batch_size=micro_batch_size,
                status="success" if micro_batch_size <= 160 else "oom",
            )

        frontier, trials = adaptive_find_frontier(
            min_micro_batch=16,
            initial_max_micro_batch=64,
            trial_runner=runner,
        )

        self.assertEqual(frontier, 160)
        self.assertIn(128, attempts)
        self.assertIn(256, attempts)
        self.assertEqual(len(attempts), len(set(attempts)))
        self.assertEqual({trial.micro_batch_size for trial in trials}, set(attempts))

    def test_progress_callback_fires_once_per_new_trial_not_cached_reads(self) -> None:
        attempts: list[int] = []
        callback_frontiers: list[int | None] = []
        callback_sizes: list[int] = []

        def runner(micro_batch_size: int) -> TrialResult:
            attempts.append(micro_batch_size)
            return _trial(
                micro_batch_size=micro_batch_size,
                status="success" if micro_batch_size <= 100 else "oom",
            )

        def progress_callback(
            trial_results: list[TrialResult],
            best_known_frontier: int | None,
        ) -> None:
            callback_frontiers.append(best_known_frontier)
            callback_sizes.append(len(trial_results))

        frontier, _ = adaptive_find_frontier(
            min_micro_batch=16,
            initial_max_micro_batch=64,
            trial_runner=runner,
            progress_callback=progress_callback,
        )

        self.assertEqual(frontier, 100)
        self.assertEqual(len(callback_frontiers), len(attempts))
        self.assertEqual(callback_sizes, list(range(1, len(attempts) + 1)))
        self.assertEqual(callback_frontiers[-1], frontier)

    def test_adaptive_search_handles_no_fit_at_minimum(self) -> None:
        attempts: list[int] = []

        def runner(micro_batch_size: int) -> TrialResult:
            attempts.append(micro_batch_size)
            return _trial(micro_batch_size=micro_batch_size, status="oom")

        frontier, trials = adaptive_find_frontier(
            min_micro_batch=16,
            initial_max_micro_batch=128,
            trial_runner=runner,
        )

        self.assertIsNone(frontier)
        self.assertEqual(attempts, [16])
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].status, "oom")

    def test_adaptive_search_aborts_on_non_oom_error_status(self) -> None:
        attempts: list[int] = []

        def runner(micro_batch_size: int) -> TrialResult:
            attempts.append(micro_batch_size)
            if micro_batch_size <= 16:
                return _trial(micro_batch_size=micro_batch_size, status="success")
            return TrialResult(
                micro_batch_size=micro_batch_size,
                run_name=f"run-mb-{micro_batch_size}",
                status="error",
                error_type="UnexpectedError",
                error_message="training failed",
            )

        with self.assertRaisesRegex(RuntimeError, "status=error"):
            adaptive_find_frontier(
                min_micro_batch=16,
                initial_max_micro_batch=64,
                trial_runner=runner,
            )
        self.assertEqual(attempts, [16, 32])


class OOMClassifierTests(unittest.TestCase):
    def test_classify_oom_exception(self) -> None:
        self.assertTrue(classify_oom_exception(torch.OutOfMemoryError("oom")))
        self.assertTrue(
            classify_oom_exception(RuntimeError("CUDA out of memory. Tried to allocate"))
        )
        self.assertFalse(classify_oom_exception(RuntimeError("unexpected shape mismatch")))


class WandbTableRowValueTests(unittest.TestCase):
    def test_none_values_remain_null_for_wandb_table_schema(self) -> None:
        self.assertIsNone(_to_row_value(None))
        self.assertEqual(_to_row_value(5), 5)
        self.assertEqual(_to_row_value(1.5), 1.5)
        self.assertEqual(_to_row_value("error"), "error")


class ThroughputDerivationTests(unittest.TestCase):
    def test_compute_avg_tokens_per_sec_from_logged_series(self) -> None:
        logged_entries = [
            (1, {"tokens_seen_train": 512.0}),
            (1, {"step_time_ms": 200.0}),
            (2, {"tokens_seen_train": 1024.0, "step_time_ms": 100.0}),
            (3, {"tokens_seen_train": 1536.0, "step_time_ms": 100.0}),
        ]
        expected = (2560.0 + 5120.0 + 5120.0) / 3.0
        observed = compute_avg_tokens_per_sec(logged_entries)
        self.assertIsNotNone(observed)
        self.assertAlmostEqual(observed or 0.0, expected, places=6)


class MemoryFrontierSummaryPluginTests(unittest.TestCase):
    def test_summary_aggregates_peak_memory_timing_and_tokens_per_sec(self) -> None:
        schedule = MetricSchedule(
            should_log_step_metrics=True,
            should_log_diagnostics=False,
            should_log_parameter_optimizer_norms=False,
            should_log_attention_entropy=False,
            capture_activation_norms=False,
            capture_attention_entropy=False,
            should_log_this_step=True,
            periodic_val_due=False,
        )

        plugin = MemoryFrontierSummaryPlugin()
        step_contexts = [
            StepMetricsContext(
                schedule=schedule,
                global_step=1,
                next_global_step=1,
                epoch=0,
                batch_idx=0,
                train_loader_len=3,
                tokens_seen_train=512,
                step_loss=1.0,
                step_time_ms=200.0,
                peak_memory_gib=10.0,
                peak_reserved_memory_gib=12.0,
            ),
            StepMetricsContext(
                schedule=schedule,
                global_step=2,
                next_global_step=2,
                epoch=0,
                batch_idx=1,
                train_loader_len=3,
                tokens_seen_train=1024,
                step_loss=1.0,
                step_time_ms=100.0,
                peak_memory_gib=11.0,
                peak_reserved_memory_gib=15.0,
            ),
            StepMetricsContext(
                schedule=schedule,
                global_step=3,
                next_global_step=3,
                epoch=0,
                batch_idx=2,
                train_loader_len=3,
                tokens_seen_train=1536,
                step_loss=1.0,
                step_time_ms=100.0,
                peak_memory_gib=9.0,
                peak_reserved_memory_gib=14.0,
            ),
        ]
        for ctx in step_contexts:
            plugin.after_optimizer_step(ctx)

        summary = plugin.summary
        self.assertEqual(summary.max_peak_memory_gib, 11.0)
        self.assertEqual(summary.max_peak_reserved_memory_gib, 15.0)
        self.assertAlmostEqual(summary.avg_step_time_ms or 0.0, 133.33333333333334, places=6)
        expected_tokens_per_sec = (2560.0 + 5120.0 + 5120.0) / 3.0
        self.assertAlmostEqual(summary.avg_tokens_per_sec or 0.0, expected_tokens_per_sec, places=6)

    def test_summary_skips_compile_warmup_steps(self) -> None:
        schedule = MetricSchedule(
            should_log_step_metrics=True,
            should_log_diagnostics=False,
            should_log_parameter_optimizer_norms=False,
            should_log_attention_entropy=False,
            capture_activation_norms=False,
            capture_attention_entropy=False,
            should_log_this_step=True,
            periodic_val_due=False,
        )

        plugin = MemoryFrontierSummaryPlugin()
        step_contexts = [
            StepMetricsContext(
                schedule=schedule,
                global_step=1,
                next_global_step=1,
                epoch=0,
                batch_idx=0,
                train_loader_len=3,
                tokens_seen_train=512,
                step_loss=1.0,
                step_time_ms=10_000.0,
                peak_memory_gib=99.0,
                peak_reserved_memory_gib=100.0,
                include_in_perf_aggregates=False,
            ),
            StepMetricsContext(
                schedule=schedule,
                global_step=2,
                next_global_step=2,
                epoch=0,
                batch_idx=1,
                train_loader_len=3,
                tokens_seen_train=1024,
                step_loss=1.0,
                step_time_ms=200.0,
                peak_memory_gib=10.0,
                peak_reserved_memory_gib=12.0,
                include_in_perf_aggregates=True,
            ),
            StepMetricsContext(
                schedule=schedule,
                global_step=3,
                next_global_step=3,
                epoch=0,
                batch_idx=2,
                train_loader_len=3,
                tokens_seen_train=1536,
                step_loss=1.0,
                step_time_ms=100.0,
                peak_memory_gib=11.0,
                peak_reserved_memory_gib=14.0,
                include_in_perf_aggregates=True,
            ),
        ]
        for ctx in step_contexts:
            plugin.after_optimizer_step(ctx)

        summary = plugin.summary
        self.assertEqual(summary.max_peak_memory_gib, 11.0)
        self.assertEqual(summary.max_peak_reserved_memory_gib, 14.0)
        self.assertAlmostEqual(summary.avg_step_time_ms or 0.0, 150.0, places=6)
        self.assertAlmostEqual(summary.avg_tokens_per_sec or 0.0, 5120.0, places=6)


class RunTrialValidityTests(unittest.TestCase):
    def test_run_trial_marks_success_when_training_only_epoch_completes(self) -> None:
        mocked_run_result = SimpleNamespace(
            global_step=7,
            run_artifact_dir="/tmp/fake-run",
            final_train_loss=1.23,
            final_val_loss=float("nan"),
            completed_epochs=1,
            epoch_end_validation_ran=False,
        )
        with mock.patch(
            "experiments.baseline.memory_experiments.microbatch_memory_frontier_exp.model_pipeline",
            return_value=mocked_run_result,
        ):
            result = run_trial(
                sweep_group="test-sweep",
                micro_batch_size=40,
                base_vocab_size=128,
            )

        self.assertEqual(result.status, "success")
        self.assertIsNone(result.error_type)
        self.assertEqual(result.global_step, 7)
        self.assertAlmostEqual(result.final_train_loss or 0.0, 1.23, places=6)
        self.assertIsNone(result.final_val_loss)

    def test_run_trial_marks_error_when_epoch_is_incomplete(self) -> None:
        mocked_run_result = SimpleNamespace(
            global_step=7,
            run_artifact_dir="/tmp/fake-run",
            final_train_loss=2.34,
            final_val_loss=float("nan"),
            completed_epochs=0,
            epoch_end_validation_ran=False,
        )
        with mock.patch(
            "experiments.baseline.memory_experiments.microbatch_memory_frontier_exp.model_pipeline",
            return_value=mocked_run_result,
        ):
            result = run_trial(
                sweep_group="test-sweep",
                micro_batch_size=40,
                base_vocab_size=128,
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_type, "IncompleteEpochOrValidation")
        self.assertIn("completed_epochs=0", result.error_message or "")
        self.assertIn("requires_validation=False", result.error_message or "")
        self.assertAlmostEqual(result.final_train_loss or 0.0, 2.34, places=6)
        self.assertIsNone(result.final_val_loss)


class MainSummaryLoggingTests(unittest.TestCase):
    def test_main_logs_partial_snapshots_and_one_final_summary(self) -> None:
        def trial_stub(
            *,
            sweep_group: str,
            micro_batch_size: int,
            base_vocab_size: int,
        ) -> TrialResult:
            _ = (sweep_group, base_vocab_size)
            return TrialResult(
                micro_batch_size=micro_batch_size,
                run_name=f"run-mb-{micro_batch_size}",
                status="success" if micro_batch_size <= 40 else "oom",
            )

        with (
            mock.patch(
                "experiments.baseline.memory_experiments.microbatch_memory_frontier_exp.torch.cuda.is_available",
                return_value=True,
            ),
            mock.patch(
                "experiments.baseline.memory_experiments.microbatch_memory_frontier_exp.ensure_wikitext_vocab_file",
                return_value=128,
            ),
            mock.patch(
                "experiments.baseline.memory_experiments.microbatch_memory_frontier_exp.run_trial",
                side_effect=trial_stub,
            ),
            mock.patch(
                "experiments.baseline.memory_experiments.microbatch_memory_frontier_exp.log_wandb_summary_tables",
            ) as log_summary_mock,
        ):
            rc = frontier_exp.main()

        self.assertEqual(rc, 0)
        partial_calls = [
            call
            for call in log_summary_mock.call_args_list
            if call.kwargs.get("summary_stage") == "partial"
        ]
        final_calls = [
            call
            for call in log_summary_mock.call_args_list
            if call.kwargs.get("summary_stage") == "final"
        ]

        self.assertGreater(len(partial_calls), 0)
        self.assertEqual(len(final_calls), 1)
        self.assertEqual(final_calls[0].kwargs.get("frontier"), 40)
        self.assertTrue(all(call.kwargs.get("snapshot_id") for call in partial_calls))
        self.assertEqual(
            len({call.kwargs["snapshot_id"] for call in partial_calls}),
            len(partial_calls),
        )


if __name__ == "__main__":
    unittest.main()
