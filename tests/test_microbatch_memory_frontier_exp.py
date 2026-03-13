from __future__ import annotations

import unittest

import torch

from experiments.baseline.memory_experiments.microbatch_memory_frontier_exp import (
    MemoryFrontierSummaryPlugin,
    TrialResult,
    adaptive_find_frontier,
    classify_oom_exception,
    compute_avg_tokens_per_sec,
)
from src.training.metrics import MetricSchedule, StepMetricsContext


def _trial(
    *,
    micro_batch_size: int,
    status: str,
) -> TrialResult:
    return TrialResult(
        budget_label="none",
        activation_memory_budget=None,
        micro_batch_size=micro_batch_size,
        run_name=f"run-mb-{micro_batch_size}",
        status=status,
    )


class AdaptiveFrontierSearchTests(unittest.TestCase):
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
            max_micro_batch=64,
            trial_runner=runner,
        )

        self.assertEqual(frontier, 40)
        self.assertIn(40, attempts)
        self.assertEqual(len(attempts), len(set(attempts)))
        self.assertEqual({trial.micro_batch_size for trial in trials}, set(attempts))

    def test_adaptive_search_handles_no_fit_at_minimum(self) -> None:
        attempts: list[int] = []

        def runner(micro_batch_size: int) -> TrialResult:
            attempts.append(micro_batch_size)
            return _trial(micro_batch_size=micro_batch_size, status="oom")

        frontier, trials = adaptive_find_frontier(
            min_micro_batch=16,
            max_micro_batch=128,
            trial_runner=runner,
        )

        self.assertIsNone(frontier)
        self.assertEqual(attempts, [16])
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0].status, "oom")


class OOMClassifierTests(unittest.TestCase):
    def test_classify_oom_exception(self) -> None:
        self.assertTrue(classify_oom_exception(torch.OutOfMemoryError("oom")))
        self.assertTrue(
            classify_oom_exception(RuntimeError("CUDA out of memory. Tried to allocate"))
        )
        self.assertFalse(classify_oom_exception(RuntimeError("unexpected shape mismatch")))


class ThroughputDerivationTests(unittest.TestCase):
    def test_compute_avg_tokens_per_sec_from_logged_series(self) -> None:
        logged_entries = [
            (1, {"tokens_seen_train": 512.0}),
            (1, {"step_time_ms": 200.0}),
            (2, {"tokens_seen_train": 1024.0, "step_time_ms": 100.0}),
            (3, {"tokens_seen_train": 1536.0, "step_time_ms": 100.0}),
        ]
        # Rates by step:
        # step1:  512 / 0.2 = 2560
        # step2:  512 / 0.1 = 5120
        # step3:  512 / 0.1 = 5120
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


if __name__ == "__main__":
    unittest.main()
