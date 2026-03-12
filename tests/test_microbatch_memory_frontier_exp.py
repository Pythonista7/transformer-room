from __future__ import annotations

import unittest

import torch

from experiments.baseline.memory_experiments.microbatch_memory_frontier_exp import (
    NoArtifactCaptureWandbSession,
    TrialResult,
    adaptive_find_frontier,
    classify_oom_exception,
    compute_avg_tokens_per_sec,
)


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


class NoArtifactSessionTests(unittest.TestCase):
    def test_save_is_suppressed_but_log_and_close_forward(self) -> None:
        class _BaseSession:
            def __init__(self) -> None:
                self.logged: list[tuple[int | None, dict[str, float]]] = []
                self.save_calls = 0
                self.closed = False

            def log(
                self,
                metrics: dict[str, float],
                step: int | None = None,
            ) -> None:
                self.logged.append((step, dict(metrics)))

            def save(self, *args, **kwargs):
                self.save_calls += 1
                _ = args
                _ = kwargs
                return "artifact-ref"

            def restore(self, *args, **kwargs) -> bool:
                _ = args
                _ = kwargs
                return True

            def watch(self, model, loss_fn) -> None:
                _ = model
                _ = loss_fn

            def close(self) -> None:
                self.closed = True

        base = _BaseSession()
        wrapped = NoArtifactCaptureWandbSession(base_session=base)

        wrapped.log({"step_time_ms": 12.0}, step=1)
        saved = wrapped.save("dummy.pt", artifact_name="x", artifact_type="model")
        restored = wrapped.restore("dummy.pt", artifact_name="x")
        wrapped.close()

        self.assertEqual(saved, None)
        self.assertFalse(restored)
        self.assertEqual(base.save_calls, 0)
        self.assertEqual(len(base.logged), 1)
        self.assertEqual(len(wrapped.logged_entries), 1)
        self.assertTrue(base.closed)


if __name__ == "__main__":
    unittest.main()
