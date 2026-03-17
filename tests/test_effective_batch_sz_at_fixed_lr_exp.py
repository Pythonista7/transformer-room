from __future__ import annotations

import unittest

import torch

import experiments.baseline.memory_experiments.effective_batch_sz_at_fixed_lr as fixed_lr_exp
from src.training.metrics import MetricSchedule, MicroBatchMetricsContext, StepMetricsContext


def _schedule() -> MetricSchedule:
    return MetricSchedule(
        should_log_step_metrics=True,
        should_log_diagnostics=True,
        should_log_parameter_optimizer_norms=True,
        should_log_attention_entropy=False,
        capture_activation_norms=False,
        capture_attention_entropy=False,
        should_log_this_step=True,
        periodic_val_due=False,
    )


def _step_ctx(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> StepMetricsContext:
    return StepMetricsContext(
        schedule=_schedule(),
        global_step=1,
        next_global_step=2,
        epoch=0,
        batch_idx=0,
        train_loader_len=10,
        tokens_seen_train=32,
        step_loss=1.0,
        include_in_perf_aggregates=True,
        model=model,
        optimizer=optimizer,
    )


class TrialSpecsTests(unittest.TestCase):
    def test_trial_specs_cover_all_effective_batch_sizes(self) -> None:
        specs = fixed_lr_exp.trial_specs()
        self.assertTrue(specs, "trial_specs() should return at least one trial.")

        accumulations = [accumulation_steps for accumulation_steps, _ in specs]
        self.assertEqual(len(accumulations), len(set(accumulations)))
        self.assertEqual(accumulations, sorted(accumulations))
        self.assertTrue(all(acc > 0 for acc in accumulations))

        for accumulation_steps, effective_batch_size in specs:
            self.assertEqual(effective_batch_size, 28 * accumulation_steps)

    def test_build_config_uses_sqrt_lr_scaling_only_when_accumulating(self) -> None:
        non_accum_cfg = fixed_lr_exp.build_config(
            run_name="non-accum",
            sweep_group="group-a",
            base_vocab_size=100,
            accumulation_steps=1,
        )
        accum_cfg = fixed_lr_exp.build_config(
            run_name="accum",
            sweep_group="group-a",
            base_vocab_size=100,
            accumulation_steps=2,
        )

        self.assertEqual(non_accum_cfg.train.lr_scaling, "none")
        self.assertEqual(accum_cfg.train.lr_scaling, "sqrt")


class GradientCoherencePluginTests(unittest.TestCase):
    def test_gradient_coherence_uses_per_token_microbatch_gradients(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctx = _step_ctx(model=model, optimizer=optimizer)

        plugin = fixed_lr_exp.GradientQualitySummaryPlugin()
        plugin.on_step_start(ctx)

        weight = model.weight
        weight.grad = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
        plugin.after_microbatch_backward(
            MicroBatchMetricsContext(
                step_ctx=ctx,
                micro_batch_in_step=1,
                accumulation_steps=2,
                valid_tokens=2,
            )
        )

        # Cumulative grad after second microbatch. Delta is [0, 4], and
        # per-token normalized delta is [0, 1], orthogonal to first [1, 0].
        weight.grad = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
        plugin.after_microbatch_backward(
            MicroBatchMetricsContext(
                step_ctx=ctx,
                micro_batch_in_step=2,
                accumulation_steps=2,
                valid_tokens=4,
            )
        )

        plugin.after_backward(ctx)
        metrics = plugin.collect_step_metrics(ctx)

        self.assertIn("gradient_coherence", metrics)
        self.assertAlmostEqual(metrics["gradient_coherence"], 0.0, places=6)
        self.assertEqual(metrics["gradient_coherence_pairs"], 1.0)

    def test_gradient_snapshots_are_kept_on_cpu(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctx = _step_ctx(model=model, optimizer=optimizer)

        plugin = fixed_lr_exp.GradientQualitySummaryPlugin()
        plugin.on_step_start(ctx)

        weight = model.weight
        weight.grad = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
        plugin.after_microbatch_backward(
            MicroBatchMetricsContext(
                step_ctx=ctx,
                micro_batch_in_step=1,
                accumulation_steps=2,
                valid_tokens=2,
            )
        )

        self.assertTrue(plugin._prev_cumulative_grads)
        self.assertTrue(plugin._prev_normalized_micro_grads)
        self.assertTrue(
            all(t.device.type == "cpu" for t in plugin._prev_cumulative_grads.values())
        )
        self.assertTrue(
            all(
                t.device.type == "cpu"
                for t in plugin._prev_normalized_micro_grads.values()
            )
        )

    def test_accumulation_one_reports_unity_coherence(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctx = _step_ctx(model=model, optimizer=optimizer)

        plugin = fixed_lr_exp.GradientQualitySummaryPlugin()
        plugin.on_step_start(ctx)

        weight = model.weight
        weight.grad = torch.tensor([[3.0, 6.0]], dtype=torch.float32)
        plugin.after_microbatch_backward(
            MicroBatchMetricsContext(
                step_ctx=ctx,
                micro_batch_in_step=1,
                accumulation_steps=1,
                valid_tokens=3,
            )
        )

        plugin.after_backward(ctx)
        metrics = plugin.collect_step_metrics(ctx)

        self.assertEqual(metrics["gradient_coherence"], 1.0)
        self.assertEqual(metrics["gradient_coherence_pairs"], 0.0)


if __name__ == "__main__":
    unittest.main()
