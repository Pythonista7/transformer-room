from __future__ import annotations

import unittest

from baseline.training.metrics import (
    BaseMetricPlugin,
    EpochMetricsContext,
    MetricSchedule,
    MetricsEngine,
    PeriodicValMetricsContext,
    StepMetricsContext,
)


def _schedule() -> MetricSchedule:
    return MetricSchedule(
        should_log_step_metrics=True,
        should_log_diagnostics=True,
        should_log_attention_entropy=False,
        capture_activation_norms=False,
        capture_attention_entropy=False,
        should_log_this_step=True,
        periodic_val_due=True,
    )


def _step_ctx() -> StepMetricsContext:
    return StepMetricsContext(
        schedule=_schedule(),
        global_step=1,
        next_global_step=2,
        epoch=0,
        batch_idx=0,
        train_loader_len=10,
        tokens_seen_train=16,
        step_loss=1.25,
    )


class _PluginA(BaseMetricPlugin):
    name = "plugin_a"

    def collect_step_metrics(self, ctx: StepMetricsContext) -> dict[str, float]:
        _ = ctx
        return {"a": 1.0}


class _PluginB(BaseMetricPlugin):
    name = "plugin_b"

    def collect_step_metrics(self, ctx: StepMetricsContext) -> dict[str, float]:
        _ = ctx
        return {"b": 2.0}


class _CollisionPlugin(BaseMetricPlugin):
    name = "collision"

    def collect_step_metrics(self, ctx: StepMetricsContext) -> dict[str, float]:
        _ = ctx
        return {"a": 9.0}


class _LifecyclePlugin(BaseMetricPlugin):
    name = "lifecycle"

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self) -> None:
        self.events.append("start")

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self.events.append("step_start")

    def after_backward(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self.events.append("after_backward")

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self.events.append("after_optimizer")

    def collect_periodic_val_metrics(
        self,
        ctx: PeriodicValMetricsContext,
    ) -> dict[str, float]:
        _ = ctx
        self.events.append("periodic")
        return {"p": 3.0}

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> dict[str, float]:
        _ = ctx
        self.events.append("epoch")
        return {"e": 4.0}

    def on_train_end(self) -> None:
        self.events.append("end")


class MetricsEngineTests(unittest.TestCase):
    def test_collect_step_metrics_deterministic_merge(self) -> None:
        engine = MetricsEngine([_PluginA(), _PluginB()])
        metrics = engine.collect_step_metrics(_step_ctx())
        self.assertEqual(list(metrics.keys()), ["a", "b"])
        self.assertEqual(metrics, {"a": 1.0, "b": 2.0})

    def test_collect_step_metrics_key_collision_raises(self) -> None:
        engine = MetricsEngine([_PluginA(), _CollisionPlugin()])
        with self.assertRaisesRegex(ValueError, "Metric key collision"):
            engine.collect_step_metrics(_step_ctx())

    def test_full_lifecycle_and_timing_snapshot(self) -> None:
        lifecycle = _LifecyclePlugin()
        engine = MetricsEngine([lifecycle], enable_timing_debug=True)
        step_ctx = _step_ctx()
        val_ctx = PeriodicValMetricsContext(
            schedule=_schedule(),
            global_step=1,
            epoch=0,
            batch_idx=0,
            train_loader_len=10,
            tokens_seen_train=16,
            val_metrics={"val_loss": 1.0, "val_perplexity": 2.0},
        )
        epoch_ctx = EpochMetricsContext(
            global_step=1,
            epoch=0,
            avg_train_loss=1.2,
            tokens_seen_train=16,
            val_metrics={"val_loss": 1.0, "val_perplexity": 2.0},
        )

        engine.on_train_start()
        engine.on_step_start(step_ctx)
        engine.after_backward(step_ctx)
        engine.after_optimizer_step(step_ctx)
        self.assertEqual(engine.collect_periodic_val_metrics(val_ctx), {"p": 3.0})
        self.assertEqual(engine.collect_epoch_metrics(epoch_ctx), {"e": 4.0})
        engine.on_train_end()

        self.assertEqual(
            lifecycle.events,
            [
                "start",
                "step_start",
                "after_backward",
                "after_optimizer",
                "periodic",
                "epoch",
                "end",
            ],
        )
        timing = engine.timing_snapshot()
        self.assertIn("on_train_start", timing)
        self.assertIn("collect_epoch_metrics", timing)


if __name__ == "__main__":
    unittest.main()
