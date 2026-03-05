from __future__ import annotations

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


class ExampleMetricPlugin(BaseMetricPlugin):
    """Template plugin for adding new scalar metrics."""

    name = "example_metric"

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if not ctx.schedule.should_log_this_step:
            return {}
        # Emit unique keys; collisions are rejected by MetricsEngine.
        return {
            "example_metric_value": 0.0,
        }
