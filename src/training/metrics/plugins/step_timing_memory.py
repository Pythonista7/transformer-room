from __future__ import annotations

import torch

from src.core.config import WandbMetricsConfig

from ..contracts import (
    BaseMetricPlugin,
    EpochMetricsContext,
    MetricPayload,
    StepMetricsContext,
)


class StepTimingAndMemoryPlugin(BaseMetricPlugin):
    name = "step_timing_memory"

    def __init__(self, *, wandb_cfg: WandbMetricsConfig, device: torch.device) -> None:
        self._wandb_cfg = wandb_cfg
        self._device = device
        self._timing_totals_ms: dict[str, float] = {}
        self._timing_counts: dict[str, int] = {}

    def _reset_epoch_aggregates(self) -> None:
        self._timing_totals_ms = {
            "step_time_ms": 0.0,
            "forward_pass_time_ms": 0.0,
            "backward_pass_time_ms": 0.0,
            "optim_step_time_ms": 0.0,
        }
        self._timing_counts = {
            "step_time_ms": 0,
            "forward_pass_time_ms": 0,
            "backward_pass_time_ms": 0,
            "optim_step_time_ms": 0,
        }

    def on_train_start(self) -> None:
        self._reset_epoch_aggregates()

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        if (
            self._wandb_cfg.enable_peak_memory
            and self._device.type == "cuda"
        ):
            torch.cuda.reset_peak_memory_stats(self._device)

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        if not self._wandb_cfg.enable_step_time:
            return
        if not ctx.include_in_perf_aggregates:
            return

        timing_values = {
            "step_time_ms": ctx.step_time_ms,
            "forward_pass_time_ms": ctx.forward_pass_time_ms,
            "backward_pass_time_ms": ctx.backward_pass_time_ms,
            "optim_step_time_ms": ctx.optim_step_time_ms,
        }
        for key, value in timing_values.items():
            if value is None:
                continue
            self._timing_totals_ms[key] += float(value)
            self._timing_counts[key] += 1

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if not ctx.schedule.should_log_this_step:
            return {}

        metrics: MetricPayload = {}
        if (
            ctx.schedule.should_log_step_metrics
            and self._wandb_cfg.enable_step_time
            and ctx.include_in_perf_aggregates
        ):
            if ctx.step_time_ms is not None:
                metrics["step_time_ms"] = float(ctx.step_time_ms)
            if ctx.forward_pass_time_ms is not None:
                metrics["forward_pass_time_ms"] = float(ctx.forward_pass_time_ms)
            if ctx.backward_pass_time_ms is not None:
                metrics["backward_pass_time_ms"] = float(ctx.backward_pass_time_ms)
            if ctx.optim_step_time_ms is not None:
                metrics["optim_step_time_ms"] = float(ctx.optim_step_time_ms)
        if (
            ctx.schedule.should_log_step_metrics
            and self._wandb_cfg.enable_peak_memory
            and ctx.peak_memory_gib is not None
        ):
            metrics["peak_memory_gib"] = float(ctx.peak_memory_gib)
            if ctx.peak_reserved_memory_gib is not None:
                metrics["peak_reserved_memory_gib"] = float(
                    ctx.peak_reserved_memory_gib
                )
        return metrics

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> MetricPayload:
        if not self._wandb_cfg.enable_step_time:
            self._reset_epoch_aggregates()
            return {}

        metrics: MetricPayload = {}
        if ctx.epoch_time_s is not None:
            metrics["epoch_time_s"] = float(ctx.epoch_time_s)

        average_metric_names = {
            "step_time_ms": "avg_step_time_ms_epoch",
            "forward_pass_time_ms": "avg_forward_pass_time_ms_epoch",
            "backward_pass_time_ms": "avg_backward_pass_time_ms_epoch",
            "optim_step_time_ms": "avg_optim_step_time_ms_epoch",
        }
        for source_key, target_key in average_metric_names.items():
            count = self._timing_counts.get(source_key, 0)
            if count <= 0:
                continue
            metrics[target_key] = float(self._timing_totals_ms[source_key] / count)

        self._reset_epoch_aggregates()
        return metrics
