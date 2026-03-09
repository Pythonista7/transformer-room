from __future__ import annotations

import time

import torch

from src.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


class StepTimingAndMemoryPlugin(BaseMetricPlugin):
    name = "step_timing_memory"

    def __init__(self, *, wandb_cfg: WandbMetricsConfig, device: torch.device) -> None:
        self._wandb_cfg = wandb_cfg
        self._device = device
        self._step_start_time: float | None = None

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        self._step_start_time = None
        if ctx.schedule.should_log_step_metrics and self._wandb_cfg.enable_step_time:
            self._step_start_time = time.perf_counter()
        if (
            ctx.schedule.should_log_step_metrics
            and self._wandb_cfg.enable_peak_memory
            and self._device.type == "cuda"
        ):
            torch.cuda.reset_peak_memory_stats(self._device)

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if not ctx.schedule.should_log_this_step:
            return {}

        if self._device.type == "cuda" and ctx.schedule.should_log_step_metrics and (
            self._wandb_cfg.enable_step_time or self._wandb_cfg.enable_peak_memory
        ):
            torch.cuda.synchronize(self._device)

        metrics: MetricPayload = {}
        if (
            ctx.schedule.should_log_step_metrics
            and self._wandb_cfg.enable_step_time
            and self._step_start_time is not None
        ):
            elapsed_ms = (time.perf_counter() - self._step_start_time) * 1000.0
            metrics["step_time_ms"] = float(elapsed_ms)
        if (
            ctx.schedule.should_log_step_metrics
            and self._wandb_cfg.enable_peak_memory
            and self._device.type == "cuda"
        ):
            peak_mem_gib = torch.cuda.max_memory_allocated(self._device) / (1024**3)
            metrics["peak_memory_gib"] = float(peak_mem_gib)
        return metrics
