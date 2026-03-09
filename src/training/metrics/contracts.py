from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

MetricPayload = dict[str, float]


@dataclass(slots=True)
class StepMetricsContext:
    schedule: MetricSchedule
    global_step: int
    next_global_step: int
    epoch: int
    batch_idx: int
    train_loader_len: int
    tokens_seen_train: int
    step_loss: float | None
    step_time_ms: float | None = None
    forward_pass_time_ms: float | None = None
    backward_pass_time_ms: float | None = None
    optim_step_time_ms: float | None = None
    peak_memory_gib: float | None = None
    peak_reserved_memory_gib: float | None = None
    include_in_perf_aggregates: bool = True

    @property
    def epoch_progress(self) -> float:
        return float(self.epoch + (self.batch_idx + 1) / max(self.train_loader_len, 1))


@dataclass(slots=True)
class PeriodicValMetricsContext:
    schedule: MetricSchedule
    global_step: int
    epoch: int
    batch_idx: int
    train_loader_len: int
    tokens_seen_train: int
    val_metrics: Mapping[str, float]

    @property
    def epoch_progress(self) -> float:
        return float(self.epoch + (self.batch_idx + 1) / max(self.train_loader_len, 1))


@dataclass(slots=True)
class EpochMetricsContext:
    global_step: int
    epoch: int
    avg_train_loss: float
    tokens_seen_train: int
    val_metrics: Mapping[str, float]
    epoch_time_s: float | None = None


class MetricPlugin(Protocol):
    name: str

    def on_train_start(self) -> None: ...

    def on_step_start(self, ctx: StepMetricsContext) -> None: ...

    def after_backward(self, ctx: StepMetricsContext) -> None: ...

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None: ...

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload: ...

    def collect_periodic_val_metrics(
        self,
        ctx: PeriodicValMetricsContext,
    ) -> MetricPayload: ...

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> MetricPayload: ...

    def on_train_end(self) -> None: ...


class BaseMetricPlugin:
    name = "base"

    def on_train_start(self) -> None:
        return

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        _ = ctx

    def after_backward(self, ctx: StepMetricsContext) -> None:
        _ = ctx

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        _ = ctx

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        _ = ctx
        return {}

    def collect_periodic_val_metrics(self, ctx: PeriodicValMetricsContext) -> MetricPayload:
        _ = ctx
        return {}

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> MetricPayload:
        _ = ctx
        return {}

    def on_train_end(self) -> None:
        return


from .schedule import MetricSchedule
