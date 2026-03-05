from __future__ import annotations

import time
from collections import defaultdict
from typing import Iterable

from .contracts import (
    EpochMetricsContext,
    MetricPayload,
    MetricPlugin,
    PeriodicValMetricsContext,
    StepMetricsContext,
)


class MetricsEngine:
    def __init__(
        self,
        plugins: Iterable[MetricPlugin],
        *,
        enable_timing_debug: bool = False,
    ) -> None:
        self._plugins = list(plugins)
        self._enable_timing_debug = enable_timing_debug
        self._timings_ms: dict[str, dict[str, float]] = defaultdict(dict)

    def _call_lifecycle(self, phase: str, method_name: str) -> None:
        for plugin in self._plugins:
            method = getattr(plugin, method_name)
            if self._enable_timing_debug:
                start = time.perf_counter()
                method()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._timings_ms[phase][plugin.name] = (
                    self._timings_ms[phase].get(plugin.name, 0.0) + elapsed_ms
                )
                continue
            method()

    def _call_lifecycle_with_ctx(self, phase: str, ctx, method_name: str) -> None:
        for plugin in self._plugins:
            method = getattr(plugin, method_name)
            if self._enable_timing_debug:
                start = time.perf_counter()
                method(ctx)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._timings_ms[phase][plugin.name] = (
                    self._timings_ms[phase].get(plugin.name, 0.0) + elapsed_ms
                )
                continue
            method(ctx)

    def _collect(
        self,
        phase: str,
        ctx,
        method_name: str,
    ) -> MetricPayload:
        merged: MetricPayload = {}
        producers: dict[str, str] = {}
        for plugin in self._plugins:
            method = getattr(plugin, method_name)
            if self._enable_timing_debug:
                start = time.perf_counter()
                metrics = dict(method(ctx))
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._timings_ms[phase][plugin.name] = (
                    self._timings_ms[phase].get(plugin.name, 0.0) + elapsed_ms
                )
            else:
                metrics = dict(method(ctx))

            for key, value in metrics.items():
                existing = producers.get(key)
                if existing is not None:
                    raise ValueError(
                        "Metric key collision during "
                        f"{phase}: key '{key}' emitted by plugins "
                        f"'{existing}' and '{plugin.name}'."
                    )
                producers[key] = plugin.name
                merged[key] = value
        return merged

    def on_train_start(self) -> None:
        self._call_lifecycle("on_train_start", "on_train_start")

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        self._call_lifecycle_with_ctx("on_step_start", ctx, "on_step_start")

    def after_backward(self, ctx: StepMetricsContext) -> None:
        self._call_lifecycle_with_ctx("after_backward", ctx, "after_backward")

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        self._call_lifecycle_with_ctx(
            "after_optimizer_step",
            ctx,
            "after_optimizer_step",
        )

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        return self._collect("collect_step_metrics", ctx, "collect_step_metrics")

    def collect_periodic_val_metrics(
        self,
        ctx: PeriodicValMetricsContext,
    ) -> MetricPayload:
        return self._collect(
            "collect_periodic_val_metrics",
            ctx,
            "collect_periodic_val_metrics",
        )

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> MetricPayload:
        return self._collect("collect_epoch_metrics", ctx, "collect_epoch_metrics")

    def on_train_end(self) -> None:
        first_error: Exception | None = None
        for plugin in self._plugins:
            try:
                if self._enable_timing_debug:
                    start = time.perf_counter()
                    plugin.on_train_end()
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._timings_ms["on_train_end"][plugin.name] = (
                        self._timings_ms["on_train_end"].get(plugin.name, 0.0)
                        + elapsed_ms
                    )
                else:
                    plugin.on_train_end()
            except Exception as exc:  # pragma: no cover - defensive cleanup path.
                if first_error is None:
                    first_error = exc

        if self._enable_timing_debug:
            self._print_timing_summary()

        if first_error is not None:
            raise first_error

    def timing_snapshot(self) -> dict[str, dict[str, float]]:
        return {
            phase: dict(plugin_times)
            for phase, plugin_times in self._timings_ms.items()
        }

    def _print_timing_summary(self) -> None:
        print("MetricsEngine timing summary (ms):")
        for phase in sorted(self._timings_ms):
            print(f"  {phase}:")
            phase_totals = self._timings_ms[phase]
            for plugin_name, elapsed in sorted(
                phase_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                print(f"    {plugin_name}: {elapsed:.3f}")
