from __future__ import annotations

from statistics import mean
from typing import Mapping


def merge_logged_metrics_by_step(
    logged_entries: list[tuple[int | None, Mapping[str, float]]],
) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {}
    for step, payload in logged_entries:
        if step is None:
            continue
        step_metrics = merged.setdefault(int(step), {})
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                step_metrics[key] = float(value)
    return merged


def compute_avg_tokens_per_sec(
    logged_entries: list[tuple[int | None, Mapping[str, float]]],
) -> float | None:
    step_metrics = merge_logged_metrics_by_step(logged_entries)
    if not step_metrics:
        return None

    rates: list[float] = []
    prev_tokens_seen: float | None = None
    for step in sorted(step_metrics):
        metrics = step_metrics[step]
        tokens_seen = metrics.get("tokens_seen_train")
        step_time_ms = metrics.get("step_time_ms")
        if tokens_seen is None:
            continue
        if step_time_ms is None or step_time_ms <= 0:
            prev_tokens_seen = tokens_seen
            continue

        delta_tokens = (
            tokens_seen if prev_tokens_seen is None else tokens_seen - prev_tokens_seen
        )
        prev_tokens_seen = tokens_seen
        if delta_tokens <= 0:
            continue
        rates.append(delta_tokens / (step_time_ms / 1000.0))

    if not rates:
        return None
    return float(mean(rates))
