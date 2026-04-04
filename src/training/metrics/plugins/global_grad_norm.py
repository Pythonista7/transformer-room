from __future__ import annotations

import torch

from src.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


def compute_global_grad_norm(model: torch.nn.Module) -> float | None:
    grads = []
    for param in model.parameters():
        if param.grad is None:
            continue
        grads.append(param.grad)

    norms = torch._foreach_norm(grads)
    total = torch.stack(norms).norm()
    return float(total.item())

class GlobalGradNormPlugin(BaseMetricPlugin):
    name = "global_grad_norm"

    def __init__(self, *, wandb_cfg: WandbMetricsConfig, model: torch.nn.Module) -> None:
        self._wandb_cfg = wandb_cfg
        self._model = model
        self._value: float | None = None

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self._value = None

    def after_backward(self, ctx: StepMetricsContext) -> None:
        if ctx.schedule.should_log_diagnostics and self._wandb_cfg.enable_global_grad_norm:
            self._value = compute_global_grad_norm(self._model)

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        _ = ctx
        if self._value is None:
            return {}
        return {"global_grad_norm": float(self._value)}
