from __future__ import annotations

import torch

from baseline.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


def compute_layernorm_grad_norms(
    model: torch.nn.Module,
    layer_labels: dict[int, list[str]],
) -> MetricPayload:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    metrics: MetricPayload = {}
    for layer_idx, labels in layer_labels.items():
        layer = dec_layers[layer_idx]

        weight_sq: torch.Tensor | None = None
        for grad in (layer.ln1.gamma.grad, layer.ln2.gamma.grad):
            if grad is None:
                continue
            grad_sq = grad.detach().float().pow(2).sum()
            weight_sq = grad_sq if weight_sq is None else weight_sq + grad_sq

        bias_sq: torch.Tensor | None = None
        for grad in (layer.ln1.beta.grad, layer.ln2.beta.grad):
            if grad is None:
                continue
            grad_sq = grad.detach().float().pow(2).sum()
            bias_sq = grad_sq if bias_sq is None else bias_sq + grad_sq

        for label in labels:
            if weight_sq is not None:
                metrics[f"ln_weight_grad_norm_{label}"] = float(weight_sq.sqrt().item())
            if bias_sq is not None:
                metrics[f"ln_bias_grad_norm_{label}"] = float(bias_sq.sqrt().item())

    return metrics


class LayerNormGradNormPlugin(BaseMetricPlugin):
    name = "layernorm_grad_norm"

    def __init__(
        self,
        *,
        wandb_cfg: WandbMetricsConfig,
        model: torch.nn.Module,
        layer_labels: dict[int, list[str]],
    ) -> None:
        self._wandb_cfg = wandb_cfg
        self._model = model
        self._layer_labels = layer_labels
        self._metrics: MetricPayload = {}

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self._metrics = {}

    def after_backward(self, ctx: StepMetricsContext) -> None:
        if ctx.schedule.should_log_diagnostics and self._wandb_cfg.enable_ln_grad_norms:
            self._metrics = compute_layernorm_grad_norms(self._model, self._layer_labels)

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        _ = ctx
        return dict(self._metrics)
