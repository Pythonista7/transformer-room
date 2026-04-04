from __future__ import annotations

import torch

from src.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


def get_sampled_decoder_layer_indices(
    model: torch.nn.Module,
    *,
    stride: int,
) -> tuple[int, ...]:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return ()

    layer_count = len(dec_layers)
    if layer_count <= 0:
        return ()

    selected = set(range(0, layer_count, stride))
    selected.update({0, layer_count // 2, layer_count - 1})
    return tuple(sorted(idx for idx in selected if 0 <= idx < layer_count))


def compute_layer_grad_norms(
    model: torch.nn.Module,
    layer_indices: tuple[int, ...],
) -> MetricPayload:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    metrics: MetricPayload = {}
    for layer_idx in layer_indices:
        layer = dec_layers[layer_idx]        
        layer_param_grads = torch._foreach_norm([p.grad for p in layer.parameters() if p.grad is not None])
        stack = torch.stack(layer_param_grads)
        layer_grad_norm = stack.norm()
        metrics[f"layer_grad_norm_layer_{layer_idx}"] = float(layer_grad_norm.item())
    return metrics


class LayerGradNormPlugin(BaseMetricPlugin):
    name = "layer_grad_norm"

    def __init__(
        self,
        *,
        wandb_cfg: WandbMetricsConfig,
        model: torch.nn.Module,
    ) -> None:
        self._wandb_cfg = wandb_cfg
        self._model = model
        self._layer_indices = get_sampled_decoder_layer_indices(
            model,
            stride=int(wandb_cfg.layer_grad_norm_stride),
        )
        self._metrics: MetricPayload = {}

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        _ = ctx
        self._metrics = {}

    def after_backward(self, ctx: StepMetricsContext) -> None:
        if (
            ctx.schedule.should_log_layer_grad_norms
            and self._wandb_cfg.enable_layer_grad_norms
        ):
            self._metrics = compute_layer_grad_norms(self._model, self._layer_indices)

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        _ = ctx
        return dict(self._metrics)
