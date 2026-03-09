from __future__ import annotations

import torch

from src.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


class ForwardMetricCollector:
    def __init__(self) -> None:
        self.capture_activation_norms = False
        self.capture_attention_entropy = False
        self.activation_norms: dict[str, torch.Tensor] = {}
        self.attention_entropy: dict[str, torch.Tensor] = {}

    def begin_step(
        self,
        *,
        capture_activation_norms: bool,
        capture_attention_entropy: bool,
    ) -> None:
        self.capture_activation_norms = capture_activation_norms
        self.capture_attention_entropy = capture_attention_entropy
        self.activation_norms.clear()
        self.attention_entropy.clear()

    def take_metrics(self) -> MetricPayload:
        metrics: MetricPayload = {}
        for key, value in self.activation_norms.items():
            metrics[key] = float(value.item())
        for key, value in self.attention_entropy.items():
            metrics[key] = float(value.item())
        self.activation_norms.clear()
        self.attention_entropy.clear()
        return metrics


def get_decoder_layer_labels(model: torch.nn.Module) -> dict[int, list[str]]:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    layer_count = len(dec_layers)
    if layer_count <= 0:
        return {}

    selected = [
        ("first", 0),
        ("middle", layer_count // 2),
        ("last", layer_count - 1),
    ]

    layer_labels: dict[int, list[str]] = {}
    for label, idx in selected:
        layer_labels.setdefault(idx, []).append(label)
    return layer_labels


def register_forward_metric_hooks(
    model: torch.nn.Module,
    collector: ForwardMetricCollector,
    layer_labels: dict[int, list[str]],
    attention_head_cap: int,
    attention_token_cap: int,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return handles

    for layer_idx, labels in layer_labels.items():
        layer = dec_layers[layer_idx]
        label_tuple = tuple(labels)

        def activation_hook(_module, _inputs, output, label_tuple=label_tuple):
            if not collector.capture_activation_norms:
                return
            if not torch.is_tensor(output):
                return
            activation_norm = output.detach().float().pow(2).mean().sqrt()
            for label in label_tuple:
                collector.activation_norms[f"activation_norm_{label}"] = activation_norm

        handles.append(layer.register_forward_hook(activation_hook))

        attn = getattr(layer, "multi_head_attention", None)
        softmax_module = getattr(attn, "softmax", None)
        if softmax_module is None:
            continue

        def attention_entropy_hook(_module, _inputs, output, label_tuple=label_tuple):
            if not collector.capture_attention_entropy:
                return
            if not torch.is_tensor(output) or output.dim() != 4:
                return

            _, heads, query_len, key_len = output.shape
            sampled_heads = min(attention_head_cap, heads)
            sampled_tokens = min(attention_token_cap, query_len, key_len)
            if sampled_heads <= 0 or sampled_tokens <= 0:
                return

            probs = output[:, :sampled_heads, :sampled_tokens, :sampled_tokens].detach().float()
            probs = probs.clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            for label in label_tuple:
                collector.attention_entropy[f"attention_entropy_{label}"] = entropy

        handles.append(softmax_module.register_forward_hook(attention_entropy_hook))

    return handles


class ForwardHookMetricsPlugin(BaseMetricPlugin):
    name = "forward_hook_metrics"

    def __init__(
        self,
        *,
        wandb_enabled: bool,
        wandb_cfg: WandbMetricsConfig,
        model: torch.nn.Module,
        layer_labels: dict[int, list[str]],
    ) -> None:
        self._wandb_enabled = wandb_enabled
        self._wandb_cfg = wandb_cfg
        self._model = model
        self._layer_labels = layer_labels
        self._collector = ForwardMetricCollector()
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def on_train_start(self) -> None:
        if self._wandb_enabled and self._layer_labels and (
            self._wandb_cfg.enable_activation_norms or self._wandb_cfg.enable_attention_entropy
        ):
            self._hook_handles = register_forward_metric_hooks(
                model=self._model,
                collector=self._collector,
                layer_labels=self._layer_labels,
                attention_head_cap=self._wandb_cfg.attention_entropy_head_cap,
                attention_token_cap=self._wandb_cfg.attention_entropy_token_cap,
            )

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        self._collector.begin_step(
            capture_activation_norms=ctx.schedule.capture_activation_norms,
            capture_attention_entropy=ctx.schedule.capture_attention_entropy,
        )

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if ctx.schedule.capture_activation_norms or ctx.schedule.capture_attention_entropy:
            return self._collector.take_metrics()
        return {}

    def on_train_end(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
