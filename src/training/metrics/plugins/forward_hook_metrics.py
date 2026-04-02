from __future__ import annotations

import torch

from src.components.attention.common import (
    attention_scale,
    build_attention_mask,
    reshape_for_multi_head,
)
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


def compute_attention_entropy_from_module_inputs(
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_head_cap: int,
    attention_token_cap: int,
    mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    is_causal: bool = True,
) -> torch.Tensor | None:
    packed_proj = getattr(attn, "packed_proj", None)
    n_heads = getattr(attn, "n_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if packed_proj is None or n_heads is None or head_dim is None:
        return None
    if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
        return None

    with torch.no_grad():
        batch_size, seq_len, _ = hidden_states.shape
        all_projs = packed_proj(hidden_states.detach())
        query, key, _ = torch.chunk(all_projs, 3, dim=-1)
        query = reshape_for_multi_head(
            query,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )
        key = reshape_for_multi_head(
            key,
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )

        sampled_heads = min(attention_head_cap, query.size(1))
        sampled_tokens = min(attention_token_cap, query.size(2), key.size(2))
        if sampled_heads <= 0 or sampled_tokens <= 0:
            return None

        attention_mask = build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            is_causal=bool(is_causal),
            mask=mask,
            key_padding_mask=key_padding_mask,
        )

        query = (
            query[:, :sampled_heads, :sampled_tokens, :]
            .detach()
            .to(dtype=torch.float32)
        )
        key = (
            key[:, :sampled_heads, :sampled_tokens, :]
            .detach()
            .to(dtype=torch.float32)
        )
        attention_mask = attention_mask[
            :,
            :,
            :sampled_tokens,
            :sampled_tokens,
        ].detach()

        scores = (query @ key.transpose(-2, -1)) / attention_scale(int(head_dim))
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        fully_masked = ~attention_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)

        probs = torch.softmax(scores, dim=-1).clamp_min(1e-12)
        return -(probs * probs.log()).sum(dim=-1).mean()


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
        if attn is None:
            continue

        def attention_entropy_hook(
            _module,
            args,
            kwargs,
            label_tuple=label_tuple,
            attn=attn,
        ):
            if not collector.capture_attention_entropy:
                return
            if not args:
                return
            hidden_states = args[0]
            if kwargs is None:
                kwargs = {}
            entropy = compute_attention_entropy_from_module_inputs(
                attn,
                hidden_states,
                attention_head_cap=attention_head_cap,
                attention_token_cap=attention_token_cap,
                mask=kwargs.get("mask"),
                key_padding_mask=kwargs.get("key_padding_mask"),
                is_causal=bool(kwargs.get("is_causal", True)),
            )
            if entropy is None:
                return
            for label in label_tuple:
                collector.attention_entropy[f"attention_entropy_{label}"] = entropy

        handles.append(
            attn.register_forward_pre_hook(attention_entropy_hook, with_kwargs=True)
        )

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
