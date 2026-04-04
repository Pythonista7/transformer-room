from __future__ import annotations

from dataclasses import dataclass

import torch

from src.components.attention.common import (
    attention_scale,
    build_attention_mask,
    reshape_for_multi_head,
)
from src.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


@dataclass(slots=True)
class AttentionProjectionStash:
    all_projs: torch.Tensor
    mask: torch.Tensor | None
    key_padding_mask: torch.Tensor | None
    is_causal: bool
    n_heads: int
    head_dim: int


class ForwardMetricCollector:
    def __init__(
        self,
        *,
        attention_head_cap: int,
        attention_token_cap: int,
    ) -> None:
        self.capture_activation_norms = False
        self.capture_attention_entropy = False
        self.attention_head_cap = int(attention_head_cap)
        self.attention_token_cap = int(attention_token_cap)
        self.activation_norms: dict[str, torch.Tensor] = {}
        self._attention_metric_keys_by_layer: dict[int, tuple[str, ...]] = {}
        self._attention_stash_by_layer: dict[int, AttentionProjectionStash] = {}
        self._attention_captured_layers: set[int] = set()

    def begin_step(
        self,
        *,
        capture_activation_norms: bool,
        capture_attention_entropy: bool,
    ) -> None:
        self.capture_activation_norms = capture_activation_norms
        self.capture_attention_entropy = capture_attention_entropy
        self.clear_step_state()

    def clear_step_state(self) -> None:
        self.activation_norms.clear()
        self._attention_stash_by_layer.clear()
        self._attention_captured_layers.clear()

    def finish_step(self) -> None:
        self.capture_activation_norms = False
        self.capture_attention_entropy = False
        self.clear_step_state()

    def register_attention_layer(self, layer_idx: int, labels: tuple[str, ...]) -> None:
        self._attention_metric_keys_by_layer[layer_idx] = tuple(
            f"attention_entropy_{label}" for label in labels
        )

    def stash_attention_projection_once(
        self,
        *,
        layer_idx: int,
        all_projs: torch.Tensor,
        mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        n_heads: int,
        head_dim: int,
    ) -> None:
        if not self.capture_attention_entropy:
            return
        if layer_idx in self._attention_captured_layers:
            return
        if layer_idx not in self._attention_metric_keys_by_layer:
            return
        if not torch.is_tensor(all_projs) or all_projs.dim() != 3:
            return

        stash = AttentionProjectionStash(
            all_projs=all_projs.detach(),
            mask=mask.detach() if torch.is_tensor(mask) else None,
            key_padding_mask=(
                key_padding_mask.detach()
                if torch.is_tensor(key_padding_mask)
                else None
            ),
            is_causal=bool(is_causal),
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )
        self._attention_stash_by_layer[layer_idx] = stash
        self._attention_captured_layers.add(layer_idx)

    def build_attention_stash_callback(self, layer_idx: int):
        def _callback(
            *,
            all_projs: torch.Tensor,
            mask: torch.Tensor | None,
            key_padding_mask: torch.Tensor | None,
            is_causal: bool,
            n_heads: int,
            head_dim: int,
        ) -> None:
            self.stash_attention_projection_once(
                layer_idx=layer_idx,
                all_projs=all_projs,
                mask=mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                n_heads=n_heads,
                head_dim=head_dim,
            )

        return _callback

    def take_metrics(self) -> MetricPayload:
        metrics: MetricPayload = {}
        for key, value in self.activation_norms.items():
            metrics[key] = float(value.item())
        for layer_idx, stash in self._attention_stash_by_layer.items():
            entropy = compute_attention_entropy_from_stash(
                stash,
                attention_head_cap=self.attention_head_cap,
                attention_token_cap=self.attention_token_cap,
            )
            if entropy is None:
                continue
            metric_keys = self._attention_metric_keys_by_layer.get(layer_idx, ())
            for key in metric_keys:
                metrics[key] = float(entropy.item())
        self.clear_step_state()
        return metrics


def compute_attention_entropy_from_stash(
    stash: AttentionProjectionStash,
    *,
    attention_head_cap: int,
    attention_token_cap: int,
) -> torch.Tensor | None:
    all_projs = stash.all_projs
    if all_projs.dim() != 3:
        return None
    if stash.n_heads <= 0 or stash.head_dim <= 0:
        return None

    with torch.no_grad():
        batch_size, seq_len, _ = all_projs.shape
        query, key, _ = torch.chunk(all_projs, 3, dim=-1)
        expected_dim = int(stash.n_heads) * int(stash.head_dim)
        if query.size(-1) != expected_dim:
            return None
        query = reshape_for_multi_head(
            query,
            n_heads=int(stash.n_heads),
            head_dim=int(stash.head_dim),
        )
        key = reshape_for_multi_head(
            key,
            n_heads=int(stash.n_heads),
            head_dim=int(stash.head_dim),
        )

        sampled_heads = min(int(attention_head_cap), query.size(1))
        sampled_tokens = min(int(attention_token_cap), query.size(2), key.size(2))
        if sampled_heads <= 0 or sampled_tokens <= 0:
            return None

        attention_mask = build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=all_projs.device,
            is_causal=bool(stash.is_causal),
            mask=stash.mask,
            key_padding_mask=stash.key_padding_mask,
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

        scores = (query @ key.transpose(-2, -1)) / attention_scale(int(stash.head_dim))
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        fully_masked = ~attention_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)

        probs = torch.softmax(scores, dim=-1).clamp_min(1e-12)
        return -(probs * probs.log()).sum(dim=-1).mean()


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
        all_projs = packed_proj(hidden_states.detach())
        stash = AttentionProjectionStash(
            all_projs=all_projs,
            mask=mask,
            key_padding_mask=key_padding_mask,
            is_causal=bool(is_causal),
            n_heads=int(n_heads),
            head_dim=int(head_dim),
        )
        return compute_attention_entropy_from_stash(
            stash,
            attention_head_cap=attention_head_cap,
            attention_token_cap=attention_token_cap,
        )


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
    *,
    enable_activation_norms: bool,
    enable_attention_entropy: bool,
) -> tuple[list[torch.utils.hooks.RemovableHandle], list[torch.nn.Module]]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    entropy_callback_modules: list[torch.nn.Module] = []
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return handles, entropy_callback_modules

    for layer_idx, labels in layer_labels.items():
        layer = dec_layers[layer_idx]
        label_tuple = tuple(labels)
        collector.register_attention_layer(layer_idx, label_tuple)

        def activation_hook(_module, _inputs, output, label_tuple=label_tuple):
            if not collector.capture_activation_norms:
                return
            if not torch.is_tensor(output):
                return
            activation_norm = output.detach().float().pow(2).mean().sqrt()
            for label in label_tuple:
                collector.activation_norms[f"activation_norm_{label}"] = activation_norm

        if enable_activation_norms:
            handles.append(layer.register_forward_hook(activation_hook))

        attn = getattr(layer, "multi_head_attention", None)
        if attn is None:
            continue

        if enable_attention_entropy:
            setattr(
                attn,
                "_forward_metric_entropy_capture",
                collector.build_attention_stash_callback(layer_idx),
            )
            entropy_callback_modules.append(attn)

    return handles, entropy_callback_modules


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
        self._collector = ForwardMetricCollector(
            attention_head_cap=self._wandb_cfg.attention_entropy_head_cap,
            attention_token_cap=self._wandb_cfg.attention_entropy_token_cap,
        )
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._entropy_callback_modules: list[torch.nn.Module] = []

    def on_train_start(self) -> None:
        if self._wandb_enabled and self._layer_labels and (
            self._wandb_cfg.enable_activation_norms or self._wandb_cfg.enable_attention_entropy
        ):
            (
                self._hook_handles,
                self._entropy_callback_modules,
            ) = register_forward_metric_hooks(
                model=self._model,
                collector=self._collector,
                layer_labels=self._layer_labels,
                enable_activation_norms=self._wandb_cfg.enable_activation_norms,
                enable_attention_entropy=self._wandb_cfg.enable_attention_entropy,
            )

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        self._collector.begin_step(
            capture_activation_norms=ctx.schedule.capture_activation_norms,
            capture_attention_entropy=ctx.schedule.capture_attention_entropy,
        )

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if ctx.schedule.capture_activation_norms or ctx.schedule.capture_attention_entropy:
            try:
                return self._collector.take_metrics()
            finally:
                self._collector.finish_step()
        return {}

    def on_train_end(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        for module in self._entropy_callback_modules:
            if hasattr(module, "_forward_metric_entropy_capture"):
                setattr(module, "_forward_metric_entropy_capture", None)
        self._entropy_callback_modules = []
        self._collector.finish_step()
