from __future__ import annotations

import torch
from torch import optim

from baseline.core.config import WandbMetricsConfig

from ..contracts import BaseMetricPlugin, MetricPayload, StepMetricsContext


def _to_cpu_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32)


def _accumulate_l2_sq_cpu(
    total: torch.Tensor | None,
    tensor_cpu_float: torch.Tensor,
) -> torch.Tensor:
    tensor_sq = tensor_cpu_float.pow(2).sum()
    return tensor_sq if total is None else total + tensor_sq


def _accumulate_l2_sq(total: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor:
    tensor_sq = _to_cpu_float_tensor(tensor).pow(2).sum()
    return tensor_sq if total is None else total + tensor_sq


def _finalize_l2_norm(total_sq: torch.Tensor | None) -> float | None:
    if total_sq is None:
        return None
    return float(total_sq.sqrt().item())


def _get_bias_corrected_moments(
    state: dict,
    beta1: float,
    beta2: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    step_val = state.get("step")

    if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
        return None

    m_hat = _to_cpu_float_tensor(exp_avg)
    v_hat = _to_cpu_float_tensor(exp_avg_sq)

    if step_val is not None:
        t = float(step_val.item() if torch.is_tensor(step_val) else step_val)
        if t > 0.0:
            m_hat = m_hat / (1 - beta1**t)
            v_hat = v_hat / (1 - beta2**t)

    return m_hat, v_hat


def compute_global_param_norm(model: torch.nn.Module) -> float | None:
    total_sq: torch.Tensor | None = None
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            total_sq = _accumulate_l2_sq(total_sq, param)
    return _finalize_l2_norm(total_sq)


def compute_layer_param_norms(
    model: torch.nn.Module,
    layer_labels: dict[int, list[str]],
) -> MetricPayload:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    metrics: MetricPayload = {}
    with torch.no_grad():
        for layer_idx, labels in layer_labels.items():
            layer = dec_layers[layer_idx]
            layer_sq: torch.Tensor | None = None
            for param in layer.parameters():
                layer_sq = _accumulate_l2_sq(layer_sq, param)

            layer_norm = _finalize_l2_norm(layer_sq)
            if layer_norm is None:
                continue
            for label in labels:
                metrics[f"layer_param_norm_{label}"] = layer_norm
    return metrics


def compute_param_update_norm(
    model: torch.nn.Module,
    pre_step_param_snapshot: dict[int, torch.Tensor],
) -> float | None:
    if not pre_step_param_snapshot:
        return None

    update_sq: torch.Tensor | None = None
    with torch.no_grad():
        for param in model.parameters():
            pre_step_value = pre_step_param_snapshot.get(id(param))
            if pre_step_value is None:
                continue
            current_value = _to_cpu_float_tensor(param)
            update = current_value - pre_step_value
            update_sq = _accumulate_l2_sq_cpu(update_sq, update)
    return _finalize_l2_norm(update_sq)


def compute_adam_state_norms(optimizer: optim.Optimizer) -> MetricPayload:

    if not isinstance(optimizer, optim.Adam) and not isinstance(optimizer, optim.AdamW):
        return {}

    m_sq: torch.Tensor | None = None
    v_sq: torch.Tensor | None = None
    snr_sq: torch.Tensor | None = None

    # TODO:@Ash - WARNING: This assumes all parameter groups use the same betas, which is true for our current configs but may not hold in general. We could add support for per-group betas if needed.
    beta1, beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999))

    with torch.no_grad():
        for state in optimizer.state.values():
            moments = _get_bias_corrected_moments(state, beta1, beta2)
            if moments is None:
                continue

            m_hat, v_hat = moments
            m_sq = _accumulate_l2_sq_cpu(m_sq, m_hat)
            v_sq = _accumulate_l2_sq_cpu(v_sq, v_hat)

            snr = m_hat / (v_hat.sqrt() + 1e-8)
            snr_sq = _accumulate_l2_sq_cpu(snr_sq, snr)

    m_norm = _finalize_l2_norm(m_sq)
    v_norm = _finalize_l2_norm(v_sq)
    snr_norm = _finalize_l2_norm(snr_sq)

    if m_norm is None or v_norm is None:
        return {}

    return {
        "adam_m_norm": m_norm,
        "adam_v_norm": v_norm,
        "adam_elemwise_snr_norm": float(snr_norm) if snr_norm is not None else 0.0,
    }


def compute_layer_estimated_variance_norms(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    layer_labels: dict[int, list[str]],
) -> MetricPayload:
    dec_layers = getattr(model, "dec_layers", None)
    if dec_layers is None:
        return {}

    metrics: MetricPayload = {}
    beta1, beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999))
    with torch.no_grad():
        for layer_idx, labels in layer_labels.items():
            layer = dec_layers[layer_idx]
            layer_v_sq: torch.Tensor | None = None
            for param in layer.parameters():
                state = optimizer.state.get(param)
                if not state:
                    continue
                moments = _get_bias_corrected_moments(state, beta1, beta2)
                if moments is None:
                    continue
                _, v_hat = moments
                layer_v_sq = _accumulate_l2_sq_cpu(layer_v_sq, v_hat)

            layer_v_norm = _finalize_l2_norm(layer_v_sq)
            if layer_v_norm is None:
                continue
            for label in labels:
                metrics[f"layer_estimated_variance_norm_{label}"] = layer_v_norm
    return metrics


class ParameterOptimizerNormsPlugin(BaseMetricPlugin):
    name = "parameter_optimizer_norms"

    def __init__(
        self,
        *,
        wandb_cfg: WandbMetricsConfig,
        model: torch.nn.Module,
        optimizer: optim.Optimizer,
        layer_labels: dict[int, list[str]],
    ) -> None:
        self._wandb_cfg = wandb_cfg
        self._model = model
        self._optimizer = optimizer
        self._layer_labels = layer_labels
        self._metrics: MetricPayload = {}
        self._pre_step_param_snapshot: dict[int, torch.Tensor] = {}
        self._pre_step_global_param_norm: float | None = None

    def _should_collect(self, ctx: StepMetricsContext) -> bool:
        if not ctx.schedule.should_log_parameter_optimizer_norms:
            return False
        return any(
            (
                self._wandb_cfg.enable_global_param_norm,
                self._wandb_cfg.enable_layer_param_norms,
                self._wandb_cfg.enable_param_update_norm,
                self._wandb_cfg.enable_update_to_weight_ratio,
                self._wandb_cfg.enable_optimizer_state_norms,
            )
        )

    def on_step_start(self, ctx: StepMetricsContext) -> None:
        self._metrics = {}
        self._pre_step_param_snapshot = {}
        self._pre_step_global_param_norm = None

        if not self._should_collect(ctx):
            return
        if not (
            self._wandb_cfg.enable_param_update_norm
            or self._wandb_cfg.enable_update_to_weight_ratio
        ):
            return

        with torch.no_grad():
            track_pre_norm = self._wandb_cfg.enable_update_to_weight_ratio
            pre_norm_sq: torch.Tensor | None = None
            for param in self._model.parameters():
                if not param.requires_grad:
                    continue
                # Keep snapshots off-GPU to avoid diagnostics-step VRAM spikes.
                snapshot = _to_cpu_float_tensor(param).clone()
                self._pre_step_param_snapshot[id(param)] = snapshot
                if track_pre_norm:
                    pre_norm_sq = _accumulate_l2_sq_cpu(pre_norm_sq, snapshot)

            if track_pre_norm:
                self._pre_step_global_param_norm = _finalize_l2_norm(pre_norm_sq)

    def after_optimizer_step(self, ctx: StepMetricsContext) -> None:
        if not self._should_collect(ctx):
            return

        metrics: MetricPayload = {}

        try:
            with torch.no_grad():
                if self._wandb_cfg.enable_global_param_norm:
                    global_param_norm = compute_global_param_norm(self._model)
                    if global_param_norm is not None:
                        metrics["global_param_norm"] = global_param_norm

                if self._wandb_cfg.enable_layer_param_norms:
                    metrics.update(compute_layer_param_norms(self._model, self._layer_labels))

                update_norm: float | None = None
                if (
                    self._wandb_cfg.enable_param_update_norm
                    or self._wandb_cfg.enable_update_to_weight_ratio
                ):
                    update_norm = compute_param_update_norm(
                        self._model,
                        self._pre_step_param_snapshot,
                    )
                    if self._wandb_cfg.enable_param_update_norm and update_norm is not None:
                        metrics["param_update_norm"] = update_norm

                if (
                    self._wandb_cfg.enable_update_to_weight_ratio
                    and update_norm is not None
                    and self._pre_step_global_param_norm is not None
                    and self._pre_step_global_param_norm > 0.0
                ):
                    metrics["update_to_weight_ratio"] = float(
                        update_norm / self._pre_step_global_param_norm
                    )

                if self._wandb_cfg.enable_optimizer_state_norms:
                    metrics.update(compute_adam_state_norms(self._optimizer))
                    metrics.update(
                        compute_layer_estimated_variance_norms(
                            self._model,
                            self._optimizer,
                            self._layer_labels,
                        )
                    )
        finally:
            self._pre_step_param_snapshot = {}
            self._pre_step_global_param_norm = None

        self._metrics = metrics

    def on_train_end(self) -> None:
        self._metrics = {}
        self._pre_step_param_snapshot = {}
        self._pre_step_global_param_norm = None

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        _ = ctx
        return dict(self._metrics)
