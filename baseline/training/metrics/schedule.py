from __future__ import annotations

from dataclasses import dataclass

from baseline.core.config import WandbMetricsConfig


@dataclass(frozen=True, slots=True)
class MetricSchedule:
    should_log_step_metrics: bool
    should_log_diagnostics: bool
    should_log_parameter_optimizer_norms: bool
    should_log_attention_entropy: bool
    capture_activation_norms: bool
    capture_attention_entropy: bool
    should_log_this_step: bool
    periodic_val_due: bool


def should_log_every(step: int, every_n_steps: int) -> bool:
    return step > 0 and step % every_n_steps == 0


def build_metric_schedule(
    *,
    next_global_step: int,
    wandb_enabled: bool,
    wandb_cfg: WandbMetricsConfig,
    layer_labels_available: bool,
) -> MetricSchedule:
    parameter_optimizer_metrics_enabled = any(
        (
            wandb_cfg.enable_global_param_norm,
            wandb_cfg.enable_layer_param_norms,
            wandb_cfg.enable_param_update_norm,
            wandb_cfg.enable_update_to_weight_ratio,
            wandb_cfg.enable_optimizer_state_norms,
        )
    )
    parameter_optimizer_norms_every_n_steps = (
        wandb_cfg.parameter_optimizer_norms_every_n_steps
        if wandb_cfg.parameter_optimizer_norms_every_n_steps is not None
        else wandb_cfg.diagnostics_every_n_steps
    )
    should_log_step_metrics = (
        wandb_enabled and should_log_every(next_global_step, wandb_cfg.log_every_n_steps)
    )
    should_log_diagnostics = (
        wandb_enabled and should_log_every(next_global_step, wandb_cfg.diagnostics_every_n_steps)
    )
    should_log_parameter_optimizer_norms = (
        wandb_enabled
        and parameter_optimizer_metrics_enabled
        and should_log_every(next_global_step, parameter_optimizer_norms_every_n_steps)
    )
    should_log_attention_entropy = (
        wandb_enabled
        and wandb_cfg.enable_attention_entropy
        and should_log_every(next_global_step, wandb_cfg.attention_entropy_every_n_steps)
    )

    capture_activation_norms = (
        should_log_diagnostics
        and wandb_cfg.enable_activation_norms
        and layer_labels_available
    )
    capture_attention_entropy = should_log_attention_entropy and layer_labels_available
    should_log_this_step = (
        should_log_step_metrics
        or should_log_diagnostics
        or should_log_parameter_optimizer_norms
        or should_log_attention_entropy
    )

    periodic_val_enabled = (
        wandb_enabled
        and wandb_cfg.val_every_n_steps > 0
        and (wandb_cfg.enable_val_loss_vs_tokens or wandb_cfg.enable_perplexity)
    )
    periodic_val_due = periodic_val_enabled and should_log_every(
        next_global_step,
        wandb_cfg.val_every_n_steps,
    )

    return MetricSchedule(
        should_log_step_metrics=should_log_step_metrics,
        should_log_diagnostics=should_log_diagnostics,
        should_log_parameter_optimizer_norms=should_log_parameter_optimizer_norms,
        should_log_attention_entropy=should_log_attention_entropy,
        capture_activation_norms=capture_activation_norms,
        capture_attention_entropy=capture_attention_entropy,
        should_log_this_step=should_log_this_step,
        periodic_val_due=periodic_val_due,
    )
