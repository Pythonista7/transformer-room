from __future__ import annotations

import math

from src.core.config import WandbMetricsConfig

from ..contracts import (
    BaseMetricPlugin,
    EpochMetricsContext,
    MetricPayload,
    PeriodicValMetricsContext,
    StepMetricsContext,
)


class LossPerplexityPlugin(BaseMetricPlugin):
    name = "loss_perplexity"

    def __init__(self, *, wandb_enabled: bool, wandb_cfg: WandbMetricsConfig) -> None:
        self._wandb_enabled = wandb_enabled
        self._wandb_cfg = wandb_cfg

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if not ctx.schedule.should_log_this_step or ctx.step_loss is None:
            return {}

        metrics: MetricPayload = {"epoch": ctx.epoch_progress}
        if self._wandb_cfg.enable_train_loss_vs_tokens:
            metrics["train_loss_step"] = float(ctx.step_loss)
            metrics["tokens_seen_train"] = float(ctx.tokens_seen_train)

        if self._wandb_cfg.enable_perplexity:
            bounded = min(float(ctx.step_loss), 60.0)
            metrics["train_perplexity"] = float(math.exp(bounded))
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        return metrics

    def collect_periodic_val_metrics(self, ctx: PeriodicValMetricsContext) -> MetricPayload:
        if not ctx.schedule.periodic_val_due:
            return {}

        metrics: MetricPayload = {
            "epoch": ctx.epoch_progress,
        }
        if self._wandb_cfg.enable_val_loss_vs_tokens:
            metrics["val_loss"] = float(ctx.val_metrics["val_loss"])
            metrics["tokens_seen_train"] = float(ctx.tokens_seen_train)
        if self._wandb_cfg.enable_perplexity:
            metrics["val_perplexity"] = float(ctx.val_metrics["val_perplexity"])
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        return metrics

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> MetricPayload:
        metrics: MetricPayload = {
            "epoch": float(ctx.epoch + 1),
            "train_loss_epoch": float(ctx.avg_train_loss),
        }
        if (not self._wandb_enabled) or self._wandb_cfg.enable_val_loss_vs_tokens:
            metrics["val_loss"] = float(ctx.val_metrics["val_loss"])
        if (not self._wandb_enabled) or self._wandb_cfg.enable_perplexity:
            metrics["val_perplexity"] = float(ctx.val_metrics["val_perplexity"])
        if self._wandb_enabled and self._wandb_cfg.enable_train_loss_vs_tokens:
            metrics["tokens_seen_train"] = float(ctx.tokens_seen_train)
        if self._wandb_enabled and self._wandb_cfg.enable_perplexity:
            bounded_epoch_loss = min(float(ctx.avg_train_loss), 60.0)
            metrics["train_perplexity_epoch"] = float(math.exp(bounded_epoch_loss))
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        return metrics
