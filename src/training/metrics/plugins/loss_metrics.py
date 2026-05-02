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


class LossMetricsPlugin(BaseMetricPlugin):
    name = "loss_metrics"

    def __init__(self, *, wandb_enabled: bool, wandb_cfg: WandbMetricsConfig) -> None:
        self._wandb_enabled = wandb_enabled
        self._wandb_cfg = wandb_cfg

    def collect_step_metrics(self, ctx: StepMetricsContext) -> MetricPayload:
        if not ctx.schedule.should_log_this_step:
            return {}

        metrics: MetricPayload = {"epoch": ctx.epoch_progress}
        if ctx.lr_current is not None:
            metrics["lr_current"] = float(ctx.lr_current)
        if ctx.step_loss is None:
            return metrics

        if self._wandb_cfg.enable_train_loss_vs_tokens:
            metrics["train_loss_step"] = float(ctx.step_loss)
            metrics["tokens_seen_train"] = float(ctx.tokens_seen_train)

        if self._wandb_cfg.enable_perplexity:
            bounded = min(float(ctx.step_loss), 60.0)
            metrics["train_perplexity"] = float(math.exp(bounded))
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        if self._wandb_cfg.enable_bits_per_byte and ctx.step_bits_per_byte is not None:
            metrics["train_bits_per_byte"] = float(ctx.step_bits_per_byte)
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        return metrics

    def collect_periodic_val_metrics(self, ctx: PeriodicValMetricsContext) -> MetricPayload:
        if not ctx.schedule.periodic_val_due:
            return {}

        metrics: MetricPayload = {
            "epoch": ctx.epoch_progress,
        }
        emitted_val_metric = False
        if self._wandb_cfg.enable_val_loss_vs_tokens:
            for key, value in ctx.val_metrics.items():
                if key.endswith("/val_loss"):
                    metrics[key] = float(value)
                    emitted_val_metric = True
        if self._wandb_cfg.enable_perplexity:
            for key, value in ctx.val_metrics.items():
                if key.endswith("/val_perplexity"):
                    metrics[key] = float(value)
                    emitted_val_metric = True
        if (
            self._wandb_cfg.enable_bits_per_byte
        ):
            for key, value in ctx.val_metrics.items():
                if key.endswith("/val_bits_per_byte"):
                    metrics[key] = float(value)
                    emitted_val_metric = True
        if emitted_val_metric:
            metrics["tokens_seen_train"] = float(ctx.tokens_seen_train)
        return metrics

    def collect_epoch_metrics(self, ctx: EpochMetricsContext) -> MetricPayload:
        metrics: MetricPayload = {
            "epoch": float(ctx.epoch + 1),
            "train_loss_epoch": float(ctx.avg_train_loss),
        }
        include_val_loss = (not self._wandb_enabled) or self._wandb_cfg.enable_val_loss_vs_tokens
        include_perplexity = (not self._wandb_enabled) or self._wandb_cfg.enable_perplexity
        include_bits_per_byte = (not self._wandb_enabled) or self._wandb_cfg.enable_bits_per_byte
        if include_val_loss:
            for key, value in ctx.val_metrics.items():
                if key.endswith("/val_loss"):
                    metrics[key] = float(value)
        if include_perplexity:
            for key, value in ctx.val_metrics.items():
                if key.endswith("/val_perplexity"):
                    metrics[key] = float(value)
        if include_bits_per_byte:
            for key, value in ctx.val_metrics.items():
                if key.endswith("/val_bits_per_byte"):
                    metrics[key] = float(value)
        if self._wandb_enabled and self._wandb_cfg.enable_train_loss_vs_tokens:
            metrics["tokens_seen_train"] = float(ctx.tokens_seen_train)
        if self._wandb_enabled and self._wandb_cfg.enable_perplexity:
            bounded_epoch_loss = min(float(ctx.avg_train_loss), 60.0)
            metrics["train_perplexity_epoch"] = float(math.exp(bounded_epoch_loss))
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        if (
            self._wandb_enabled
            and self._wandb_cfg.enable_bits_per_byte
            and ctx.train_bits_per_byte_epoch is not None
        ):
            metrics["train_bits_per_byte_epoch"] = float(ctx.train_bits_per_byte_epoch)
            metrics.setdefault("tokens_seen_train", float(ctx.tokens_seen_train))
        return metrics
