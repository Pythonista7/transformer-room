from __future__ import annotations

import unittest

from src.config import WandbMetricsConfig
from src.training.metrics.schedule import build_metric_schedule


class MetricScheduleTests(unittest.TestCase):
    def test_wandb_disabled_disables_all_step_logging(self) -> None:
        schedule = build_metric_schedule(
            next_global_step=10,
            wandb_enabled=False,
            wandb_cfg=WandbMetricsConfig(),
            layer_labels_available=True,
        )
        self.assertFalse(schedule.should_log_step_metrics)
        self.assertFalse(schedule.should_log_diagnostics)
        self.assertFalse(schedule.should_log_parameter_optimizer_norms)
        self.assertFalse(schedule.should_log_attention_entropy)
        self.assertFalse(schedule.capture_activation_norms)
        self.assertFalse(schedule.capture_attention_entropy)
        self.assertFalse(schedule.should_log_this_step)
        self.assertFalse(schedule.periodic_val_due)

    def test_step_and_diagnostic_cadence_match(self) -> None:
        cfg = WandbMetricsConfig(
            log_every_n_steps=5,
            diagnostics_every_n_steps=3,
            enable_global_param_norm=True,
            attention_entropy_every_n_steps=7,
            val_every_n_steps=11,
        )
        schedule = build_metric_schedule(
            next_global_step=15,
            wandb_enabled=True,
            wandb_cfg=cfg,
            layer_labels_available=True,
        )
        self.assertTrue(schedule.should_log_step_metrics)
        self.assertTrue(schedule.should_log_diagnostics)
        self.assertTrue(schedule.should_log_parameter_optimizer_norms)
        self.assertFalse(schedule.should_log_attention_entropy)
        self.assertTrue(schedule.capture_activation_norms)
        self.assertFalse(schedule.capture_attention_entropy)
        self.assertTrue(schedule.should_log_this_step)
        self.assertFalse(schedule.periodic_val_due)

    def test_param_optimizer_norms_cadence_can_override_diagnostics(self) -> None:
        cfg = WandbMetricsConfig(
            diagnostics_every_n_steps=6,
            parameter_optimizer_norms_every_n_steps=5,
            enable_param_update_norm=True,
            attention_entropy_every_n_steps=7,
        )
        schedule = build_metric_schedule(
            next_global_step=10,
            wandb_enabled=True,
            wandb_cfg=cfg,
            layer_labels_available=True,
        )
        self.assertFalse(schedule.should_log_diagnostics)
        self.assertTrue(schedule.should_log_parameter_optimizer_norms)
        self.assertFalse(schedule.should_log_attention_entropy)
        self.assertTrue(schedule.should_log_this_step)

    def test_periodic_val_due_requires_val_metric_enabled(self) -> None:
        cfg = WandbMetricsConfig(
            enable_val_loss_vs_tokens=False,
            enable_perplexity=False,
            val_every_n_steps=2,
        )
        schedule = build_metric_schedule(
            next_global_step=2,
            wandb_enabled=True,
            wandb_cfg=cfg,
            layer_labels_available=True,
        )
        self.assertFalse(schedule.periodic_val_due)

    def test_attention_capture_requires_labels(self) -> None:
        cfg = WandbMetricsConfig(
            diagnostics_every_n_steps=1,
            attention_entropy_every_n_steps=1,
        )
        schedule = build_metric_schedule(
            next_global_step=1,
            wandb_enabled=True,
            wandb_cfg=cfg,
            layer_labels_available=False,
        )
        self.assertFalse(schedule.capture_activation_norms)
        self.assertFalse(schedule.capture_attention_entropy)
        self.assertTrue(schedule.should_log_attention_entropy)


if __name__ == "__main__":
    unittest.main()
