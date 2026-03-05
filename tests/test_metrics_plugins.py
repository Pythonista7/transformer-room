from __future__ import annotations

import time
import unittest

import torch

from baseline.config import WandbMetricsConfig
from baseline.training.metrics import EpochMetricsContext, MetricSchedule, PeriodicValMetricsContext, StepMetricsContext
from baseline.training.metrics.plugins import get_decoder_layer_labels
from baseline.training.metrics.plugins.forward_hook_metrics import ForwardHookMetricsPlugin
from baseline.training.metrics.plugins.global_grad_norm import GlobalGradNormPlugin
from baseline.training.metrics.plugins.layernorm_grad_norm import LayerNormGradNormPlugin
from baseline.training.metrics.plugins.loss_perplexity import LossPerplexityPlugin
from baseline.training.metrics.plugins.parameter_optimizer_norms import ParameterOptimizerNormsPlugin
from baseline.training.metrics.plugins.step_timing_memory import StepTimingAndMemoryPlugin


class _FakeLayerNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(width))
        self.beta = torch.nn.Parameter(torch.zeros(width))


class _FakeMultiHeadAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        scores = torch.randn(batch_size, 2, 4, 4, device=x.device, dtype=x.dtype)
        return self.softmax(scores)


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln1 = _FakeLayerNorm(width)
        self.ln2 = _FakeLayerNorm(width)
        self.multi_head_attention = _FakeMultiHeadAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ln1.gamma.mean() + self.ln1.beta.mean()
        x = x * self.ln2.gamma.mean() + self.ln2.beta.mean()
        _ = self.multi_head_attention(x)
        return x


class _FakeDecoderModel(torch.nn.Module):
    def __init__(self, width: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.dec_layers = torch.nn.ModuleList(_FakeDecoderLayer(width) for _ in range(layers))
        self.proj = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.dec_layers:
            x = layer(x)
        return self.proj(x)


def _make_schedule(
    *,
    should_log_step_metrics: bool = True,
    should_log_diagnostics: bool = True,
    should_log_attention_entropy: bool = True,
    capture_activation_norms: bool = True,
    capture_attention_entropy: bool = True,
    should_log_this_step: bool = True,
    periodic_val_due: bool = True,
) -> MetricSchedule:
    return MetricSchedule(
        should_log_step_metrics=should_log_step_metrics,
        should_log_diagnostics=should_log_diagnostics,
        should_log_attention_entropy=should_log_attention_entropy,
        capture_activation_norms=capture_activation_norms,
        capture_attention_entropy=capture_attention_entropy,
        should_log_this_step=should_log_this_step,
        periodic_val_due=periodic_val_due,
    )


def _step_ctx(schedule: MetricSchedule, *, step_loss: float | None = 1.25) -> StepMetricsContext:
    return StepMetricsContext(
        schedule=schedule,
        global_step=1,
        next_global_step=2,
        epoch=0,
        batch_idx=0,
        train_loader_len=10,
        tokens_seen_train=32,
        step_loss=step_loss,
    )


def _periodic_val_ctx(schedule: MetricSchedule) -> PeriodicValMetricsContext:
    return PeriodicValMetricsContext(
        schedule=schedule,
        global_step=1,
        epoch=0,
        batch_idx=0,
        train_loader_len=10,
        tokens_seen_train=32,
        val_metrics={"val_loss": 1.75, "val_perplexity": 3.5},
    )


def _epoch_ctx() -> EpochMetricsContext:
    return EpochMetricsContext(
        global_step=5,
        epoch=0,
        avg_train_loss=2.0,
        tokens_seen_train=64,
        val_metrics={"val_loss": 1.5, "val_perplexity": 2.5},
    )


def _count_forward_hooks(model: _FakeDecoderModel) -> int:
    count = 0
    for layer in model.dec_layers:
        count += len(layer._forward_hooks)
        count += len(layer.multi_head_attention.softmax._forward_hooks)
    return count


class LossPerplexityPluginTests(unittest.TestCase):
    def test_enabled_metrics_emitted(self) -> None:
        plugin = LossPerplexityPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=True,
                enable_perplexity=True,
            ),
        )

        step_metrics = plugin.collect_step_metrics(_step_ctx(_make_schedule()))
        periodic_metrics = plugin.collect_periodic_val_metrics(
            _periodic_val_ctx(_make_schedule())
        )
        epoch_metrics = plugin.collect_epoch_metrics(_epoch_ctx())

        self.assertIn("epoch", step_metrics)
        self.assertIn("train_loss_step", step_metrics)
        self.assertIn("train_perplexity", step_metrics)
        self.assertIn("tokens_seen_train", step_metrics)

        self.assertIn("epoch", periodic_metrics)
        self.assertIn("val_loss", periodic_metrics)
        self.assertIn("val_perplexity", periodic_metrics)

        self.assertIn("train_loss_epoch", epoch_metrics)
        self.assertIn("val_loss", epoch_metrics)
        self.assertIn("val_perplexity", epoch_metrics)
        self.assertIn("train_perplexity_epoch", epoch_metrics)
        self.assertIn("tokens_seen_train", epoch_metrics)

    def test_disabled_wandb_metrics_keep_epoch_progress_only(self) -> None:
        plugin = LossPerplexityPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_train_loss_vs_tokens=False,
                enable_val_loss_vs_tokens=False,
                enable_perplexity=False,
            ),
        )
        step_metrics = plugin.collect_step_metrics(_step_ctx(_make_schedule()))
        periodic_metrics = plugin.collect_periodic_val_metrics(
            _periodic_val_ctx(_make_schedule())
        )
        epoch_metrics = plugin.collect_epoch_metrics(_epoch_ctx())

        self.assertEqual(set(step_metrics.keys()), {"epoch"})
        self.assertEqual(set(periodic_metrics.keys()), {"epoch"})
        self.assertEqual(set(epoch_metrics.keys()), {"epoch", "train_loss_epoch"})


class StepTimingAndMemoryPluginTests(unittest.TestCase):
    def test_step_time_emitted_on_cpu(self) -> None:
        plugin = StepTimingAndMemoryPlugin(
            wandb_cfg=WandbMetricsConfig(enable_step_time=True, enable_peak_memory=False),
            device=torch.device("cpu"),
        )
        ctx = _step_ctx(
            _make_schedule(
                should_log_step_metrics=True,
                should_log_this_step=True,
            )
        )

        plugin.on_step_start(ctx)
        time.sleep(0.001)
        metrics = plugin.collect_step_metrics(ctx)

        self.assertIn("step_time_ms", metrics)
        self.assertGreaterEqual(metrics["step_time_ms"], 0.0)

    def test_no_metrics_when_step_cadence_disabled(self) -> None:
        plugin = StepTimingAndMemoryPlugin(
            wandb_cfg=WandbMetricsConfig(enable_step_time=True, enable_peak_memory=False),
            device=torch.device("cpu"),
        )
        ctx = _step_ctx(
            _make_schedule(
                should_log_step_metrics=False,
                should_log_this_step=True,
            )
        )

        plugin.on_step_start(ctx)
        metrics = plugin.collect_step_metrics(ctx)
        self.assertEqual(metrics, {})


class GradNormPluginTests(unittest.TestCase):
    def test_global_grad_norm_enabled_and_disabled(self) -> None:
        model = _FakeDecoderModel()
        input_tensor = torch.randn(2, 4, 8)
        loss = model(input_tensor).pow(2).mean()
        loss.backward()

        schedule = _make_schedule(should_log_diagnostics=True)
        ctx = _step_ctx(schedule)

        enabled = GlobalGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(enable_global_grad_norm=True),
            model=model,
        )
        enabled.on_step_start(ctx)
        enabled.after_backward(ctx)
        enabled_metrics = enabled.collect_step_metrics(ctx)
        self.assertIn("global_grad_norm", enabled_metrics)

        disabled = GlobalGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(enable_global_grad_norm=False),
            model=model,
        )
        disabled.on_step_start(ctx)
        disabled.after_backward(ctx)
        disabled_metrics = disabled.collect_step_metrics(ctx)
        self.assertEqual(disabled_metrics, {})

    def test_layernorm_grad_norm_enabled_and_disabled(self) -> None:
        model = _FakeDecoderModel()
        labels = get_decoder_layer_labels(model)
        input_tensor = torch.randn(2, 4, 8)
        loss = model(input_tensor).pow(2).mean()
        loss.backward()

        schedule = _make_schedule(should_log_diagnostics=True)
        ctx = _step_ctx(schedule)

        enabled = LayerNormGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(enable_ln_grad_norms=True),
            model=model,
            layer_labels=labels,
        )
        enabled.on_step_start(ctx)
        enabled.after_backward(ctx)
        enabled_metrics = enabled.collect_step_metrics(ctx)
        self.assertIn("ln_weight_grad_norm_first", enabled_metrics)
        self.assertIn("ln_bias_grad_norm_first", enabled_metrics)

        disabled = LayerNormGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(enable_ln_grad_norms=False),
            model=model,
            layer_labels=labels,
        )
        disabled.on_step_start(ctx)
        disabled.after_backward(ctx)
        disabled_metrics = disabled.collect_step_metrics(ctx)
        self.assertEqual(disabled_metrics, {})


class ForwardHookMetricsPluginTests(unittest.TestCase):
    def test_hook_metrics_emitted_when_capture_enabled(self) -> None:
        model = _FakeDecoderModel()
        labels = get_decoder_layer_labels(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=True,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=labels,
        )

        self.assertEqual(_count_forward_hooks(model), 0)
        plugin.on_train_start()
        self.assertGreater(_count_forward_hooks(model), 0)

        ctx = _step_ctx(
            _make_schedule(
                capture_activation_norms=True,
                capture_attention_entropy=True,
            )
        )
        plugin.on_step_start(ctx)
        _ = model(torch.randn(2, 4, 8))

        metrics = plugin.collect_step_metrics(ctx)
        self.assertIn("activation_norm_first", metrics)
        self.assertIn("attention_entropy_first", metrics)

        plugin.on_train_end()
        self.assertEqual(_count_forward_hooks(model), 0)

    def test_hook_handles_removed_on_exception_cleanup(self) -> None:
        model = _FakeDecoderModel()
        labels = get_decoder_layer_labels(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=True,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=labels,
        )

        plugin.on_train_start()
        self.assertGreater(_count_forward_hooks(model), 0)

        with self.assertRaises(RuntimeError):
            try:
                raise RuntimeError("forced")
            finally:
                plugin.on_train_end()

        self.assertEqual(_count_forward_hooks(model), 0)


class ParameterOptimizerNormsPluginTests(unittest.TestCase):
    def _run_single_optimization_step(
        self,
        *,
        model: _FakeDecoderModel,
        optimizer: torch.optim.Optimizer,
        plugin: ParameterOptimizerNormsPlugin,
        schedule: MetricSchedule,
    ) -> dict[str, float]:
        ctx = _step_ctx(schedule)
        plugin.on_step_start(ctx)

        optimizer.zero_grad()
        loss = model(torch.randn(2, 4, 8)).pow(2).mean()
        loss.backward()
        optimizer.step()

        plugin.after_optimizer_step(ctx)
        return plugin.collect_step_metrics(ctx)

    def test_emits_param_update_and_layer_norms_when_enabled(self) -> None:
        model = _FakeDecoderModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        plugin = ParameterOptimizerNormsPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_global_param_norm=True,
                enable_layer_param_norms=True,
                enable_param_update_norm=True,
                enable_update_to_weight_ratio=True,
                enable_optimizer_state_norms=True,
            ),
            model=model,
            optimizer=optimizer,
            layer_labels=get_decoder_layer_labels(model),
        )

        metrics = self._run_single_optimization_step(
            model=model,
            optimizer=optimizer,
            plugin=plugin,
            schedule=_make_schedule(should_log_diagnostics=True),
        )

        self.assertIn("global_param_norm", metrics)
        self.assertIn("layer_param_norm_first", metrics)
        self.assertIn("layer_param_norm_middle", metrics)
        self.assertIn("layer_param_norm_last", metrics)
        self.assertIn("param_update_norm", metrics)
        self.assertIn("update_to_weight_ratio", metrics)
        self.assertIn("adam_m_norm", metrics)
        self.assertIn("adam_v_norm", metrics)
        self.assertIn("adam_elemwise_snr_norm", metrics)
        self.assertIn("layer_estimated_variance_norm_first", metrics)
        self.assertIn("layer_estimated_variance_norm_middle", metrics)
        self.assertIn("layer_estimated_variance_norm_last", metrics)
        self.assertGreater(metrics["param_update_norm"], 0.0)
        self.assertGreater(metrics["update_to_weight_ratio"], 0.0)

    def test_adam_and_adamw_emit_state_norms_but_sgd_does_not(self) -> None:
        for optimizer_name, optimizer_factory, emits_state_metrics in (
            ("adam", lambda params: torch.optim.Adam(params, lr=1e-3), True),
            ("adamw", lambda params: torch.optim.AdamW(params, lr=1e-3), True),
            ("sgd", lambda params: torch.optim.SGD(params, lr=1e-3), False),
        ):
            with self.subTest(optimizer=optimizer_name):
                model = _FakeDecoderModel()
                optimizer = optimizer_factory(model.parameters())
                plugin = ParameterOptimizerNormsPlugin(
                    wandb_cfg=WandbMetricsConfig(enable_optimizer_state_norms=True),
                    model=model,
                    optimizer=optimizer,
                    layer_labels=get_decoder_layer_labels(model),
                )
                metrics = self._run_single_optimization_step(
                    model=model,
                    optimizer=optimizer,
                    plugin=plugin,
                    schedule=_make_schedule(should_log_diagnostics=True),
                )
                if emits_state_metrics:
                    self.assertIn("adam_m_norm", metrics)
                    self.assertIn("adam_v_norm", metrics)
                    self.assertIn("adam_elemwise_snr_norm", metrics)
                    self.assertIn("layer_estimated_variance_norm_first", metrics)
                else:
                    self.assertNotIn("adam_m_norm", metrics)
                    self.assertNotIn("adam_v_norm", metrics)
                    self.assertNotIn("adam_elemwise_snr_norm", metrics)
                    self.assertNotIn("layer_estimated_variance_norm_first", metrics)

    def test_emits_nothing_when_diagnostics_cadence_is_off(self) -> None:
        model = _FakeDecoderModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        plugin = ParameterOptimizerNormsPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_global_param_norm=True,
                enable_layer_param_norms=True,
                enable_param_update_norm=True,
                enable_update_to_weight_ratio=True,
                enable_optimizer_state_norms=True,
            ),
            model=model,
            optimizer=optimizer,
            layer_labels=get_decoder_layer_labels(model),
        )

        metrics = self._run_single_optimization_step(
            model=model,
            optimizer=optimizer,
            plugin=plugin,
            schedule=_make_schedule(should_log_diagnostics=False),
        )
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
