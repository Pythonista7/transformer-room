from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

from src.config import WandbMetricsConfig
from src.components.attention.basic_mh_self_attn import BasicMultiHeadSelfAttention
from src.components.attention.sdpa_mh_self_attn import SDPASelfAttn
from src.training.metrics import EpochMetricsContext, MetricSchedule, PeriodicValMetricsContext, StepMetricsContext
from src.training.metrics.plugins import get_decoder_layer_labels
from src.training.metrics.plugins.forward_hook_metrics import (
    ForwardHookMetricsPlugin,
    compute_attention_entropy_from_module_inputs,
)
from src.training.metrics.plugins.global_grad_norm import GlobalGradNormPlugin
from src.training.metrics.plugins.layer_grad_norm import (
    LayerGradNormPlugin,
    compute_layer_grad_norms,
    get_sampled_decoder_layer_indices,
)
from src.training.metrics.plugins.layernorm_grad_norm import LayerNormGradNormPlugin
from src.training.metrics.plugins.loss_metrics import LossMetricsPlugin
from src.training.metrics.plugins.parameter_optimizer_norms import ParameterOptimizerNormsPlugin
from src.training.metrics.plugins.step_timing_memory import StepTimingAndMemoryPlugin


class _FakeLayerNorm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(width))
        self.beta = torch.nn.Parameter(torch.zeros(width))


class _FakeMultiHeadAttention(torch.nn.Module):
    def __init__(self, width: int, n_heads: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = width // n_heads
        self.packed_proj = torch.nn.Linear(width, width * 3, bias=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        _ = self.packed_proj(x)
        return x


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln1 = _FakeLayerNorm(width)
        self.ln2 = _FakeLayerNorm(width)
        self.multi_head_attention = _FakeMultiHeadAttention(width)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x * self.ln1.gamma.mean() + self.ln1.beta.mean()
        x = x * self.ln2.gamma.mean() + self.ln2.beta.mean()
        _ = self.multi_head_attention(x, **kwargs)
        return x


class _FakeDecoderModel(torch.nn.Module):
    def __init__(self, width: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.dec_layers = torch.nn.ModuleList(_FakeDecoderLayer(width) for _ in range(layers))
        self.proj = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.dec_layers:
            x = layer(x, **kwargs)
        return self.proj(x)


def _make_schedule(
    *,
    should_log_step_metrics: bool = True,
    should_log_diagnostics: bool = True,
    should_log_layer_grad_norms: bool = False,
    should_log_parameter_optimizer_norms: bool | None = None,
    should_log_attention_entropy: bool = True,
    capture_activation_norms: bool = True,
    capture_attention_entropy: bool = True,
    should_log_this_step: bool = True,
    periodic_val_due: bool = True,
) -> MetricSchedule:
    if should_log_parameter_optimizer_norms is None:
        should_log_parameter_optimizer_norms = should_log_diagnostics
    return MetricSchedule(
        should_log_step_metrics=should_log_step_metrics,
        should_log_diagnostics=should_log_diagnostics,
        should_log_layer_grad_norms=should_log_layer_grad_norms,
        should_log_parameter_optimizer_norms=should_log_parameter_optimizer_norms,
        should_log_attention_entropy=should_log_attention_entropy,
        capture_activation_norms=capture_activation_norms,
        capture_attention_entropy=capture_attention_entropy,
        should_log_this_step=should_log_this_step,
        periodic_val_due=periodic_val_due,
    )


def _step_ctx(
    schedule: MetricSchedule,
    *,
    step_loss: float | None = 1.25,
    step_bits_per_byte: float | None = 0.5,
    lr_current: float | None = None,
    step_time_ms: float | None = None,
    forward_pass_time_ms: float | None = None,
    backward_pass_time_ms: float | None = None,
    optim_step_time_ms: float | None = None,
    peak_memory_gib: float | None = None,
    peak_reserved_memory_gib: float | None = None,
    include_in_perf_aggregates: bool = True,
) -> StepMetricsContext:
    return StepMetricsContext(
        schedule=schedule,
        global_step=1,
        next_global_step=2,
        epoch=0,
        batch_idx=0,
        train_loader_len=10,
        tokens_seen_train=32,
        step_loss=step_loss,
        step_bits_per_byte=step_bits_per_byte,
        step_time_ms=step_time_ms,
        forward_pass_time_ms=forward_pass_time_ms,
        backward_pass_time_ms=backward_pass_time_ms,
        optim_step_time_ms=optim_step_time_ms,
        peak_memory_gib=peak_memory_gib,
        peak_reserved_memory_gib=peak_reserved_memory_gib,
        include_in_perf_aggregates=include_in_perf_aggregates,
        lr_current=lr_current,
    )


def _periodic_val_ctx(schedule: MetricSchedule) -> PeriodicValMetricsContext:
    return PeriodicValMetricsContext(
        schedule=schedule,
        global_step=1,
        epoch=0,
        batch_idx=0,
        train_loader_len=10,
        tokens_seen_train=32,
        val_metrics={
            "val_loss": 1.75,
            "val_perplexity": 3.5,
            "val_bits_per_byte": 2.25,
        },
    )


def _epoch_ctx() -> EpochMetricsContext:
    return EpochMetricsContext(
        global_step=5,
        epoch=0,
        avg_train_loss=2.0,
        tokens_seen_train=64,
        val_metrics={
            "val_loss": 1.5,
            "val_perplexity": 2.5,
            "val_bits_per_byte": 2.0,
        },
        train_bits_per_byte_epoch=2.75,
    )


def _count_forward_hooks(model: _FakeDecoderModel) -> int:
    count = 0
    for layer in model.dec_layers:
        count += len(layer._forward_hooks)
        count += len(layer.multi_head_attention._forward_hooks)
        count += len(layer.multi_head_attention.packed_proj._forward_hooks)
    return count


class LossMetricsPluginTests(unittest.TestCase):
    def test_enabled_metrics_emitted(self) -> None:
        plugin = LossMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=True,
                enable_perplexity=True,
                enable_bits_per_byte=True,
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
        self.assertIn("train_bits_per_byte", step_metrics)
        self.assertIn("tokens_seen_train", step_metrics)

        self.assertIn("epoch", periodic_metrics)
        self.assertIn("val_loss", periodic_metrics)
        self.assertIn("val_perplexity", periodic_metrics)
        self.assertIn("val_bits_per_byte", periodic_metrics)

        self.assertIn("train_loss_epoch", epoch_metrics)
        self.assertIn("val_loss", epoch_metrics)
        self.assertIn("val_perplexity", epoch_metrics)
        self.assertIn("val_bits_per_byte", epoch_metrics)
        self.assertIn("train_perplexity_epoch", epoch_metrics)
        self.assertIn("train_bits_per_byte_epoch", epoch_metrics)
        self.assertIn("tokens_seen_train", epoch_metrics)

    def test_disabled_wandb_metrics_keep_epoch_progress_only(self) -> None:
        plugin = LossMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_train_loss_vs_tokens=False,
                enable_val_loss_vs_tokens=False,
                enable_perplexity=False,
                enable_bits_per_byte=False,
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

    def test_lr_current_is_emitted_on_logged_steps(self) -> None:
        plugin = LossMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_train_loss_vs_tokens=False,
                enable_val_loss_vs_tokens=False,
                enable_perplexity=False,
                enable_bits_per_byte=False,
            ),
        )
        step_metrics = plugin.collect_step_metrics(
            _step_ctx(_make_schedule(), lr_current=1.25e-4)
        )

        self.assertEqual(step_metrics["lr_current"], 1.25e-4)
        self.assertEqual(set(step_metrics.keys()), {"epoch", "lr_current"})


class StepTimingAndMemoryPluginTests(unittest.TestCase):
    def test_timing_metrics_emitted_when_present_in_context(self) -> None:
        plugin = StepTimingAndMemoryPlugin(
            wandb_cfg=WandbMetricsConfig(enable_step_time=True, enable_peak_memory=False),
            device=torch.device("cpu"),
        )
        ctx = _step_ctx(
            _make_schedule(
                should_log_step_metrics=True,
                should_log_this_step=True,
            ),
            step_time_ms=12.5,
            forward_pass_time_ms=6.0,
            backward_pass_time_ms=3.5,
            optim_step_time_ms=2.5,
        )

        plugin.on_train_start()
        metrics = plugin.collect_step_metrics(ctx)

        self.assertIn("step_time_ms", metrics)
        self.assertEqual(metrics["step_time_ms"], 12.5)
        self.assertEqual(metrics["forward_pass_time_ms"], 6.0)
        self.assertEqual(metrics["backward_pass_time_ms"], 3.5)
        self.assertEqual(metrics["optim_step_time_ms"], 2.5)

    def test_reserved_memory_metric_emitted_when_available(self) -> None:
        plugin = StepTimingAndMemoryPlugin(
            wandb_cfg=WandbMetricsConfig(enable_step_time=False, enable_peak_memory=True),
            device=torch.device("cuda"),
        )
        ctx = _step_ctx(
            _make_schedule(
                should_log_step_metrics=True,
                should_log_this_step=True,
            ),
            peak_memory_gib=1.25,
            peak_reserved_memory_gib=2.5,
        )

        metrics = plugin.collect_step_metrics(ctx)
        self.assertEqual(metrics["peak_memory_gib"], 1.25)
        self.assertEqual(metrics["peak_reserved_memory_gib"], 2.5)

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

        metrics = plugin.collect_step_metrics(ctx)
        self.assertEqual(metrics, {})

    def test_epoch_timing_aggregates_skip_compile_warmup_steps(self) -> None:
        plugin = StepTimingAndMemoryPlugin(
            wandb_cfg=WandbMetricsConfig(enable_step_time=True, enable_peak_memory=False),
            device=torch.device("cpu"),
        )
        plugin.on_train_start()

        warmup_ctx = _step_ctx(
            _make_schedule(),
            step_time_ms=30.0,
            forward_pass_time_ms=12.0,
            backward_pass_time_ms=10.0,
            optim_step_time_ms=8.0,
            include_in_perf_aggregates=False,
        )
        measured_ctx = _step_ctx(
            _make_schedule(),
            step_time_ms=20.0,
            forward_pass_time_ms=9.0,
            backward_pass_time_ms=7.0,
            optim_step_time_ms=4.0,
            include_in_perf_aggregates=True,
        )
        plugin.after_optimizer_step(warmup_ctx)
        plugin.after_optimizer_step(measured_ctx)

        epoch_metrics = plugin.collect_epoch_metrics(
            EpochMetricsContext(
                global_step=2,
                epoch=0,
                avg_train_loss=1.0,
                tokens_seen_train=32,
                val_metrics={"val_loss": 1.0, "val_perplexity": 2.0},
                epoch_time_s=1.5,
            )
        )
        self.assertEqual(epoch_metrics["epoch_time_s"], 1.5)
        self.assertEqual(epoch_metrics["avg_step_time_ms_epoch"], 20.0)
        self.assertEqual(epoch_metrics["avg_forward_pass_time_ms_epoch"], 9.0)
        self.assertEqual(epoch_metrics["avg_backward_pass_time_ms_epoch"], 7.0)
        self.assertEqual(epoch_metrics["avg_optim_step_time_ms_epoch"], 4.0)

    def test_peak_memory_reset_only_runs_on_step_metric_cadence(self) -> None:
        plugin = StepTimingAndMemoryPlugin(
            wandb_cfg=WandbMetricsConfig(enable_step_time=False, enable_peak_memory=True),
            device=torch.device("cuda"),
        )
        due_ctx = _step_ctx(
            _make_schedule(
                should_log_step_metrics=True,
                should_log_this_step=True,
            )
        )
        not_due_ctx = _step_ctx(
            _make_schedule(
                should_log_step_metrics=False,
                should_log_this_step=True,
            )
        )

        with mock.patch("torch.cuda.reset_peak_memory_stats") as reset_mock:
            plugin.on_step_start(not_due_ctx)
            reset_mock.assert_not_called()

            plugin.on_step_start(due_ctx)
            reset_mock.assert_called_once_with(torch.device("cuda"))


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

    def test_layer_grad_norm_selection_includes_stride_and_endpoints(self) -> None:
        model = _FakeDecoderModel(layers=5)
        selected = get_sampled_decoder_layer_indices(model, stride=4)
        self.assertEqual(selected, (0, 2, 4))

    def test_layer_grad_norm_values_match_reference_formula(self) -> None:
        model = _FakeDecoderModel(layers=4)
        input_tensor = torch.randn(2, 4, 8)
        loss = model(input_tensor).pow(2).mean()
        loss.backward()

        selected = get_sampled_decoder_layer_indices(model, stride=2)
        actual = compute_layer_grad_norms(model, selected)
        expected: dict[str, float] = {}
        for layer_idx in selected:
            layer_sq: torch.Tensor | None = None
            for param in model.dec_layers[layer_idx].parameters():
                if param.grad is None:
                    continue
                term = param.grad.detach().float().pow(2).sum()
                layer_sq = term if layer_sq is None else layer_sq + term
            if layer_sq is not None:
                expected[f"layer_grad_norm_layer_{layer_idx}"] = float(layer_sq.sqrt().item())

        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for key, expected_value in expected.items():
            self.assertAlmostEqual(actual[key], expected_value, places=6)

    def test_layer_grad_norm_plugin_respects_enable_flag_and_schedule(self) -> None:
        model = _FakeDecoderModel(layers=4)
        input_tensor = torch.randn(2, 4, 8)
        loss = model(input_tensor).pow(2).mean()
        loss.backward()
        enabled_ctx = _step_ctx(_make_schedule(should_log_layer_grad_norms=True))
        disabled_ctx = _step_ctx(_make_schedule(should_log_layer_grad_norms=False))

        enabled = LayerGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_layer_grad_norms=True,
                layer_grad_norm_stride=2,
            ),
            model=model,
        )
        enabled.on_step_start(enabled_ctx)
        enabled.after_backward(enabled_ctx)
        enabled_metrics = enabled.collect_step_metrics(enabled_ctx)
        self.assertIn("layer_grad_norm_layer_0", enabled_metrics)
        self.assertIn("layer_grad_norm_layer_2", enabled_metrics)
        self.assertIn("layer_grad_norm_layer_3", enabled_metrics)

        disabled_by_flag = LayerGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_layer_grad_norms=False,
                layer_grad_norm_stride=2,
            ),
            model=model,
        )
        disabled_by_flag.on_step_start(enabled_ctx)
        disabled_by_flag.after_backward(enabled_ctx)
        self.assertEqual(disabled_by_flag.collect_step_metrics(enabled_ctx), {})

        disabled_by_schedule = LayerGradNormPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_layer_grad_norms=True,
                layer_grad_norm_stride=2,
            ),
            model=model,
        )
        disabled_by_schedule.on_step_start(disabled_ctx)
        disabled_by_schedule.after_backward(disabled_ctx)
        self.assertEqual(disabled_by_schedule.collect_step_metrics(disabled_ctx), {})

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

    def test_attention_entropy_hooks_use_each_layer_attention_module(self) -> None:
        model = _FakeDecoderModel(layers=3)
        labels = get_decoder_layer_labels(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=False,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=labels,
        )

        packed_proj_calls: list[int] = [0 for _ in range(len(model.dec_layers))]
        handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer_idx, layer in enumerate(model.dec_layers):
            handle = layer.multi_head_attention.packed_proj.register_forward_hook(
                lambda _module, _inputs, _output, layer_idx=layer_idx: packed_proj_calls.__setitem__(
                    layer_idx,
                    packed_proj_calls[layer_idx] + 1,
                )
            )
            handles.append(handle)

        plugin.on_train_start()
        try:
            ctx = _step_ctx(
                _make_schedule(
                    capture_activation_norms=False,
                    capture_attention_entropy=True,
                )
            )
            plugin.on_step_start(ctx)
            _ = model(torch.randn(2, 4, 8))
            _ = plugin.collect_step_metrics(ctx)
        finally:
            plugin.on_train_end()
            for handle in handles:
                handle.remove()

        self.assertEqual(packed_proj_calls, [1, 1, 1])

    def test_attention_entropy_uses_first_microbatch_in_step(self) -> None:
        torch.manual_seed(42)
        model = _FakeDecoderModel(layers=3)
        labels = get_decoder_layer_labels(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=False,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=labels,
        )

        ctx = _step_ctx(
            _make_schedule(
                capture_activation_norms=False,
                capture_attention_entropy=True,
            )
        )
        first_input = torch.randn(2, 4, 8)
        second_input = torch.zeros_like(first_input)
        expected_first = compute_attention_entropy_from_module_inputs(
            model.dec_layers[0].multi_head_attention,
            first_input,
            attention_head_cap=1,
            attention_token_cap=4,
            is_causal=True,
        )
        expected_second = compute_attention_entropy_from_module_inputs(
            model.dec_layers[0].multi_head_attention,
            second_input,
            attention_head_cap=1,
            attention_token_cap=4,
            is_causal=True,
        )

        self.assertIsNotNone(expected_first)
        self.assertIsNotNone(expected_second)
        if expected_first is None or expected_second is None:
            self.fail("Expected entropy helpers to return tensors.")
        self.assertNotAlmostEqual(
            float(expected_first.item()),
            float(expected_second.item()),
            places=6,
        )

        plugin.on_train_start()
        try:
            plugin.on_step_start(ctx)
            _ = model(first_input)
            _ = model(second_input)
            metrics = plugin.collect_step_metrics(ctx)
        finally:
            plugin.on_train_end()

        self.assertIn("attention_entropy_first", metrics)
        self.assertAlmostEqual(
            metrics["attention_entropy_first"],
            float(expected_first.item()),
            places=6,
        )

    def test_attention_entropy_hook_preserves_mask_and_causal_kwargs(self) -> None:
        torch.manual_seed(7)
        model = _FakeDecoderModel(layers=3)
        labels = get_decoder_layer_labels(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=False,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=labels,
        )

        ctx = _step_ctx(
            _make_schedule(
                capture_activation_norms=False,
                capture_attention_entropy=True,
            )
        )
        inputs = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        )
        key_padding_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        )
        expected_first = compute_attention_entropy_from_module_inputs(
            model.dec_layers[0].multi_head_attention,
            inputs,
            attention_head_cap=1,
            attention_token_cap=4,
            mask=mask,
            key_padding_mask=key_padding_mask,
            is_causal=False,
        )

        self.assertIsNotNone(expected_first)
        if expected_first is None:
            self.fail("Expected entropy helper to return a tensor.")

        plugin.on_train_start()
        try:
            plugin.on_step_start(ctx)
            _ = model(
                inputs,
                mask=mask,
                key_padding_mask=key_padding_mask,
                is_causal=False,
            )
            metrics = plugin.collect_step_metrics(ctx)
        finally:
            plugin.on_train_end()

        self.assertIn("attention_entropy_first", metrics)
        self.assertAlmostEqual(
            metrics["attention_entropy_first"],
            float(expected_first.item()),
            places=6,
        )

    def test_attention_entropy_capture_disabled_after_step_collection(self) -> None:
        model = _FakeDecoderModel(layers=3)
        labels = get_decoder_layer_labels(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=False,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=labels,
        )
        ctx = _step_ctx(
            _make_schedule(
                capture_activation_norms=False,
                capture_attention_entropy=True,
            )
        )

        plugin.on_train_start()
        try:
            plugin.on_step_start(ctx)
            _ = model(torch.randn(2, 4, 8))
            first_metrics = plugin.collect_step_metrics(ctx)
            self.assertIn("attention_entropy_first", first_metrics)

            _ = model(torch.randn(2, 4, 8))
            second_metrics = plugin.collect_step_metrics(ctx)
        finally:
            plugin.on_train_end()

        self.assertEqual(second_metrics, {})

    def test_forward_hook_metrics_do_not_change_step_update(self) -> None:
        torch.manual_seed(0)
        base_model = _FakeDecoderModel(layers=3)
        model_with_metrics = copy.deepcopy(base_model)
        model_without_metrics = copy.deepcopy(base_model)

        optimizer_with_metrics = torch.optim.Adam(
            model_with_metrics.parameters(),
            lr=1e-3,
        )
        optimizer_without_metrics = torch.optim.Adam(
            model_without_metrics.parameters(),
            lr=1e-3,
        )

        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=True,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model_with_metrics,
            layer_labels=get_decoder_layer_labels(model_with_metrics),
        )
        ctx = _step_ctx(
            _make_schedule(
                capture_activation_norms=True,
                capture_attention_entropy=True,
            )
        )
        inputs = torch.randn(2, 4, 8)

        plugin.on_train_start()
        try:
            optimizer_with_metrics.zero_grad()
            plugin.on_step_start(ctx)
            output_with_metrics = model_with_metrics(inputs)
            _ = plugin.collect_step_metrics(ctx)
            loss_with_metrics = output_with_metrics.pow(2).mean()
            loss_with_metrics.backward()
            optimizer_with_metrics.step()
        finally:
            plugin.on_train_end()

        optimizer_without_metrics.zero_grad()
        output_without_metrics = model_without_metrics(inputs)
        loss_without_metrics = output_without_metrics.pow(2).mean()
        loss_without_metrics.backward()
        optimizer_without_metrics.step()

        self.assertAlmostEqual(
            float(loss_with_metrics.item()),
            float(loss_without_metrics.item()),
            places=8,
        )
        for with_metrics, without_metrics in zip(
            model_with_metrics.parameters(),
            model_without_metrics.parameters(),
        ):
            self.assertTrue(
                torch.equal(with_metrics, without_metrics),
                "Metric hooks changed optimization results.",
            )

    def test_unified_entropy_matches_basic_attention_probs(self) -> None:
        attn = BasicMultiHeadSelfAttention(E_q=8, E_out=8, n_heads=2, E_bias=True)
        inputs = torch.randn(2, 4, 8)
        key_padding_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        )
        captured: dict[str, torch.Tensor] = {}

        def capture_probs(_module, _inputs, output) -> None:
            captured["probs"] = output.detach()

        handle = attn.softmax.register_forward_hook(capture_probs)
        try:
            _ = attn(inputs, key_padding_mask=key_padding_mask, is_causal=True)
        finally:
            handle.remove()

        probs = captured["probs"][:, :1, :4, :4].float().clamp_min(1e-12)
        expected_entropy = -(probs * probs.log()).sum(dim=-1).mean()
        computed_entropy = compute_attention_entropy_from_module_inputs(
            attn,
            inputs,
            attention_head_cap=1,
            attention_token_cap=4,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )

        self.assertIsNotNone(computed_entropy)
        self.assertAlmostEqual(
            float(computed_entropy.item()),
            float(expected_entropy.item()),
            places=6,
        )
        self.assertEqual(computed_entropy.device, inputs.device)

    def test_unified_entropy_matches_between_basic_and_sdpa_modules(self) -> None:
        basic_attn = BasicMultiHeadSelfAttention(E_q=8, E_out=8, n_heads=2, E_bias=True)
        sdpa_attn = SDPASelfAttn(d_model=8, n_heads=2, dropout=0.0)
        inputs = torch.randn(2, 4, 8)
        key_padding_mask = torch.tensor(
            [[True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        )

        with torch.no_grad():
            sdpa_attn.packed_proj.W.copy_(basic_attn.packed_proj.W)
            sdpa_attn.packed_proj.b.copy_(basic_attn.packed_proj.b)

        basic_entropy = compute_attention_entropy_from_module_inputs(
            basic_attn,
            inputs,
            attention_head_cap=2,
            attention_token_cap=4,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        sdpa_entropy = compute_attention_entropy_from_module_inputs(
            sdpa_attn,
            inputs,
            attention_head_cap=2,
            attention_token_cap=4,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )

        self.assertIsNotNone(basic_entropy)
        self.assertIsNotNone(sdpa_entropy)
        self.assertAlmostEqual(
            float(basic_entropy.item()),
            float(sdpa_entropy.item()),
            places=6,
        )


class ParameterOptimizerNormsPluginTests(unittest.TestCase):
    def _finalize_norm(self, total_sq: torch.Tensor | None) -> float | None:
        if total_sq is None:
            return None
        return float(total_sq.sqrt().item())

    def _expected_metrics_from_reference(
        self,
        *,
        model: _FakeDecoderModel,
        optimizer: torch.optim.Optimizer,
        layer_labels: dict[int, list[str]],
        pre_step_global_param_norm: float | None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        global_sq: torch.Tensor | None = None
        for param in model.parameters():
            if not param.requires_grad:
                continue
            term = param.detach().float().pow(2).sum()
            global_sq = term if global_sq is None else global_sq + term
        global_param_norm = self._finalize_norm(global_sq)
        if global_param_norm is not None:
            metrics["global_param_norm"] = global_param_norm

        dec_layers = model.dec_layers
        for layer_idx, labels in layer_labels.items():
            layer_sq: torch.Tensor | None = None
            for param in dec_layers[layer_idx].parameters():
                term = param.detach().float().pow(2).sum()
                layer_sq = term if layer_sq is None else layer_sq + term
            layer_norm = self._finalize_norm(layer_sq)
            if layer_norm is None:
                continue
            for label in labels:
                metrics[f"layer_param_norm_{label}"] = layer_norm

        update_norm = self._expected_update_norm_from_optimizer_state(optimizer)
        if update_norm is not None:
            metrics["param_update_norm"] = update_norm
            if (
                pre_step_global_param_norm is not None
                and pre_step_global_param_norm > 0.0
            ):
                metrics["update_to_weight_ratio"] = float(update_norm / pre_step_global_param_norm)

        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            beta1, beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999))
            m_sq: torch.Tensor | None = None
            v_sq: torch.Tensor | None = None
            snr_sq: torch.Tensor | None = None

            for state in optimizer.state.values():
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step_val = state.get("step")
                if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
                    continue

                m_hat = exp_avg.detach().float()
                v_hat = exp_avg_sq.detach().float()
                if step_val is not None:
                    t = float(step_val.item() if torch.is_tensor(step_val) else step_val)
                    if t > 0.0:
                        m_hat = m_hat / (1 - beta1**t)
                        v_hat = v_hat / (1 - beta2**t)

                m_term = m_hat.pow(2).sum()
                v_term = v_hat.pow(2).sum()
                snr_term = (m_hat / (v_hat.sqrt() + 1e-8)).pow(2).sum()
                m_sq = m_term if m_sq is None else m_sq + m_term
                v_sq = v_term if v_sq is None else v_sq + v_term
                snr_sq = snr_term if snr_sq is None else snr_sq + snr_term

            m_norm = self._finalize_norm(m_sq)
            v_norm = self._finalize_norm(v_sq)
            snr_norm = self._finalize_norm(snr_sq)
            if m_norm is not None and v_norm is not None:
                metrics["adam_m_norm"] = m_norm
                metrics["adam_v_norm"] = v_norm
                metrics["adam_elemwise_snr_norm"] = float(snr_norm) if snr_norm is not None else 0.0

            for layer_idx, labels in layer_labels.items():
                layer_v_sq: torch.Tensor | None = None
                for param in dec_layers[layer_idx].parameters():
                    state = optimizer.state.get(param)
                    if not state:
                        continue
                    exp_avg_sq = state.get("exp_avg_sq")
                    step_val = state.get("step")
                    if not torch.is_tensor(exp_avg_sq):
                        continue

                    v_hat = exp_avg_sq.detach().float()
                    if step_val is not None:
                        t = float(step_val.item() if torch.is_tensor(step_val) else step_val)
                        if t > 0.0:
                            v_hat = v_hat / (1 - beta2**t)
                    term = v_hat.pow(2).sum()
                    layer_v_sq = term if layer_v_sq is None else layer_v_sq + term

                layer_v_norm = self._finalize_norm(layer_v_sq)
                if layer_v_norm is None:
                    continue
                for label in labels:
                    metrics[f"layer_estimated_variance_norm_{label}"] = layer_v_norm

        return metrics

    def _expected_update_norm_from_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> float | None:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            return None

        update_sq: torch.Tensor | None = None
        for group in optimizer.param_groups:
            lr = group.get("lr")
            betas = group.get("betas")
            eps = group.get("eps", 1e-8)
            if not isinstance(lr, (int, float)):
                return None
            if not isinstance(eps, (int, float)):
                return None
            if not isinstance(betas, tuple) or len(betas) != 2:
                return None
            beta1, beta2 = betas
            if not isinstance(beta1, (int, float)) or not isinstance(beta2, (int, float)):
                return None

            for param in group["params"]:
                state = optimizer.state.get(param)
                if not state:
                    continue
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                step_val = state.get("step")
                if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
                    continue

                m_hat = exp_avg.detach().float()
                v_hat = exp_avg_sq.detach().float()
                if step_val is not None:
                    t = float(step_val.item() if torch.is_tensor(step_val) else step_val)
                    if t > 0.0:
                        m_hat = m_hat / (1 - float(beta1) ** t)
                        v_hat = v_hat / (1 - float(beta2) ** t)
                update = float(lr) * (m_hat / (v_hat.sqrt() + float(eps)))
                term = update.pow(2).sum()
                update_sq = term if update_sq is None else update_sq + term

        return self._finalize_norm(update_sq)

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

    def test_exact_metric_values_match_reference_formula(self) -> None:
        torch.manual_seed(7)
        model = _FakeDecoderModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        layer_labels = get_decoder_layer_labels(model)
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
            layer_labels=layer_labels,
        )
        schedule = _make_schedule(should_log_diagnostics=True)
        ctx = _step_ctx(schedule)

        plugin.on_step_start(ctx)
        pre_step_global_norm = plugin._pre_step_global_param_norm

        optimizer.zero_grad()
        torch.manual_seed(11)
        loss = model(torch.randn(2, 4, 8)).pow(2).mean()
        loss.backward()
        optimizer.step()

        plugin.after_optimizer_step(ctx)
        actual = plugin.collect_step_metrics(ctx)
        expected = self._expected_metrics_from_reference(
            model=model,
            optimizer=optimizer,
            layer_labels=layer_labels,
            pre_step_global_param_norm=pre_step_global_norm,
        )

        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for key, expected_value in expected.items():
            self.assertAlmostEqual(actual[key], expected_value, places=5, msg=key)

    def test_clears_pre_step_norm_buffers_after_step_and_train_end(self) -> None:
        model = _FakeDecoderModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        plugin = ParameterOptimizerNormsPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_param_update_norm=True,
                enable_update_to_weight_ratio=True,
            ),
            model=model,
            optimizer=optimizer,
            layer_labels=get_decoder_layer_labels(model),
        )
        schedule = _make_schedule(should_log_diagnostics=True)
        ctx = _step_ctx(schedule)

        plugin.on_step_start(ctx)
        self.assertIsNotNone(plugin._pre_step_global_param_norm)

        optimizer.zero_grad()
        loss = model(torch.randn(2, 4, 8)).pow(2).mean()
        loss.backward()
        optimizer.step()
        plugin.after_optimizer_step(ctx)

        self.assertIsNone(plugin._pre_step_global_param_norm)

        plugin.on_step_start(ctx)
        self.assertIsNotNone(plugin._pre_step_global_param_norm)
        plugin.on_train_end()
        self.assertIsNone(plugin._pre_step_global_param_norm)
        self.assertEqual(plugin._metrics, {})

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

    def test_sgd_skips_update_metrics_and_warns_once(self) -> None:
        model = _FakeDecoderModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        plugin = ParameterOptimizerNormsPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_param_update_norm=True,
                enable_update_to_weight_ratio=True,
            ),
            model=model,
            optimizer=optimizer,
            layer_labels=get_decoder_layer_labels(model),
        )
        schedule = _make_schedule(should_log_diagnostics=True)

        with self.assertLogs(
            "src.training.metrics.plugins.parameter_optimizer_norms",
            level="WARNING",
        ) as warning_logs:
            metrics_first = self._run_single_optimization_step(
                model=model,
                optimizer=optimizer,
                plugin=plugin,
                schedule=schedule,
            )
            metrics_second = self._run_single_optimization_step(
                model=model,
                optimizer=optimizer,
                plugin=plugin,
                schedule=schedule,
            )

        self.assertNotIn("param_update_norm", metrics_first)
        self.assertNotIn("update_to_weight_ratio", metrics_first)
        self.assertNotIn("param_update_norm", metrics_second)
        self.assertNotIn("update_to_weight_ratio", metrics_second)
        self.assertEqual(len(warning_logs.output), 1)
        self.assertIn("Skipping param_update_norm/update_to_weight_ratio", warning_logs.output[0])

    def test_zero_trainable_params_omits_update_and_ratio_without_crashing(self) -> None:
        model = _FakeDecoderModel()
        for param in model.parameters():
            param.requires_grad_(False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        plugin = ParameterOptimizerNormsPlugin(
            wandb_cfg=WandbMetricsConfig(
                enable_param_update_norm=True,
                enable_update_to_weight_ratio=True,
            ),
            model=model,
            optimizer=optimizer,
            layer_labels=get_decoder_layer_labels(model),
        )
        ctx = _step_ctx(_make_schedule(should_log_diagnostics=True))

        plugin.on_step_start(ctx)
        plugin.after_optimizer_step(ctx)
        metrics = plugin.collect_step_metrics(ctx)

        self.assertNotIn("param_update_norm", metrics)
        self.assertNotIn("update_to_weight_ratio", metrics)

    def test_param_update_norm_uses_per_group_hparams(self) -> None:
        torch.manual_seed(17)
        model = _FakeDecoderModel()
        params = list(model.parameters())
        split_at = len(params) // 2
        optimizer = torch.optim.Adam(
            [
                {
                    "params": params[:split_at],
                    "lr": 1e-3,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                },
                {
                    "params": params[split_at:],
                    "lr": 3e-4,
                    "betas": (0.8, 0.99),
                    "eps": 1e-6,
                },
            ]
        )
        plugin = ParameterOptimizerNormsPlugin(
            wandb_cfg=WandbMetricsConfig(enable_param_update_norm=True),
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
        expected_update_norm = self._expected_update_norm_from_optimizer_state(optimizer)

        self.assertIn("param_update_norm", metrics)
        self.assertIsNotNone(expected_update_norm)
        self.assertAlmostEqual(metrics["param_update_norm"], expected_update_norm, places=6)

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
