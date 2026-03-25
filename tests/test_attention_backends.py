from __future__ import annotations

import importlib
import unittest

import torch

from src.components.attention import BasicMultiHeadSelfAttention, SDPASelfAttn
from src.components.blocks import SelfAttnDecoderBlock
from src.components.models.baseline_model import BaselineModel
from src.components.models.baseline_with_AC_model import ACEveryN_DecoderModel
from src.components.models.baseline_with_SAC_model import SelectiveAC_DecoderModel
from src.config import WandbMetricsConfig
from src.training.metrics import MetricSchedule, StepMetricsContext
from src.training.metrics.plugins import get_decoder_layer_labels
from src.training.metrics.plugins.forward_hook_metrics import ForwardHookMetricsPlugin


class _RecorderAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_q: torch.Tensor | None = None

    def forward(
        self,
        q: torch.Tensor,
        *,
        is_causal: bool,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del is_causal
        del key_padding_mask
        self.last_q = q.detach().clone()
        return torch.zeros_like(q)


def _metric_schedule(*, capture_attention_entropy: bool) -> MetricSchedule:
    return MetricSchedule(
        should_log_step_metrics=False,
        should_log_diagnostics=False,
        should_log_layer_grad_norms=False,
        should_log_parameter_optimizer_norms=False,
        should_log_attention_entropy=capture_attention_entropy,
        capture_activation_norms=False,
        capture_attention_entropy=capture_attention_entropy,
        should_log_this_step=capture_attention_entropy,
        periodic_val_due=False,
    )


def _step_ctx(schedule: MetricSchedule) -> StepMetricsContext:
    return StepMetricsContext(
        schedule=schedule,
        global_step=0,
        next_global_step=1,
        epoch=0,
        batch_idx=0,
        train_loader_len=1,
        tokens_seen_train=0,
        step_loss=1.0,
    )


class AttentionBackendSelectionTests(unittest.TestCase):
    def test_blocks_module_only_exports_shared_decoder_block(self) -> None:
        blocks_module = importlib.import_module("src.components.blocks")

        self.assertIs(blocks_module.SelfAttnDecoderBlock, SelfAttnDecoderBlock)
        self.assertEqual(blocks_module.__all__, ["SelfAttnDecoderBlock"])
        self.assertFalse(hasattr(blocks_module, "BasicSelfAttnDecoder"))
        self.assertFalse(hasattr(blocks_module, "SDPASelfAttnDecoder"))

    def test_baseline_model_uses_sdpa_attention_when_requested(self) -> None:
        model = BaselineModel(
            vocab_size=32,
            layers=1,
            d_model=8,
            n_heads=2,
            pad_id=31,
            attention_impl="sdpa",
        )
        self.assertIsInstance(model.dec_layers[0].multi_head_attention, SDPASelfAttn)

    def test_ac_model_uses_sdpa_attention_when_requested(self) -> None:
        model = ACEveryN_DecoderModel(
            vocab_size=32,
            layers=1,
            d_model=8,
            n_heads=2,
            pad_id=31,
            attention_impl="sdpa",
            checkpoint_every_n_layers=1,
        )
        self.assertIsInstance(model.dec_layers[0].multi_head_attention, SDPASelfAttn)

    def test_sac_model_uses_sdpa_attention_when_requested(self) -> None:
        model = SelectiveAC_DecoderModel(
            vocab_size=32,
            layers=1,
            d_model=8,
            n_heads=2,
            pad_id=31,
            attention_impl="sdpa",
        )
        self.assertIsInstance(model.dec_layers[0].multi_head_attention, SDPASelfAttn)

    def test_basic_attention_remains_default(self) -> None:
        model = BaselineModel(
            vocab_size=32,
            layers=1,
            d_model=8,
            n_heads=2,
            pad_id=31,
        )
        self.assertIsInstance(
            model.dec_layers[0].multi_head_attention,
            BasicMultiHeadSelfAttention,
        )

    def test_decoder_block_delegates_dropout_to_basic_attention(self) -> None:
        block = SelfAttnDecoderBlock(
            d_model=8,
            n_heads=2,
            dropout=0.25,
            attention_impl="basic",
            norm_placement="post"
        )

        self.assertFalse(hasattr(block, "attn_dropout"))
        self.assertEqual(block.multi_head_attention.attn_dropout.p, 0.25)

    def test_decoder_block_delegates_dropout_to_sdpa_attention(self) -> None:
        block = SelfAttnDecoderBlock(
            d_model=8,
            n_heads=2,
            dropout=0.25,
            attention_impl="sdpa",
            norm_placement="post"
        )

        self.assertFalse(hasattr(block, "attn_dropout"))
        self.assertEqual(block.multi_head_attention.p, 0.25)

    def test_baseline_model_adds_final_layer_norm_only_for_pre_norm(self) -> None:
        """Validates: `BaselineModel` creates `final_ln` only for pre-norm mode.
        Why: pre-norm requires a final normalization pass before logits; post-norm should not add this extra layer.
        """
        pre_model = BaselineModel(
            vocab_size=32,
            layers=1,
            d_model=8,
            n_heads=2,
            pad_id=31,
            norm_placement="pre",
        )
        post_model = BaselineModel(
            vocab_size=32,
            layers=1,
            d_model=8,
            n_heads=2,
            pad_id=31,
            norm_placement="post",
        )

        self.assertTrue(hasattr(pre_model, "final_ln"))
        self.assertFalse(hasattr(post_model, "final_ln"))

    def test_pre_norm_block_passes_normalized_input_to_attention(self) -> None:
        """Validates: pre-norm path feeds `ln1(Q)` into attention, not raw `Q`.
        Why: this is the defining behavior of pre-norm; regressions here silently change training dynamics.
        """
        block = SelfAttnDecoderBlock(
            d_model=8,
            n_heads=2,
            norm_placement="pre",
        )
        recorder = _RecorderAttention()
        block.multi_head_attention = recorder
        q = torch.randn(2, 4, 8)
        _ = block(q)

        expected = block.ln1(q)
        captured = recorder.last_q
        self.assertIsNotNone(captured)
        if captured is None:
            self.fail("Expected attention module to receive an input tensor.")
        self.assertTrue(torch.allclose(captured, expected))

    def test_post_norm_block_passes_raw_input_to_attention(self) -> None:
        """Validates: post-norm path feeds raw `Q` into attention.
        Why: post-norm semantics differ from pre-norm at this exact point; this protects the intended architecture split.
        """
        block = SelfAttnDecoderBlock(
            d_model=8,
            n_heads=2,
            norm_placement="post",
        )
        recorder = _RecorderAttention()
        block.multi_head_attention = recorder
        q = torch.randn(2, 4, 8)
        _ = block(q)

        captured = recorder.last_q
        self.assertIsNotNone(captured)
        if captured is None:
            self.fail("Expected attention module to receive an input tensor.")
        self.assertTrue(torch.allclose(captured, q))


class BasicAttentionDropoutTests(unittest.TestCase):
    def test_basic_attention_applies_dropout_only_in_train_mode(self) -> None:
        attn = BasicMultiHeadSelfAttention(
            E_q=8,
            E_out=8,
            n_heads=2,
            E_bias=True,
            dropout=0.5,
        )
        inputs = torch.randn(2, 4, 8)

        attn.train()
        torch.manual_seed(0)
        train_output = attn(inputs, is_causal=True)

        attn.eval()
        torch.manual_seed(0)
        eval_output = attn(inputs, is_causal=True)

        self.assertFalse(torch.allclose(train_output, eval_output))


class CompiledSDPAMetricsSmokeTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile unavailable")
    def test_sdpa_attention_entropy_metrics_work_with_compiled_model(self) -> None:
        model = BaselineModel(
            vocab_size=32,
            layers=2,
            d_model=8,
            n_heads=2,
            pad_id=31,
            attention_impl="sdpa",
        )
        compiled_model = torch.compile(model)
        plugin = ForwardHookMetricsPlugin(
            wandb_enabled=True,
            wandb_cfg=WandbMetricsConfig(
                enable_activation_norms=False,
                enable_attention_entropy=True,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=4,
            ),
            model=model,
            layer_labels=get_decoder_layer_labels(model),
        )

        plugin.on_train_start()
        try:
            ctx = _step_ctx(_metric_schedule(capture_attention_entropy=True))
            plugin.on_step_start(ctx)
            inputs = torch.randint(0, 30, (2, 4), dtype=torch.long)
            key_padding_mask = torch.ones((2, 4), dtype=torch.bool)
            _ = compiled_model(inputs, key_padding_mask=key_padding_mask)
            metrics = plugin.collect_step_metrics(ctx)
        finally:
            plugin.on_train_end()

        self.assertIn("attention_entropy_first", metrics)


if __name__ == "__main__":
    unittest.main()
