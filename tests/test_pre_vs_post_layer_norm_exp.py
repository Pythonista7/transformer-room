from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import experiments.baseline.pre_vs_post_layer_norm as layer_norm_exp


class PreVsPostLayerNormExperimentTests(unittest.TestCase):
    def test_build_configs_emits_four_wandb_variants(self) -> None:
        with mock.patch.object(
            layer_norm_exp.training_wikitext,
            "ensure_wikitext_vocab_file",
            return_value=128,
        ):
            configs = layer_norm_exp.build_configs()

        self.assertEqual(
            set(configs.keys()),
            {
                "pre_no_warmup",
                "pre_warmup",
                "post_no_warmup",
                "post_warmup",
            },
        )
        self.assertEqual(configs["pre_no_warmup"].model.norm_placement, "pre")
        self.assertEqual(configs["post_warmup"].model.norm_placement, "post")
        self.assertEqual(configs["pre_no_warmup"].train.lr_warmup_steps, 0)
        self.assertEqual(
            configs["pre_warmup"].train.lr_warmup_steps,
            layer_norm_exp.LR_WARMUP_STEPS,
        )
        self.assertEqual(
            configs["pre_warmup"].train.lr_warmup_start_factor,
            layer_norm_exp.LR_WARMUP_START_FACTOR,
        )
        self.assertEqual(configs["post_warmup"].logging.provider, "wandb")
        self.assertTrue(configs["post_warmup"].logging.wandb.enable_layer_grad_norms)
        self.assertEqual(
            configs["post_warmup"].logging.wandb.layer_grad_norm_stride,
            layer_norm_exp.LAYER_GRAD_STRIDE,
        )
        self.assertEqual(
            configs["post_warmup"].logging.wandb.layer_grad_norms_every_n_steps,
            layer_norm_exp.LAYER_GRAD_EVERY_N_STEPS,
        )

    def test_main_runs_all_variants(self) -> None:
        fake_result = SimpleNamespace(run_artifact_dir="/tmp/run-dir")
        with (
            mock.patch.object(
                layer_norm_exp.training_wikitext,
                "ensure_wikitext_vocab_file",
                return_value=128,
            ),
            mock.patch.object(
                layer_norm_exp,
                "model_pipeline",
                return_value=fake_result,
            ) as pipeline_mock,
        ):
            exit_code = layer_norm_exp.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(pipeline_mock.call_count, 4)


if __name__ == "__main__":
    unittest.main()
