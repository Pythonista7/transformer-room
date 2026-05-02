from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import experiments.baseline.pre_vs_post_layer_norm as layer_norm_exp


class PreVsPostLayerNormExperimentTests(unittest.TestCase):
    def test_build_variant_configs_emits_expected_variants(self) -> None:
        with mock.patch.object(
            layer_norm_exp.training_wikitext,
            "ensure_wikitext_vocab_file",
            return_value=128,
        ):
            pairs = layer_norm_exp.build_variant_configs()
        configs = {
            spec.run_name: config
            for spec, config in pairs
        }

        self.assertEqual(
            set(configs),
            {
                "post-ln-shallow-warmup0",
                f"post-ln-shallow-warmup{layer_norm_exp.LR_WARMUP_STEPS}",
                "post-ln-deep-warmup0",
                f"post-ln-deep-warmup{layer_norm_exp.LR_WARMUP_STEPS}",
                "pre-ln-shallow-warmup0",
                "pre-ln-deep-warmup0",
            },
        )
        self.assertEqual(configs["pre-ln-shallow-warmup0"].model.norm_placement, "pre")
        self.assertEqual(configs["post-ln-deep-warmup0"].model.norm_placement, "post")
        self.assertIsNone(configs["pre-ln-shallow-warmup0"].train.lr_scheduler)
        warmup_cfg = configs[f"post-ln-shallow-warmup{layer_norm_exp.LR_WARMUP_STEPS}"]
        self.assertIsNotNone(warmup_cfg.train.lr_scheduler)
        self.assertEqual(len(warmup_cfg.train.lr_scheduler.stages), 1)
        stage = warmup_cfg.train.lr_scheduler.stages[0]
        self.assertEqual(stage.type, "linear")
        self.assertEqual(
            stage.steps,
            layer_norm_exp.LR_WARMUP_STEPS,
        )
        self.assertEqual(
            stage.start_factor,
            layer_norm_exp.LR_WARMUP_START_FACTOR,
        )
        self.assertEqual(stage.end_factor, 1.0)
        self.assertEqual(warmup_cfg.logging.provider, "wandb")
        self.assertTrue(warmup_cfg.logging.wandb.enable_layer_grad_norms)
        self.assertEqual(
            warmup_cfg.logging.wandb.layer_grad_norm_stride,
            layer_norm_exp.LAYER_GRAD_STRIDE,
        )
        self.assertEqual(
            warmup_cfg.logging.wandb.layer_grad_norms_every_n_steps,
            layer_norm_exp.LAYER_GRAD_EVERY_N_STEPS,
        )

    def test_main_runs_all_variants(self) -> None:
        fake_result = SimpleNamespace(
            run_artifact_dir="/tmp/run-dir",
            global_step=10,
            final_train_loss=1.0,
            final_val_metrics_by_source={
                "holdout": {
                    "val_loss": 1.1,
                    "val_perplexity": 3.0,
                }
            },
        )
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
            mock.patch.object(
                layer_norm_exp,
                "_write_summary_artifacts",
                return_value=layer_norm_exp.SUMMARY_ROOT / "fake-summary",
            ),
        ):
            exit_code = layer_norm_exp.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(pipeline_mock.call_count, len(layer_norm_exp.VARIANT_SPECS))


if __name__ == "__main__":
    unittest.main()
