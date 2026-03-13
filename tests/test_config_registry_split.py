from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters import register_builtin_adapters
from src.config import (
    ACEveryNDecoderConfig,
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    SACDecoderConfig,
    TrainConfig,
    WandbMetricsConfig,
    validate_experiment_config,
)
from src.core.registry import get_dataset_adapter, get_model_adapter
from experiments.baseline.hyperparam_sweeps.LRvsBatchSizeEmpiricalSweep import (
    build_config as build_lr_vs_batch_config,
)

register_builtin_adapters()


class DummyDataset:
    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> int:
        return idx


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(project_name="test-project", artifacts_root="/tmp/artifacts"),
        dataset=LocalTextDatasetConfig(path="/tmp/dataset.txt"),
        tokenizer=BPETokenizerConfig(vocab_path="/tmp/vocab.txt"),
        model=BaselineDecoderConfig(d_model=32, n_heads=4, layers=1),
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
            effective_batch_size=4,
            seq_len=16,
            stride=16,
        ),
        split=HoldoutSplitConfig(train_fraction=0.9, seed=42, shuffle=True),
        logging=LoggingConfig(provider="console"),
    )


class ConfigValidationTests(unittest.TestCase):
    def test_train_batching_defaults_to_non_accumulation(self) -> None:
        train_cfg = TrainConfig(effective_batch_size=8)
        self.assertEqual(train_cfg.micro_batch_size, 8)
        self.assertEqual(train_cfg.accumulation_steps, 1)

    def test_train_batching_explicit_accumulation_passes(self) -> None:
        train_cfg = TrainConfig(
            effective_batch_size=8,
            micro_batch_size=4,
            accumulation_steps=2,
        )
        self.assertEqual(train_cfg.effective_batch_size, 8)
        self.assertEqual(train_cfg.micro_batch_size, 4)
        self.assertEqual(train_cfg.accumulation_steps, 2)

    def test_train_batching_equation_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "effective_batch_size must equal",
        ):
            TrainConfig(
                effective_batch_size=8,
                micro_batch_size=3,
                accumulation_steps=2,
            )

    def test_train_batching_non_positive_values_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "effective_batch_size must be > 0"):
            TrainConfig(effective_batch_size=0)
        with self.assertRaisesRegex(ValueError, "micro_batch_size must be > 0"):
            TrainConfig(effective_batch_size=8, micro_batch_size=0)
        with self.assertRaisesRegex(ValueError, "accumulation_steps must be > 0"):
            TrainConfig(effective_batch_size=8, accumulation_steps=0)

    def test_train_config_rejects_legacy_batch_size_keyword(self) -> None:
        with self.assertRaises(TypeError):
            TrainConfig(batch_size=8)  # type: ignore[call-arg]

    def test_validate_experiment_config_rechecks_mutated_batching(self) -> None:
        config = make_config()
        config.train.micro_batch_size = 3
        config.train.accumulation_steps = 1
        with self.assertRaisesRegex(
            ValueError,
            "effective_batch_size must equal",
        ):
            validate_experiment_config(config)

    def test_invalid_dataset_name_fails(self) -> None:
        config = make_config()
        config.dataset.name = "not_real"  # type: ignore[assignment]
        with self.assertRaisesRegex(ValueError, "Unsupported dataset.name"):
            validate_experiment_config(config)

    def test_invalid_num_special_tokens_fails(self) -> None:
        config = make_config()
        config.tokenizer.num_special_tokens = 1
        with self.assertRaisesRegex(ValueError, "at least 2 special tokens"):
            validate_experiment_config(config)

    def test_invalid_optimizer_weight_decay_fails(self) -> None:
        config = make_config()
        config.train.optimizer.weight_decay = -0.1
        with self.assertRaisesRegex(
            ValueError,
            "train.optimizer.weight_decay must be >= 0",
        ):
            validate_experiment_config(config)

    def test_supported_optimizer_names_pass_validation(self) -> None:
        for optimizer_name in ("adam", "adamw", "sgd"):
            config = make_config()
            config.train.optimizer.name = optimizer_name
            validate_experiment_config(config)

    def test_invalid_compile_warmup_steps_fails(self) -> None:
        config = make_config()
        config.run.compile_warmup_steps = -1
        with self.assertRaisesRegex(ValueError, "compile_warmup_steps must be >= 0"):
            validate_experiment_config(config)

    def test_invalid_activation_memory_budget_fails(self) -> None:
        config = make_config()
        config.run.activation_memory_budget = 1.5
        with self.assertRaisesRegex(
            ValueError,
            "activation_memory_budget must be in \\(0, 1\\] when set",
        ):
            validate_experiment_config(config)

    def test_artifact_controls_default_to_enabled_and_serialize(self) -> None:
        config = make_config()
        self.assertTrue(config.run.persist_local_artifacts)
        self.assertTrue(config.logging.enable_artifact_io)

        payload = config.to_dict()
        self.assertTrue(payload["run"]["persist_local_artifacts"])
        self.assertTrue(payload["logging"]["enable_artifact_io"])

    def test_invalid_optimizer_name_fails(self) -> None:
        config = make_config()
        config.train.optimizer.name = "rmsprop"  # type: ignore[assignment]
        with self.assertRaisesRegex(
            ValueError,
            "Expected one of: adam, adamw, sgd",
        ):
            validate_experiment_config(config)

    def test_invalid_wandb_log_every_n_steps_fails(self) -> None:
        config = make_config()
        config.run.run_name = "wandb-validation-run"
        config.logging = LoggingConfig(
            provider="wandb",
            wandb=WandbMetricsConfig(log_every_n_steps=0),
        )
        with self.assertRaisesRegex(ValueError, "log_every_n_steps must be > 0"):
            validate_experiment_config(config)

    def test_invalid_wandb_val_every_n_steps_fails(self) -> None:
        config = make_config()
        config.run.run_name = "wandb-validation-run"
        config.logging = LoggingConfig(
            provider="wandb",
            wandb=WandbMetricsConfig(val_every_n_steps=-1),
        )
        with self.assertRaisesRegex(ValueError, "val_every_n_steps must be >= 0"):
            validate_experiment_config(config)

    def test_invalid_wandb_param_optimizer_norms_every_n_steps_fails(self) -> None:
        config = make_config()
        config.run.run_name = "wandb-validation-run"
        config.logging = LoggingConfig(
            provider="wandb",
            wandb=WandbMetricsConfig(parameter_optimizer_norms_every_n_steps=0),
        )
        with self.assertRaisesRegex(
            ValueError,
            "parameter_optimizer_norms_every_n_steps must be > 0",
        ):
            validate_experiment_config(config)

    def test_wandb_requires_run_name(self) -> None:
        config = make_config()
        config.logging = LoggingConfig(provider="wandb")
        config.run.run_name = None
        with self.assertRaisesRegex(ValueError, "requires run.run_name"):
            validate_experiment_config(config)

    def test_sweep_run_name_is_deterministic_and_group_name_can_vary(self) -> None:
        config_a = build_lr_vs_batch_config(learning_rate=1e-4, effective_batch_size=20, sweep_group="group-a")
        config_b = build_lr_vs_batch_config(learning_rate=1e-4, effective_batch_size=20, sweep_group="group-b")

        self.assertEqual(
            config_a.run.run_name,
            "p0-group-LRvsBSz-wikitext2-gpt2-lr0p0001-bs20",
        )
        self.assertEqual(config_a.run.run_name, config_b.run.run_name)
        self.assertNotEqual(config_a.run.group_name, config_b.run.group_name)

    def test_ac_every_n_model_validation(self) -> None:
        config = make_config()
        config.model = ACEveryNDecoderConfig(
            d_model=32,
            n_heads=4,
            layers=1,
            checkpoint_every_n_layers=2,
        )
        validate_experiment_config(config)

    def test_ac_every_n_invalid_checkpoint_interval_fails(self) -> None:
        config = make_config()
        config.model = ACEveryNDecoderConfig(
            d_model=32,
            n_heads=4,
            layers=1,
            checkpoint_every_n_layers=0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "model.checkpoint_every_n_layers must be > 0",
        ):
            validate_experiment_config(config)

    def test_sac_model_validation(self) -> None:
        config = make_config()
        config.model = SACDecoderConfig(d_model=32, n_heads=4, layers=1)
        validate_experiment_config(config)


class RegistryAndSplitTests(unittest.TestCase):
    def test_missing_registry_key_is_actionable(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown dataset adapter"):
            get_dataset_adapter("does_not_exist")

    def test_holdout_split_is_deterministic(self) -> None:
        adapter = get_dataset_adapter("local_text")
        self.assertIsNotNone(adapter)

        from src.core.registry import get_split_adapter

        split_adapter = get_split_adapter("holdout")
        cfg = HoldoutSplitConfig(train_fraction=0.6, seed=7, shuffle=True)
        dataset = DummyDataset(20)

        train_a, val_a = split_adapter.split(dataset=dataset, cfg=cfg)
        train_b, val_b = split_adapter.split(dataset=dataset, cfg=cfg)

        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(val_a.indices, val_b.indices)

    def test_new_model_adapters_are_registered(self) -> None:
        self.assertIsNotNone(get_model_adapter("ac_every_n_decoder"))
        self.assertIsNotNone(get_model_adapter("sac_decoder"))


class LocalDatasetAdapterTests(unittest.TestCase):
    def test_local_dataset_empty_file_fails(self) -> None:
        adapter = get_dataset_adapter("local_text")
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "empty.txt"
            dataset_path.write_text("", encoding="utf-8")
            cfg = LocalTextDatasetConfig(path=str(dataset_path))
            with self.assertRaisesRegex(ValueError, "Local dataset is empty"):
                adapter.load(cfg)


if __name__ == "__main__":
    unittest.main()
