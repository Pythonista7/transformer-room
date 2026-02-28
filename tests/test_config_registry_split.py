from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import baseline.adapters  # noqa: F401
from baseline.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
    validate_experiment_config,
)
from baseline.core.registry import get_dataset_adapter
from baseline.experiments.LRvsBatchSizeEmpiricalSweep import build_config as build_lr_vs_batch_config


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
        train=TrainConfig(epochs=1, learning_rate=1e-3, batch_size=4, seq_len=16, stride=16),
        split=HoldoutSplitConfig(train_fraction=0.9, seed=42, shuffle=True),
        logging=LoggingConfig(provider="console"),
    )


class ConfigValidationTests(unittest.TestCase):
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

    def test_wandb_requires_run_name(self) -> None:
        config = make_config()
        config.logging = LoggingConfig(provider="wandb")
        config.run.run_name = None
        with self.assertRaisesRegex(ValueError, "requires run.run_name"):
            validate_experiment_config(config)

    def test_sweep_run_name_is_deterministic_and_group_name_can_vary(self) -> None:
        config_a = build_lr_vs_batch_config(learning_rate=1e-4, batch_size=20, sweep_group="group-a")
        config_b = build_lr_vs_batch_config(learning_rate=1e-4, batch_size=20, sweep_group="group-b")

        self.assertEqual(
            config_a.run.run_name,
            "p0-LRvsBSz-wikitext2-gpt2-lr0p0001-bs20",
        )
        self.assertEqual(config_a.run.run_name, config_b.run.run_name)
        self.assertNotEqual(config_a.run.group_name, config_b.run.group_name)


class RegistryAndSplitTests(unittest.TestCase):
    def test_missing_registry_key_is_actionable(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown dataset adapter"):
            get_dataset_adapter("does_not_exist")

    def test_holdout_split_is_deterministic(self) -> None:
        adapter = get_dataset_adapter("local_text")
        self.assertIsNotNone(adapter)

        from baseline.core.registry import get_split_adapter

        split_adapter = get_split_adapter("holdout")
        cfg = HoldoutSplitConfig(train_fraction=0.6, seed=7, shuffle=True)
        dataset = DummyDataset(20)

        train_a, val_a = split_adapter.split(dataset=dataset, cfg=cfg)
        train_b, val_b = split_adapter.split(dataset=dataset, cfg=cfg)

        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(val_a.indices, val_b.indices)


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
