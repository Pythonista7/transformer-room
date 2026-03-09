from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

from experiments.baseline.memory_experiments.activation_checkpointing_exp import (
    VariantSpec,
    preflight_dynamo_activation_memory_budget_api,
)
from src.config import BaselineDecoderConfig


class ActivationCheckpointingExperimentPreflightTests(unittest.TestCase):
    def test_preflight_fails_when_budget_api_is_missing(self) -> None:
        variants = [
            VariantSpec(
                key="budgeted",
                model_cfg=BaselineDecoderConfig(d_model=32, n_heads=4, layers=1),
                use_torch_compile=True,
                activation_memory_budget=0.5,
            )
        ]
        fake_dynamo = types.SimpleNamespace(config=types.SimpleNamespace())
        with patch.object(torch, "_dynamo", fake_dynamo, create=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "torch._dynamo.config.activation_memory_budget is unavailable",
            ):
                preflight_dynamo_activation_memory_budget_api(variants)

    def test_preflight_skips_when_no_budgeted_variants(self) -> None:
        variants = [
            VariantSpec(
                key="non_budget",
                model_cfg=BaselineDecoderConfig(d_model=32, n_heads=4, layers=1),
                use_torch_compile=True,
                activation_memory_budget=None,
            )
        ]
        preflight_dynamo_activation_memory_budget_api(variants)


if __name__ == "__main__":
    unittest.main()
