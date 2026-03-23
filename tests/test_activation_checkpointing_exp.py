from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

from src.training.runtime import preflight_dynamo_activation_memory_budget_api


class ActivationCheckpointingExperimentPreflightTests(unittest.TestCase):
    def test_preflight_fails_when_budget_api_is_missing(self) -> None:
        budgets = [0.5]
        fake_dynamo = types.SimpleNamespace(config=types.SimpleNamespace())
        fake_functorch = types.SimpleNamespace(config=types.SimpleNamespace())
        with (
            patch.object(torch, "_dynamo", fake_dynamo, create=True),
            patch.object(torch, "_functorch", fake_functorch, create=True),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "torch._functorch.config.activation_memory_budget is unavailable",
            ):
                preflight_dynamo_activation_memory_budget_api(budgets)

    def test_preflight_skips_when_no_budgeted_variants(self) -> None:
        preflight_dynamo_activation_memory_budget_api([None])


if __name__ == "__main__":
    unittest.main()
