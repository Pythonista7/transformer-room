from __future__ import annotations

import math
import unittest

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.training.evaluate import evaluate


class _ConstantLogitsModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_seq: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = key_padding_mask
        batch_size, seq_len = input_seq.shape
        return torch.zeros(
            (batch_size, seq_len, self.vocab_size),
            dtype=torch.float32,
            device=input_seq.device,
        )


class EvaluateBitsPerByteTests(unittest.TestCase):
    def test_returns_true_byte_normalized_bits_per_byte(self) -> None:
        vocab_size = 5
        pad_id = 4
        model = _ConstantLogitsModel(vocab_size=vocab_size)
        loss_fn = CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

        samples = [
            (
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.tensor([1, 2, 3], dtype=torch.long),
                torch.tensor([True, True, True], dtype=torch.bool),
            ),
            (
                torch.tensor([0, 1, 2], dtype=torch.long),
                torch.tensor([4, 4, 4], dtype=torch.long),
                torch.tensor([True, True, True], dtype=torch.bool),
            ),
        ]
        loader = DataLoader(samples, batch_size=2, shuffle=False)
        token_byte_lengths = [1, 2, 3, 4, 0]

        val_metrics = evaluate(
            model,
            loader,
            loss_fn,
            device=torch.device("cpu"),
            use_bf16=False,
            token_byte_lengths=token_byte_lengths,
        )

        expected_avg_loss = math.log(vocab_size)
        expected_perplexity = float(vocab_size)
        expected_bits_per_byte = (math.log(vocab_size) / math.log(2.0)) / 3.0
        self.assertAlmostEqual(val_metrics["val_loss"], expected_avg_loss, places=7)
        self.assertAlmostEqual(
            val_metrics["val_perplexity"],
            expected_perplexity,
            places=6,
        )
        self.assertAlmostEqual(
            val_metrics["val_bits_per_byte"],
            expected_bits_per_byte,
            places=7,
        )

    def test_returns_nan_bits_per_byte_when_denominator_is_zero(self) -> None:
        vocab_size = 5
        pad_id = 4
        model = _ConstantLogitsModel(vocab_size=vocab_size)
        loss_fn = CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

        samples = [
            (
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([3, 3], dtype=torch.long),
                torch.tensor([True, True], dtype=torch.bool),
            )
        ]
        loader = DataLoader(samples, batch_size=1, shuffle=False)
        token_byte_lengths = [1, 2, 3, 0, 0]

        val_metrics = evaluate(
            model,
            loader,
            loss_fn,
            device=torch.device("cpu"),
            use_bf16=False,
            token_byte_lengths=token_byte_lengths,
        )

        self.assertAlmostEqual(val_metrics["val_loss"], math.log(vocab_size), places=7)
        self.assertAlmostEqual(
            val_metrics["val_perplexity"],
            float(vocab_size),
            places=6,
        )
        self.assertTrue(math.isnan(val_metrics["val_bits_per_byte"]))


if __name__ == "__main__":
    unittest.main()
