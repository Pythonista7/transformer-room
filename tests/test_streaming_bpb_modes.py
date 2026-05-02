from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from src.config import HFTextDatasetConfig
from src.training.data import HFStreamingWindowDataset
from src.training.evaluate import evaluate


class _FakeStreamingRows:
    def __init__(self, rows):
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def shuffle(self, *, seed: int, buffer_size: int):
        _ = seed
        _ = buffer_size
        return self

    def set_epoch(self, epoch: int) -> None:
        _ = epoch

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        _ = state_dict


class _FakeTokenizer:
    def __init__(self, *, offsets_supported: bool = True) -> None:
        self.offsets_supported = offsets_supported

    def __call__(self, text: str, **kwargs):
        _ = kwargs["add_special_tokens"]
        _ = kwargs["return_attention_mask"]
        _ = kwargs["return_token_type_ids"]
        if text != "a🙂":
            raise ValueError(f"Unexpected sample text: {text!r}")

        output = {"input_ids": [1, 2]}
        if kwargs.get("return_offsets_mapping", False) and self.offsets_supported:
            output["offset_mapping"] = [(0, 1), (1, 2)]
        return output


class _MultiTextTokenizer:
    _token_ids_by_text = {
        "ab": [1, 2],
        "cd": [3, 4],
    }

    def __call__(self, text: str, **kwargs):
        _ = kwargs["add_special_tokens"]
        _ = kwargs["return_attention_mask"]
        _ = kwargs["return_token_type_ids"]
        return {"input_ids": list(self._token_ids_by_text[text])}


class _RecordingZeroLogitsModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.inputs_seen: list[list[list[int]]] = []

    def forward(
        self,
        input_seq: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = key_padding_mask
        self.inputs_seen.append(input_seq.detach().cpu().tolist())
        batch_size, seq_len = input_seq.shape
        return torch.zeros(
            (batch_size, seq_len, self.vocab_size),
            dtype=torch.float32,
            device=input_seq.device,
        )


class StreamingBpbModesTests(unittest.TestCase):
    def _make_dataset(
        self,
        *,
        bpb_mode: str,
        tokenizer: _FakeTokenizer | None = None,
        approx_lookup: list[int] | None = None,
        reset_on_iter_close: bool = False,
    ) -> HFStreamingWindowDataset:
        dataset_cfg = HFTextDatasetConfig(
            dataset_name="dummy/dataset",
            split="train",
            text_field="text",
            shuffle_buffer_size=8,
        )
        return HFStreamingWindowDataset(
            dataset_cfg=dataset_cfg,
            split="train",
            tokenizer=tokenizer or _FakeTokenizer(),
            seq_len=2,
            stride=2,
            pad_id=4,
            eos_id=3,
            seed=42,
            shuffle=False,
            bpb_mode=bpb_mode,  # type: ignore[arg-type]
            approx_token_byte_lengths=approx_lookup,
            reset_on_iter_close=reset_on_iter_close,
        )

    def _make_multi_text_dataset(
        self,
        *,
        reset_on_iter_close: bool,
    ) -> HFStreamingWindowDataset:
        dataset_cfg = HFTextDatasetConfig(
            dataset_name="dummy/dataset",
            split="train",
            text_field="text",
            shuffle_buffer_size=8,
        )
        return HFStreamingWindowDataset(
            dataset_cfg=dataset_cfg,
            split="train",
            tokenizer=_MultiTextTokenizer(),
            seq_len=2,
            stride=2,
            pad_id=6,
            eos_id=5,
            seed=42,
            shuffle=False,
            reset_on_iter_close=reset_on_iter_close,
        )

    def _first_target_byte_lengths(self, dataset: HFStreamingWindowDataset) -> list[int]:
        sample = next(iter(dataset))
        self.assertEqual(len(sample), 4)
        return sample[3].tolist()

    @staticmethod
    def _fake_loader_factory(rows):
        def _load_dataset(**kwargs):
            _ = kwargs
            return _FakeStreamingRows(rows)

        return _load_dataset

    def test_bpb_mode_off_emits_zero_target_byte_lengths(self) -> None:
        dataset = self._make_dataset(bpb_mode="off")
        with patch(
            "src.training.data._resolve_load_dataset",
            return_value=self._fake_loader_factory([{"text": "a🙂"}]),
        ):
            target_bytes = self._first_target_byte_lengths(dataset)
        self.assertEqual(target_bytes, [0, 0])

    def test_bpb_mode_approx_uses_lookup_table(self) -> None:
        # Token id 2 corresponds to "🙂" in our fake sample, but lookup sets it to 1
        # to keep approx intentionally different from exact.
        approx_lookup = [0, 1, 1, 0, 0]
        dataset = self._make_dataset(
            bpb_mode="approx",
            approx_lookup=approx_lookup,
        )
        with patch(
            "src.training.data._resolve_load_dataset",
            return_value=self._fake_loader_factory([{"text": "a🙂"}]),
        ):
            target_bytes = self._first_target_byte_lengths(dataset)
        self.assertEqual(target_bytes, [1, 0])

    def test_bpb_mode_exact_uses_offsets_based_utf8_bytes(self) -> None:
        dataset = self._make_dataset(bpb_mode="exact")
        with patch(
            "src.training.data._resolve_load_dataset",
            return_value=self._fake_loader_factory([{"text": "a🙂"}]),
        ):
            target_bytes = self._first_target_byte_lengths(dataset)
        self.assertEqual(target_bytes, [4, 0])

    def test_bpb_mode_exact_fails_when_offsets_are_missing(self) -> None:
        dataset = self._make_dataset(
            bpb_mode="exact",
            tokenizer=_FakeTokenizer(offsets_supported=False),
        )
        with patch(
            "src.training.data._resolve_load_dataset",
            return_value=self._fake_loader_factory([{"text": "a🙂"}]),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "did not provide offsets",
            ):
                next(iter(dataset))

    def test_stateless_streaming_validation_resets_after_capped_evaluate(self) -> None:
        dataset = self._make_multi_text_dataset(reset_on_iter_close=True)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = _RecordingZeroLogitsModel(vocab_size=7)
        loss_fn = CrossEntropyLoss(ignore_index=6, reduction="sum")
        rows = [{"text": "ab"}, {"text": "cd"}]

        with patch(
            "src.training.data._resolve_load_dataset",
            return_value=self._fake_loader_factory(rows),
        ):
            first_metrics = evaluate(
                model,
                loader,
                loss_fn,
                device=torch.device("cpu"),
                use_bf16=False,
                max_eval_batches=1,
            )
            first_input = model.inputs_seen[-1]
            first_state = dataset.state_dict()

            second_metrics = evaluate(
                model,
                loader,
                loss_fn,
                device=torch.device("cpu"),
                use_bf16=False,
                max_eval_batches=1,
            )
            second_input = model.inputs_seen[-1]
            second_state = dataset.state_dict()

        self.assertEqual(first_input, second_input)
        self.assertAlmostEqual(first_metrics["val_loss"], second_metrics["val_loss"])
        self.assertTrue(math.isnan(first_metrics["val_bits_per_byte"]))
        self.assertEqual(first_state["buffer_tokens"], [])
        self.assertEqual(first_state["total_tokens_seen"], 0)
        self.assertEqual(first_state["usable_rows_seen"], 0)
        self.assertIsNone(first_state["source_state_dict"])
        self.assertEqual(second_state["buffer_tokens"], [])
        self.assertEqual(second_state["total_tokens_seen"], 0)
        self.assertEqual(second_state["usable_rows_seen"], 0)
        self.assertIsNone(second_state["source_state_dict"])

    def test_stateful_streaming_dataset_preserves_state_on_early_close(self) -> None:
        dataset = self._make_multi_text_dataset(reset_on_iter_close=False)
        rows = [{"text": "ab"}, {"text": "cd"}]

        with patch(
            "src.training.data._resolve_load_dataset",
            return_value=self._fake_loader_factory(rows),
        ):
            iterator = iter(dataset)
            first_sample = next(iterator)
            iterator.close()

        saved_state = dataset.state_dict()
        restored = self._make_multi_text_dataset(reset_on_iter_close=False)
        restored.load_state_dict(saved_state)

        self.assertEqual(first_sample[0].tolist(), [1, 2])
        self.assertNotEqual(saved_state["buffer_tokens"], [])
        self.assertGreater(saved_state["total_tokens_seen"], 0)
        self.assertEqual(restored.state_dict()["buffer_tokens"], saved_state["buffer_tokens"])
        self.assertEqual(
            restored.state_dict()["total_tokens_seen"],
            saved_state["total_tokens_seen"],
        )


if __name__ == "__main__":
    unittest.main()
