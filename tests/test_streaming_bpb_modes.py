from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import HFTextDatasetConfig
from src.training.data import HFStreamingWindowDataset


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


class StreamingBpbModesTests(unittest.TestCase):
    def _make_dataset(
        self,
        *,
        bpb_mode: str,
        tokenizer: _FakeTokenizer | None = None,
        approx_lookup: list[int] | None = None,
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


if __name__ == "__main__":
    unittest.main()
