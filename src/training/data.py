from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..adapters.datasets import _infer_text_field_from_sample, _normalize_text_value
from ..core.config import (
    ExperimentConfig,
    HFStreamingSourceConfig,
    normalize_validation_source_name,
    resolve_train_batching,
    resolve_validation_sources,
)
from ..core.registry import get_split_adapter
from ..core.types import TokenizedCorpus


class LMWindowDataset(Dataset):
    """Fixed-window LM dataset with optional tail padding and key padding mask."""

    def __init__(self, tokens: list[int], seq_len: int, stride: int, pad_id: int):
        if not tokens:
            raise ValueError("Token stream is empty after preprocessing.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")

        self.tokens = tokens
        self.seq_len = seq_len
        self.window = seq_len + 1
        self.stride = stride
        self.pad_id = pad_id
        self.starts = self._build_starts()

    def _build_starts(self) -> list[int]:
        token_count = len(self.tokens)
        if token_count <= self.window:
            return [0]

        full_limit = token_count - self.window + 1
        starts = list(range(0, full_limit, self.stride))
        if not starts:
            starts = [0]

        next_start = starts[-1] + self.stride
        if next_start < token_count:
            starts.append(next_start)
        return starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        sample = self.tokens[start : start + self.window]
        if len(sample) < self.window:
            sample = sample + [self.pad_id] * (self.window - len(sample))

        sample_tensor = torch.tensor(sample, dtype=torch.long)
        input_seq = sample_tensor[:-1]
        target_seq = sample_tensor[1:]
        key_padding_mask = input_seq != self.pad_id
        return input_seq, target_seq, key_padding_mask


type ValLoaderSpec = tuple[str, DataLoader, int | None]


def _build_lm_tensors(
    sample: list[int],
    pad_id: int,
    sample_byte_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    sample_tensor = torch.tensor(sample, dtype=torch.long)
    input_seq = sample_tensor[:-1]
    target_seq = sample_tensor[1:]
    key_padding_mask = input_seq != pad_id
    if sample_byte_lengths is None:
        return input_seq, target_seq, key_padding_mask
    sample_byte_lengths_tensor = torch.tensor(sample_byte_lengths, dtype=torch.long)
    target_byte_lengths = sample_byte_lengths_tensor[1:]
    return input_seq, target_seq, key_padding_mask, target_byte_lengths


def _utf8_char_byte_prefix(text: str) -> list[int]:
    prefix = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        prefix.append(total)
    return prefix


def _bytes_from_offsets(
    offsets: list[tuple[int, int]],
    char_byte_prefix: list[int],
) -> list[int]:
    byte_lengths: list[int] = []
    max_char_idx = len(char_byte_prefix) - 1
    for start, end in offsets:
        if start < 0 or end < 0 or end < start or end > max_char_idx:
            byte_lengths.append(0)
            continue
        byte_lengths.append(char_byte_prefix[end] - char_byte_prefix[start])
    return byte_lengths


def _resolve_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset support requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc
    return load_dataset


def _resolve_stateful_dataloader():
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError as exc:
        raise ImportError(
            "Streaming checkpoint resume requires `torchdata`. "
            "Install it with `pip install torchdata`."
        ) from exc
    return StatefulDataLoader


class HFStreamingWindowDataset(IterableDataset):
    """HF streaming dataset that tokenizes on the fly and yields LM windows."""

    def __init__(
        self,
        *,
        dataset_cfg,
        split: str,
        tokenizer,
        seq_len: int,
        stride: int,
        pad_id: int,
        eos_id: int,
        seed: int,
        shuffle: bool,
        bpb_mode: Literal["off", "approx", "exact"] = "off",
        approx_token_byte_lengths: list[int] | None = None,
        reset_on_iter_close: bool = False,
    ) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}")
        self.dataset_cfg = dataset_cfg
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.window = int(seq_len) + 1
        self.stride = int(stride)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.bpb_mode: Literal["off", "approx", "exact"] = bpb_mode
        self.approx_token_byte_lengths = approx_token_byte_lengths
        self.reset_on_iter_close = bool(reset_on_iter_close)
        self.epoch = 0
        self._text_field = dataset_cfg.text_field
        self._source_dataset = None
        self._source_state_dict = None
        self._buffer_tokens: list[int] = []
        self._buffer_byte_lengths: list[int] = []
        self._buffer_start = 0
        self._next_start = 0
        self._total_tokens_seen = 0
        self._usable_rows_seen = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, object]:
        source_state_dict = self._source_state_dict
        if self._source_dataset is not None and hasattr(self._source_dataset, "state_dict"):
            source_state_dict = self._source_dataset.state_dict()
        return {
            "epoch": int(self.epoch),
            "text_field": self._text_field,
            "source_state_dict": source_state_dict,
            "buffer_tokens": list(self._buffer_tokens),
            "buffer_byte_lengths": list(self._buffer_byte_lengths),
            "buffer_start": int(self._buffer_start),
            "next_start": int(self._next_start),
            "total_tokens_seen": int(self._total_tokens_seen),
            "usable_rows_seen": int(self._usable_rows_seen),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.epoch = int(state_dict.get("epoch", 0))
        self._text_field = state_dict.get("text_field")
        self._source_state_dict = state_dict.get("source_state_dict")
        self._buffer_tokens = [
            int(token_id) for token_id in state_dict.get("buffer_tokens", [])
        ]
        self._buffer_byte_lengths = [
            int(byte_len) for byte_len in state_dict.get("buffer_byte_lengths", [])
        ]
        self._buffer_start = int(state_dict.get("buffer_start", 0))
        self._next_start = int(state_dict.get("next_start", 0))
        self._total_tokens_seen = int(state_dict.get("total_tokens_seen", 0))
        self._usable_rows_seen = int(state_dict.get("usable_rows_seen", 0))

    def _reset_iteration_state(self) -> None:
        self._source_dataset = None
        self._source_state_dict = None
        self._buffer_tokens = []
        self._buffer_byte_lengths = []
        self._buffer_start = 0
        self._next_start = 0
        self._total_tokens_seen = 0
        self._usable_rows_seen = 0

    def _build_source_dataset(self):
        load_dataset = _resolve_load_dataset()
        load_kwargs = {
            "path": self.dataset_cfg.dataset_name,
            "split": self.split,
            "streaming": True,
        }
        if self.dataset_cfg.dataset_config:
            load_kwargs["name"] = self.dataset_cfg.dataset_config
        dataset = load_dataset(**load_kwargs)
        if self.shuffle:
            dataset = dataset.shuffle(
                seed=self.seed,
                buffer_size=self.dataset_cfg.shuffle_buffer_size,
            )
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)
        if self._source_state_dict is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(self._source_state_dict)
        self._source_dataset = dataset
        return dataset

    def _window_from_buffer(self, start: int) -> tuple[list[int], list[int]]:
        offset = start - self._buffer_start
        sample = self._buffer_tokens[offset : offset + self.window]
        sample_byte_lengths = self._buffer_byte_lengths[offset : offset + self.window]
        if len(sample) < self.window:
            pad = self.window - len(sample)
            sample = sample + [self.pad_id] * pad
            sample_byte_lengths = sample_byte_lengths + [0] * pad
        return sample, sample_byte_lengths

    def _maybe_trim_buffer(self) -> None:
        trim_count = self._next_start - self._buffer_start
        if trim_count <= 0:
            return
        self._buffer_tokens = self._buffer_tokens[trim_count:]
        self._buffer_byte_lengths = self._buffer_byte_lengths[trim_count:]
        self._buffer_start = self._next_start

    def _infer_or_validate_text_field(self, row):
        if self._text_field is not None:
            if self._text_field not in row:
                available = ", ".join(row.keys())
                raise ValueError(
                    f"dataset.text_field '{self._text_field}' not found in rows. "
                    f"Available fields: {available}"
                )
            return self._text_field
        text_field = _infer_text_field_from_sample(row)
        self._text_field = text_field
        return text_field

    def __iter__(self):
        if self.reset_on_iter_close:
            self._reset_iteration_state()

        try:
            source = self._build_source_dataset()
            iterator = iter(source)

            if self._text_field is None:
                try:
                    first_row = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"Hugging Face dataset '{self.dataset_cfg.dataset_name}' split "
                        f"'{self.split}' is empty."
                    ) from exc
                if not isinstance(first_row, Mapping):
                    raise ValueError("Expected Hugging Face dataset rows to be dict-like objects.")
                self._infer_or_validate_text_field(first_row)
                iterator = itertools.chain([first_row], iterator)

            for row in iterator:
                if self.dataset_cfg.max_rows > 0 and self._usable_rows_seen >= self.dataset_cfg.max_rows:
                    break
                if not isinstance(row, Mapping):
                    continue
                text_field = self._infer_or_validate_text_field(row)
                text = _normalize_text_value(row.get(text_field))
                if not text:
                    continue
                self._usable_rows_seen += 1

                tokenizer_kwargs = {
                    "add_special_tokens": False,
                    "return_attention_mask": False,
                    "return_token_type_ids": False,
                }
                if self.bpb_mode == "exact":
                    tokenizer_kwargs["return_offsets_mapping"] = True
                tokenized = self.tokenizer(text, **tokenizer_kwargs)
                encoded_ids = [int(token_id) for token_id in tokenized["input_ids"]]
                encoded_byte_lengths: list[int]
                if self.bpb_mode == "off":
                    encoded_byte_lengths = [0] * len(encoded_ids)
                elif self.bpb_mode == "approx":
                    if self.approx_token_byte_lengths is None:
                        raise ValueError(
                            "HF streaming bpb_mode='approx' requires token byte lookup table."
                        )
                    encoded_byte_lengths = [
                        int(self.approx_token_byte_lengths[token_id])
                        if 0 <= int(token_id) < len(self.approx_token_byte_lengths)
                        else 0
                        for token_id in encoded_ids
                    ]
                else:
                    offsets = tokenized.get("offset_mapping")
                    if offsets is None:
                        raise ValueError(
                            "HF tokenizer did not provide offsets for bpb_mode='exact'. "
                            "Use tokenizer.use_fast=True and a tokenizer with offset mapping support."
                        )
                    normalized_offsets = [
                        (int(start), int(end))
                        for start, end in offsets
                    ]
                    encoded_byte_lengths = _bytes_from_offsets(
                        normalized_offsets,
                        _utf8_char_byte_prefix(text),
                    )
                encoded_ids.append(self.eos_id)
                encoded_byte_lengths.append(0)
                self._buffer_tokens.extend(encoded_ids)
                self._buffer_byte_lengths.extend(encoded_byte_lengths)
                self._total_tokens_seen += len(encoded_ids)

                while self._next_start + self.window <= self._total_tokens_seen:
                    sample, sample_byte_lengths = self._window_from_buffer(self._next_start)
                    yield _build_lm_tensors(
                        sample,
                        self.pad_id,
                        sample_byte_lengths,
                    )
                    self._next_start += self.stride
                    self._maybe_trim_buffer()

            if self._total_tokens_seen <= 0:
                raise ValueError(
                    f"No usable text found in field '{self._text_field}' for "
                    f"dataset '{self.dataset_cfg.dataset_name}' split '{self.split}'."
                )

            while self._next_start < self._total_tokens_seen:
                sample, sample_byte_lengths = self._window_from_buffer(self._next_start)
                yield _build_lm_tensors(
                    sample,
                    self.pad_id,
                    sample_byte_lengths,
                )
                self._next_start += self.stride
                self._maybe_trim_buffer()

            self._reset_iteration_state()
        finally:
            if self.reset_on_iter_close:
                self._reset_iteration_state()


def truncate_stream_by_fraction_at_eos(
    token_stream: list[int], data_fraction: float, eos_id: int
) -> list[int]:
    if not 0 < data_fraction <= 1:
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")
    if data_fraction >= 1:
        return token_stream

    target_len = max(1, int(len(token_stream) * data_fraction))
    if target_len >= len(token_stream):
        return token_stream

    prefix = token_stream[:target_len]
    if eos_id in prefix:
        cutoff = max(i for i, token in enumerate(prefix) if token == eos_id) + 1
        return token_stream[:cutoff]

    for idx in range(target_len, len(token_stream)):
        if token_stream[idx] == eos_id:
            return token_stream[: idx + 1]
    return token_stream


def build_data_loaders(
    config: ExperimentConfig,
    tokenized: TokenizedCorpus,
    pin_memory: bool,
) -> tuple[DataLoader, list[ValLoaderSpec]]:
    batching = resolve_train_batching(config.train)
    special = tokenized.vocab.special
    if tokenized.token_stream is None:
        raise ValueError("Materialized data loading requires tokenized.token_stream.")

    token_stream = truncate_stream_by_fraction_at_eos(
        token_stream=tokenized.token_stream,
        data_fraction=config.train.data_fraction,
        eos_id=special.eos_id,
    )

    retained_eos = sum(1 for token in token_stream if token == special.eos_id)
    print(
        f"Encoded stream tokens: {len(token_stream):,} | "
        f"EOS inserted: {tokenized.eos_inserted:,} | EOS retained: {retained_eos:,} | "
        f"UNK replacements: {tokenized.unk_replacements:,}"
    )

    if any(token < 0 or token >= tokenized.vocab.vocab_size for token in token_stream):
        raise ValueError("Token stream contains token ids outside model vocab range.")

    dataset = LMWindowDataset(
        token_stream,
        seq_len=config.train.seq_len,
        stride=config.train.stride,
        pad_id=special.pad_id,
    )

    print(
        f"Dataset samples: {len(dataset):,} | "
        f"seq_len={config.train.seq_len} | stride={config.train.stride} | pad_id={special.pad_id}"
    )

    split_adapter = get_split_adapter(config.split.name)
    train_set, val_set = split_adapter.split(dataset=dataset, cfg=config.split)

    train_loader = DataLoader(
        train_set,
        batch_size=batching.loader_batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loaders: list[ValLoaderSpec] = []
    if config.train.run_validation:
        val_loader = DataLoader(
            val_set,
            batch_size=batching.loader_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        val_loaders.append(("holdout", val_loader, None))
    return train_loader, val_loaders


def build_streaming_data_loaders(
    config: ExperimentConfig,
    tokenized: TokenizedCorpus,
    pin_memory: bool,
) -> tuple[DataLoader, list[ValLoaderSpec]]:
    batching = resolve_train_batching(config.train)
    special = tokenized.vocab.special
    StatefulDataLoader = _resolve_stateful_dataloader()

    bpb_metrics_enabled = (
        config.logging.provider == "wandb"
        and config.logging.wandb.enable_bits_per_byte
    )
    bpb_mode: Literal["off", "approx", "exact"] = "off"
    approx_token_byte_lengths: list[int] | None = None
    if bpb_metrics_enabled and config.tokenizer.name == "hf_pretrained":
        bpb_mode = config.tokenizer.bpb_mode
        if bpb_mode == "approx":
            approx_token_byte_lengths = tokenized.token_byte_lengths

    train_dataset = HFStreamingWindowDataset(
        dataset_cfg=config.dataset,
        split=config.dataset.split,
        tokenizer=tokenized.tokenizer,
        seq_len=config.train.seq_len,
        stride=config.train.stride,
        pad_id=special.pad_id,
        eos_id=special.eos_id,
        seed=config.run.seed,
        shuffle=True,
        bpb_mode=bpb_mode,
        approx_token_byte_lengths=approx_token_byte_lengths,
    )
    train_loader = StatefulDataLoader(
        train_dataset,
        batch_size=batching.loader_batch_size, # This is effective batch and not micro batch
        pin_memory=pin_memory,
        num_workers=0,
        # using multiproc here seems to cause one or the other problem.
        # prefetch_factor=2,
        # multiprocessing_context="spawn"
    )

    val_loaders: list[ValLoaderSpec] = []
    if config.train.run_validation:
        for val_source in resolve_validation_sources(config):
            val_source_name = normalize_validation_source_name(val_source.name)
            val_dataset_source = HFStreamingSourceConfig(
                dataset_name=val_source.source.dataset_name,
                dataset_config=val_source.source.dataset_config,
                split=val_source.source.split,
                text_field=val_source.source.text_field,
                shuffle_buffer_size=val_source.source.shuffle_buffer_size,
                max_rows=val_source.source.max_rows,
            )
            val_dataset = HFStreamingWindowDataset(
                dataset_cfg=val_dataset_source,
                split=val_source.source.split,
                tokenizer=tokenized.tokenizer,
                seq_len=config.train.seq_len,
                stride=config.train.stride,
                pad_id=special.pad_id,
                eos_id=special.eos_id,
                seed=config.run.seed,
                shuffle=False,
                bpb_mode=bpb_mode,
                approx_token_byte_lengths=approx_token_byte_lengths,
                reset_on_iter_close=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batching.loader_batch_size,
                pin_memory=pin_memory,
            )
            val_loaders.append(
                (
                    val_source_name,
                    val_loader,
                    int(val_source.max_eval_batches),
                )
            )

    print(
        "Streaming data loaders ready: "
        f"split={config.dataset.split} | "
        f"validation_sources={','.join(name for name, _, _ in val_loaders) or '<disabled>'} | "
        f"seq_len={config.train.seq_len} | stride={config.train.stride}"
    )
    return train_loader, val_loaders
