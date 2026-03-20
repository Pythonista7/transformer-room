from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from ..core.config import ExperimentConfig, resolve_train_batching
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
) -> tuple[DataLoader, DataLoader]:
    batching = resolve_train_batching(config.train)
    special = tokenized.vocab.special

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
    val_loader = DataLoader(
        val_set,
        batch_size=batching.loader_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
