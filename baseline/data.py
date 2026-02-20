from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from tokenizer import BPETokenizer

from config import Config

class CustomShakespeareDataset(Dataset):
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
        """Create start offsets for sliding windows and include one tail window when needed."""
        token_count = len(self.tokens)
        if token_count <= self.window:
            return [0]

        full_limit = token_count - self.window + 1
        starts = list(range(0, full_limit, self.stride))
        if not starts:
            starts = [0]

        # Add one short tail sample if stride leaves a remainder.
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


def resolve_special_token_ids(config: Config) -> tuple[int, int, int, int, int]:
    """Validate vocab sizing config and return `(base_vocab_size, num_special, vocab_size, eos_id, pad_id)`."""
    base_vocab_size = int(config.get("base_vocab_size", config["vocab_size"]))
    num_special_tokens = int(config.get("num_special_tokens", 0))
    vocab_size = int(config["vocab_size"])

    if base_vocab_size <= 0:
        raise ValueError(f"base_vocab_size must be > 0, got {base_vocab_size}")
    if num_special_tokens < 0:
        raise ValueError(f"num_special_tokens must be >= 0, got {num_special_tokens}")
    if vocab_size != base_vocab_size + num_special_tokens:
        raise ValueError(
            f"Expected vocab_size == base_vocab_size + num_special_tokens, got "
            f"{vocab_size} != {base_vocab_size} + {num_special_tokens}"
        )
    if num_special_tokens < 2:
        raise ValueError(
            "This training pipeline expects at least 2 special tokens (EOS and PAD)."
        )

    eos_id = base_vocab_size
    pad_id = base_vocab_size + 1
    return base_vocab_size, num_special_tokens, vocab_size, eos_id, pad_id


def truncate_stream_by_fraction_at_eos(
    token_stream: list[int], data_fraction: float, eos_id: int
) -> list[int]:
    """Truncate by `data_fraction` while ending on an EOS boundary when possible."""
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

    # If prefix has no EOS yet, grow to the first EOS to avoid cutting mid-segment.
    for idx in range(target_len, len(token_stream)):
        if token_stream[idx] == eos_id:
            return token_stream[: idx + 1]
    return token_stream


def get_data(
    config: Config, pin_memory: bool = False
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    base_vocab_size, _, _, eos_id, pad_id = resolve_special_token_ids(config)
    data_fraction = float(config.get("data_fraction", 0.1))
    training_stride = int(config.get("training_stride", 1))

    with open(config["dataset_path"]) as f:
        corpus = f.read()

    if corpus is None:
        raise Exception("Invalid dataset read!")

    tokenizer = BPETokenizer(
        corpus=corpus,
        max_vocab_count=base_vocab_size,
        path=config["tokenizer_vocab_path"],
    )

    vocab = tokenizer.vocab
    tokenizer_vocab_size = len(vocab)
    assert (
        tokenizer_vocab_size == base_vocab_size
    ), (
        f"Tokenizer vocab size {tokenizer_vocab_size} does not match "
        f"base_vocab_size {base_vocab_size}"
    )
    print(
        f"Tokenizer vocab size: {tokenizer_vocab_size} | "
        f"Model vocab size: {config['vocab_size']} (includes special tokens)"
    )

    token_to_id = {token: idx for idx, token in enumerate(vocab)}

    segments = corpus.split("\n\n")
    token_stream: list[int] = []
    eos_inserted = 0
    for segment in segments:
        encoded_segment = [token_to_id[token] for token in tokenizer.encode(segment)]
        token_stream.extend(encoded_segment)
        token_stream.append(eos_id)
        eos_inserted += 1

    if not token_stream:
        raise ValueError("Encoded token stream is empty.")

    # If you set data_fraction=1.0, this function is effectively a no-op.
    token_stream = truncate_stream_by_fraction_at_eos(
        token_stream=token_stream,
        data_fraction=data_fraction,
        eos_id=eos_id,
    )
    retained_eos = sum(1 for token in token_stream if token == eos_id)
    print(
        f"Encoded stream tokens: {len(token_stream):,} | "
        f"EOS inserted: {eos_inserted:,} | EOS retained: {retained_eos:,}"
    )

    if any(token < 0 or token >= config["vocab_size"] for token in token_stream):
        raise ValueError("Token stream contains token ids outside model vocab range.")

    dataset = CustomShakespeareDataset(
        token_stream,
        seq_len=config["training_seq_len"],
        stride=training_stride,
        pad_id=pad_id,
    )
    print(
        f"Dataset samples: {len(dataset):,} | "
        f"seq_len={config['training_seq_len']} | stride={training_stride} | pad_id={pad_id}"
    )

    train_set = torch.utils.data.Subset(dataset, range(0, int(0.9 * len(dataset))))
    val_set = torch.utils.data.Subset(
        dataset, range(int(0.9 * len(dataset)), len(dataset))
    )

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        # pin_memory=pin_memory,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        # pin_memory=pin_memory,
    )
    return dataloader, val_dataloader
