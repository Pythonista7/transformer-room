from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable


def _resolve_hf_load_dataset():
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset support requires the `datasets` package. "
            "Install it with `pip install datasets`."
        ) from exc

    load_dataset = getattr(datasets_module, "load_dataset", None)
    if not callable(load_dataset):
        raise ImportError(
            "Resolved `datasets` module does not expose `load_dataset`. "
            "A local `datasets/` directory may be shadowing the Hugging Face package."
        )
    return load_dataset


def _iter_wikitext_tokens(text: str) -> Iterable[str]:
    for token in text.strip().split():
        if token:
            yield token


def ensure_wikitext_vocab_file(
    dataset_name: str,
    dataset_config: str,
    vocab_path: Path,
) -> int:
    if vocab_path.exists():
        size = 0
        with vocab_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    size += 1
        if size <= 0:
            raise ValueError(f"Existing vocab file is empty: {vocab_path}")
        print(f"Using existing Wikitext vocab file: {vocab_path} | size={size:,}")
        return size

    load_dataset = _resolve_hf_load_dataset()
    splits = ("train", "validation", "test")
    token_set: set[str] = {" ", "\n", "\t"}

    for split in splits:
        dataset = load_dataset(dataset_name, name=dataset_config, split=split)
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            token_set.update(_iter_wikitext_tokens(text))

    ordered_tokens = sorted(token_set)
    byte_tokens = [tuple(token.encode("utf-8")) for token in ordered_tokens]

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as handle:
        for token in byte_tokens:
            handle.write(f"{token}\n")

    print(
        f"Created Wikitext vocab file: {vocab_path} | "
        f"tokens={len(byte_tokens):,} | splits={','.join(splits)}"
    )
    return len(byte_tokens)
