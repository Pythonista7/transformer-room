from __future__ import annotations

from pathlib import Path
from typing import Any

from src.components.tokenizers.bpe_tokenizer import BPETokenizer
from src.core.config import (
    BPETokenizerConfig,
    HFPretrainedTokenizerConfig,
    resolve_special_token_ids,
)
from src.core.registry import register_tokenizer_adapter
from src.core.types import SpecialTokenIds, TextCorpus, TokenizedCorpus, VocabInfo


EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def _resolve_auto_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Hugging Face tokenizer support requires the `transformers` package. "
            "Install it with `pip install transformers`."
        ) from exc
    return AutoTokenizer


def _normalize_hf_tokenizer_specials(tokenizer) -> dict[str, str]:
    added_special_tokens: dict[str, str] = {}
    special_updates: dict[str, str] = {}
    if tokenizer.pad_token_id is None:
        special_updates["pad_token"] = PAD_TOKEN
    if tokenizer.eos_token_id is None:
        special_updates["eos_token"] = EOS_TOKEN
    if special_updates:
        tokenizer.add_special_tokens(special_updates)
        added_special_tokens.update(special_updates)
    if tokenizer.pad_token_id is None:
        raise ValueError("Resolved HF tokenizer does not define a pad_token_id.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Resolved HF tokenizer does not define an eos_token_id.")
    return added_special_tokens


def _build_vocab_info_from_hf_tokenizer(tokenizer) -> VocabInfo:
    vocab_size = int(len(tokenizer))
    raw_token_to_id = dict(tokenizer.get_vocab())
    id_to_token: list[Any] = [None] * vocab_size
    for token, token_id in raw_token_to_id.items():
        if 0 <= int(token_id) < vocab_size:
            id_to_token[int(token_id)] = token
    for token_id, token in enumerate(id_to_token):
        if token is None:
            id_to_token[token_id] = tokenizer.convert_ids_to_tokens(token_id)
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}

    all_special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    special_info = SpecialTokenIds(
        vocab_size=vocab_size,
        eos_id=int(tokenizer.eos_token_id),
        pad_id=int(tokenizer.pad_token_id),
        unk_id=None if tokenizer.unk_token_id is None else int(tokenizer.unk_token_id),
        base_vocab_size=int(getattr(tokenizer, "vocab_size", vocab_size)),
        num_special_tokens=len(all_special_ids),
    )
    return VocabInfo(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        special=special_info,
    )


def _build_approx_token_byte_lengths(tokenizer, *, vocab_size: int) -> list[int]:
    special_ids = set(int(token_id) for token_id in tokenizer.all_special_ids)
    token_byte_lengths: list[int] = [0] * vocab_size
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_byte_lengths[token_id] = 0
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token_byte_lengths[token_id] = len(decoded.encode("utf-8"))
    return token_byte_lengths


def build_hf_pretrained_tokenizer_bundle(
    cfg: HFPretrainedTokenizerConfig,
    *,
    bpb_metrics_enabled: bool = False,
) -> TokenizedCorpus:
    AutoTokenizer = _resolve_auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_name_or_path,
        use_fast=cfg.use_fast,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
    )
    # Suppress the "sequence longer than model_max_length" warning — the data pipeline
    # tokenizes full documents and does its own windowing, so long documents are expected.
    tokenizer.model_max_length = int(1e30)
    added_special_tokens = _normalize_hf_tokenizer_specials(tokenizer)
    vocab_info = _build_vocab_info_from_hf_tokenizer(tokenizer)
    if bpb_metrics_enabled and cfg.bpb_mode == "exact":
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if not cfg.use_fast or backend_tokenizer is None:
            raise ValueError(
                "tokenizer.bpb_mode='exact' requires an HF fast tokenizer backend. "
                "Set tokenizer.use_fast=True and choose a tokenizer with offsets support."
            )

    approx_token_byte_lengths: list[int] | None = None
    if bpb_metrics_enabled and cfg.bpb_mode == "approx":
        approx_token_byte_lengths = _build_approx_token_byte_lengths(
            tokenizer,
            vocab_size=vocab_info.vocab_size,
        )
    print(
        "Loaded Hugging Face tokenizer: "
        f"{cfg.pretrained_name_or_path} | vocab_size={vocab_info.vocab_size}"
    )
    return TokenizedCorpus(
        token_stream=None,
        vocab=vocab_info,
        tokenizer=tokenizer,
        eos_inserted=0,
        unk_replacements=0,
        token_byte_lengths=approx_token_byte_lengths,
        tokenizer_source=cfg.pretrained_name_or_path,
        tokenizer_revision=cfg.revision,
        added_special_tokens=added_special_tokens,
    )


class BPETokenizerAdapter:
    def build(self, corpus: TextCorpus, cfg: BPETokenizerConfig) -> TokenizedCorpus:
        special = resolve_special_token_ids(cfg)
        vocab_path = Path(cfg.vocab_path).expanduser().resolve()
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer = BPETokenizer(
            corpus=corpus.full_text,
            max_vocab_count=special.base_vocab_size,
            path=str(vocab_path),
        )

        vocab = tokenizer.vocab
        tokenizer_vocab_size = len(vocab)
        if tokenizer_vocab_size != special.base_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size {tokenizer_vocab_size} does not match "
                f"base_vocab_size {special.base_vocab_size}."
            )

        id_to_token = list(vocab)
        id_to_token.append(EOS_TOKEN)
        id_to_token.append(PAD_TOKEN)
        if special.unk_id is not None:
            id_to_token.append(UNK_TOKEN)
        for idx in range(3, special.num_special_tokens):
            id_to_token.append(f"<SPECIAL_{idx}>")

        if len(id_to_token) != special.vocab_size:
            raise ValueError(
                f"Runtime vocab assembly mismatch: got {len(id_to_token)} tokens, "
                f"expected {special.vocab_size}."
            )

        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}

        token_stream: list[int] = []
        eos_inserted = 0
        unk_replacements = 0
        for segment in corpus.segments:
            encoded_segment: list[int] = []
            for token in tokenizer.encode(segment):
                token_id = token_to_id.get(token)
                if token_id is None:
                    if special.unk_id is None:
                        raise ValueError(
                            "Tokenizer produced unknown token but UNK is not configured. "
                            "Set tokenizer.num_special_tokens >= 3."
                        )
                    token_id = special.unk_id
                    unk_replacements += 1
                encoded_segment.append(token_id)

            token_stream.extend(encoded_segment)
            token_stream.append(special.eos_id)
            eos_inserted += 1

        if not token_stream:
            raise ValueError("Encoded token stream is empty.")

        print(
            f"Tokenizer vocab size: {tokenizer_vocab_size} | "
            f"Model vocab size: {special.vocab_size} (includes special tokens)"
        )

        vocab_info = VocabInfo(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special=special,
        )

        return TokenizedCorpus(
            token_stream=token_stream,
            vocab=vocab_info,
            tokenizer=tokenizer,
            eos_inserted=eos_inserted,
            unk_replacements=unk_replacements,
            tokenizer_source=str(vocab_path),
        )


class HFPretrainedTokenizerAdapter:
    def build(
        self,
        corpus: TextCorpus,
        cfg: HFPretrainedTokenizerConfig,
    ) -> TokenizedCorpus:
        tokenized = build_hf_pretrained_tokenizer_bundle(cfg)
        tokenizer = tokenized.tokenizer
        special = tokenized.vocab.special
        token_stream: list[int] = []
        eos_inserted = 0
        unk_replacements = 0
        for segment in corpus.segments:
            encoded_segment = tokenizer(
                segment,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            token_stream.extend(int(token_id) for token_id in encoded_segment)
            token_stream.append(int(special.eos_id))
            eos_inserted += 1
            if special.unk_id is not None:
                unk_replacements += sum(
                    1 for token_id in encoded_segment if int(token_id) == special.unk_id
                )

        if not token_stream:
            raise ValueError("Encoded token stream is empty.")

        return TokenizedCorpus(
            token_stream=token_stream,
            vocab=tokenized.vocab,
            tokenizer=tokenizer,
            eos_inserted=eos_inserted,
            unk_replacements=unk_replacements,
            tokenizer_source=tokenized.tokenizer_source,
            tokenizer_revision=tokenized.tokenizer_revision,
            added_special_tokens=dict(tokenized.added_special_tokens),
        )


def register_tokenizer_adapters() -> None:
    register_tokenizer_adapter("bpe", BPETokenizerAdapter())
    register_tokenizer_adapter("hf_pretrained", HFPretrainedTokenizerAdapter())
