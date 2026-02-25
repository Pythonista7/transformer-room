from __future__ import annotations

from pathlib import Path

from baseline.bpe_tokenizer import BPETokenizer
from baseline.core.config import BPETokenizerConfig, resolve_special_token_ids
from baseline.core.registry import register_tokenizer_adapter
from baseline.core.types import TextCorpus, TokenizedCorpus, VocabInfo


EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


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
        )


register_tokenizer_adapter("bpe", BPETokenizerAdapter())
