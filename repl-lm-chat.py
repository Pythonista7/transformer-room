#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "baseline"

from baseline.model import BaselineModel
from baseline.bpe_tokenizer import BPETokenizer


QUIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
MODEL_CONFIG_KEYS = ("vocab_size", "d_model", "n_heads", "layers")
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple terminal REPL for BaselineModel .pt checkpoint/state_dict inference."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to .pt file. Can be a training checkpoint dict or plain model state_dict.",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config path. Required when the .pt file does not include config.",
    )
    parser.add_argument(
        "--tokenizer-vocab",
        help="Override tokenizer vocab file path from config (if needed).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Max tokens fed to model per step. Defaults to config.training_seq_len or 128.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generated tokens per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 disables top-k filter).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable multi-turn history and prompt each turn independently.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "chat"),
        default="plain",
        help=(
            "Prompt template style. 'plain' uses raw text completion "
            "(best for base LMs). 'chat' uses User/Assistant markers."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional instruction prefix added before the first turn.",
    )
    return parser.parse_args()


def resolve_existing_path(raw_path: str, model_path: Path) -> Path:
    candidate = Path(raw_path)
    search_order = [
        candidate,
        PROJECT_ROOT / candidate,
        BASELINE_DIR / candidate,
        model_path.parent / candidate,
    ]
    for path in search_order:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Could not find path '{raw_path}'. Tried: {', '.join(str(p) for p in search_order)}"
    )


def load_json_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config JSON must be an object: {config_path}")
    return data


def choose_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_flag == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def load_pt_file(model_path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(model_path, map_location=device)
    checkpoint_config: dict[str, Any] = {}

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        checkpoint_config = payload.get("config", {}) or {}
    elif isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        state_dict = payload
    else:
        raise ValueError(
            f"Unsupported .pt format at {model_path}. Expected checkpoint dict or plain state_dict."
        )

    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid model_state_dict in {model_path}")

    state_dict = dict(state_dict)
    state_dict = strip_prefix(state_dict, "_orig_mod.")
    state_dict = strip_prefix(state_dict, "module.")
    state_dict.pop("pos_encoding.pos_enc_cache", None)
    return state_dict, checkpoint_config


def resolve_config(
    checkpoint_config: dict[str, Any],
    user_config: dict[str, Any],
    tokenizer_vocab_override: str | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if checkpoint_config:
        merged.update(checkpoint_config)
    if user_config:
        merged.update(user_config)
    if tokenizer_vocab_override:
        merged["tokenizer_vocab_path"] = tokenizer_vocab_override
    return merged


def validate_model_config(config: dict[str, Any]) -> None:
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(
            f"Missing required model config keys: {missing}. "
            "Pass --config JSON or load from checkpoint with embedded config."
        )
    if "tokenizer_vocab_path" not in config:
        raise ValueError(
            "Missing tokenizer_vocab_path in config. Set it in --config or pass --tokenizer-vocab."
        )


def safe_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Config key '{key}' must be int-like, got {value!r}") from exc


def resolve_vocab_and_special_ids(
    config: dict[str, Any],
) -> tuple[int, int, int, int | None, int | None, int | None]:
    vocab_size = safe_int(config["vocab_size"], "vocab_size")
    base_vocab_size = safe_int(config.get("base_vocab_size", vocab_size), "base_vocab_size")
    num_special_tokens = safe_int(config.get("num_special_tokens", 0), "num_special_tokens")

    if num_special_tokens < 0:
        raise ValueError(f"num_special_tokens must be >= 0, got {num_special_tokens}")
    if vocab_size != base_vocab_size + num_special_tokens:
        raise ValueError(
            f"Expected vocab_size == base_vocab_size + num_special_tokens, got "
            f"{vocab_size} != {base_vocab_size} + {num_special_tokens}"
        )

    eos_id = base_vocab_size if num_special_tokens >= 1 else None
    pad_id = base_vocab_size + 1 if num_special_tokens >= 2 else None
    unk_id = base_vocab_size + 2 if num_special_tokens >= 3 else None
    return vocab_size, base_vocab_size, num_special_tokens, eos_id, pad_id, unk_id


def encode_text(
    text: str,
    tokenizer: BPETokenizer,
    token_to_id: dict[Any, int],
    unk_id: int | None = None,
) -> list[int]:
    tokenized = tokenizer.encode(text)
    ids: list[int] = []
    for token in tokenized:
        token_id = token_to_id.get(token)
        if token_id is None:
            if unk_id is None:
                raise ValueError(
                    f"Tokenizer produced unknown token {token!r}, but UNK is not configured. "
                    "Set num_special_tokens >= 3 and adjust vocab_size."
                )
            ids.append(unk_id)
            continue
        ids.append(token_id)
    return ids


def decode_ids(
    ids: list[int],
    id_to_token: list[Any],
    tokenizer: BPETokenizer,
    eos_id: int | None = None,
    pad_id: int | None = None,
    unk_id: int | None = None,
) -> str:
    pieces: list[str] = []
    byte_tokens: list[Any] = []

    def flush_bytes() -> None:
        if byte_tokens:
            pieces.append(tokenizer.decode(byte_tokens))
            byte_tokens.clear()

    for idx in ids:
        if eos_id is not None and idx == eos_id:
            continue
        if pad_id is not None and idx == pad_id:
            continue
        if unk_id is not None and idx == unk_id:
            flush_bytes()
            pieces.append(UNK_TOKEN)
            continue
        if idx < 0 or idx >= len(id_to_token):
            continue
        token = id_to_token[idx]
        if isinstance(token, (int, tuple)):
            byte_tokens.append(token)
            continue
        if token == UNK_TOKEN:
            flush_bytes()
            pieces.append(UNK_TOKEN)

    flush_bytes()
    return "".join(pieces)


def sample_next_id(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    blocked_ids: list[int] | None = None,
) -> int:
    if blocked_ids:
        logits = logits.clone()
        for blocked_id in blocked_ids:
            if 0 <= blocked_id < logits.numel():
                logits[blocked_id] = float("-inf")

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(indices[sampled].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: BaselineModel,
    tokenizer: BPETokenizer,
    token_to_id: dict[Any, int],
    id_to_token: list[Any],
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    context_window: int,
    device: torch.device,
    eos_id: int | None = None,
    pad_id: int | None = None,
    unk_id: int | None = None,
) -> str:
    prompt_ids = encode_text(prompt_text, tokenizer, token_to_id, unk_id=unk_id)
    if not prompt_ids:
        return ""

    token_ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context_ids = token_ids[-context_window:] if context_window > 0 else token_ids
        input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits = model(input_tensor)
        blocked_ids = [pad_id] if pad_id is not None else None
        next_id = sample_next_id(
            logits[0, -1, :],
            temperature=temperature,
            top_k=top_k,
            blocked_ids=blocked_ids,
        )
        if eos_id is not None and next_id == eos_id:
            break
        token_ids.append(next_id)

    generated_ids = token_ids[len(prompt_ids) :]
    return decode_ids(
        generated_ids,
        id_to_token,
        tokenizer,
        eos_id=eos_id,
        pad_id=pad_id,
        unk_id=unk_id,
    )


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[
    BaselineModel,
    BPETokenizer,
    dict[Any, int],
    list[Any],
    dict[str, Any],
    torch.device,
    int | None,
    int | None,
    int | None,
]:
    model_path = resolve_existing_path(args.model_path, model_path=PROJECT_ROOT / args.model_path)
    device = choose_device(args.device)
    state_dict, checkpoint_config = load_pt_file(model_path, device)

    user_config: dict[str, Any] = {}
    if args.config:
        config_path = resolve_existing_path(args.config, model_path=model_path)
        user_config = load_json_config(config_path)

    config = resolve_config(
        checkpoint_config=checkpoint_config,
        user_config=user_config,
        tokenizer_vocab_override=args.tokenizer_vocab,
    )
    validate_model_config(config)

    tokenizer_vocab_path = resolve_existing_path(config["tokenizer_vocab_path"], model_path=model_path)
    vocab_size, base_vocab_size, num_special_tokens, eos_id, pad_id, unk_id = resolve_vocab_and_special_ids(
        config
    )
    model = BaselineModel(
        vocab_size=vocab_size,
        d_model=safe_int(config["d_model"], "d_model"),
        n_heads=safe_int(config["n_heads"], "n_heads"),
        layers=safe_int(config["layers"], "layers"),
        pad_id=pad_id,
    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "State dict mismatch.\n"
            f"Missing: {missing_keys}\n"
            f"Unexpected: {unexpected_keys}"
        )

    model.to(device)
    model.eval()

    tokenizer = BPETokenizer(max_vocab_count=base_vocab_size, path=str(tokenizer_vocab_path))
    id_to_token = list(tokenizer.vocab)
    if num_special_tokens >= 1:
        id_to_token.append(EOS_TOKEN)
    if num_special_tokens >= 2:
        id_to_token.append(PAD_TOKEN)
    if num_special_tokens >= 3:
        id_to_token.append(UNK_TOKEN)
    for idx in range(3, num_special_tokens):
        id_to_token.append(f"<SPECIAL_{idx}>")

    if len(id_to_token) != vocab_size:
        raise RuntimeError(
            f"Runtime vocab assembly mismatch: got {len(id_to_token)} tokens, expected {vocab_size}"
        )

    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return model, tokenizer, token_to_id, id_to_token, config, device, eos_id, pad_id, unk_id


def run_repl(
    model: BaselineModel,
    tokenizer: BPETokenizer,
    token_to_id: dict[Any, int],
    id_to_token: list[Any],
    config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    eos_id: int | None,
    pad_id: int | None,
    unk_id: int | None,
) -> None:
    context_window = args.context_window
    if context_window is None:
        context_window = safe_int(config.get("training_seq_len", 128), "training_seq_len")
    if context_window <= 0:
        raise ValueError("--context-window must be > 0")

    print(f"Loaded model on {device}")
    print(f"Context window: {context_window}")
    print(f"Prompt format: {args.prompt_format}")
    print(f"Type a prompt and press enter. Type /quit to exit.\n")

    history = ""
    if args.system_prompt.strip():
        system_prompt = args.system_prompt.strip()
        if args.prompt_format == "chat":
            history = f"System: {system_prompt}\n"
        else:
            history = f"{system_prompt}\n"

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in QUIT_COMMANDS:
            break

        if args.prompt_format == "chat":
            if args.no_history:
                prompt = f"User: {user_text}\nAssistant:"
            else:
                prompt = f"{history}User: {user_text}\nAssistant:"
        else:
            if args.no_history:
                prompt = user_text
            else:
                prompt = f"{history}{user_text}\n"

        try:
            raw_reply = generate(
                model=model,
                tokenizer=tokenizer,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                prompt_text=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                context_window=context_window,
                device=device,
                eos_id=eos_id,
                pad_id=pad_id,
                unk_id=unk_id,
            )
        except Exception as exc:
            print(f"bot> [generation error] {exc}")
            continue

        if args.prompt_format == "chat":
            # Stop if model starts writing the next user turn marker.
            stop = raw_reply.find("\nUser:")
            assistant_raw = raw_reply if stop == -1 else raw_reply[:stop]
        else:
            assistant_raw = raw_reply
        assistant_text = assistant_raw.strip() or "(empty reply)"
        print(f"bot> {assistant_text}\n")

        if not args.no_history:
            history = f"{prompt}{assistant_raw}\n"


def main() -> int:
    args = parse_args()
    model, tokenizer, token_to_id, id_to_token, config, device, eos_id, pad_id, unk_id = load_model_and_tokenizer(args)
    run_repl(
        model=model,
        tokenizer=tokenizer,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        config=config,
        device=device,
        args=args,
        eos_id=eos_id,
        pad_id=pad_id,
        unk_id=unk_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
