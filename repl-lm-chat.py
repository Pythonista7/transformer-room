#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "src"
DEFAULT_WANDB_PROJECT = "transformer-room-baseline"

from src.adapters.tokenizers import build_hf_pretrained_tokenizer_bundle
from src.components.models.baseline_model import BaselineModel
from src.components.tokenizers.bpe_tokenizer import BPETokenizer
from src.config import HFPretrainedTokenizerConfig


QUIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
MODEL_CONFIG_KEYS = ("vocab_size", "d_model", "n_heads", "layers")
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple terminal REPL for BaselineModel inference from local files or W&B."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model-path",
        help="Path to .pt file. Can be a training checkpoint dict or plain model state_dict.",
    )
    source_group.add_argument(
        "--wandb-run",
        help=(
            "W&B run selector. Accepts entity/project/run_id, bare run_id, "
            "or a wandb.ai run URL."
        ),
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config path. Required when the local .pt file does not include config.",
    )
    parser.add_argument(
        "--tokenizer-vocab",
        help="Override tokenizer vocab file path from local config (if needed).",
    )
    parser.add_argument(
        "--wandb-entity",
        help="Optional W&B entity. Used when --wandb-run is a bare run ID.",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=(
            "Optional W&B project. Used when --wandb-run is a bare run ID. "
            f"Defaults to {DEFAULT_WANDB_PROJECT!r}."
        ),
    )
    parser.add_argument(
        "--wandb-cache-dir",
        help=(
            "Optional cache directory for downloaded W&B artifacts. "
            "Defaults to artifacts/wandb-repl/<entity>/<project>/<run_id>/."
        ),
    )
    parser.add_argument(
        "--wandb-model-alias",
        default="final",
        help="Preferred W&B model artifact alias. Falls back to 'latest' if not found.",
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
    args = parser.parse_args(argv)

    if args.wandb_run and (args.config or args.tokenizer_vocab):
        parser.error(
            "--config and --tokenizer-vocab are only supported with --model-path."
        )

    return args


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


def safe_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Config key '{key}' must be float-like, got {value!r}") from exc


def resolve_model_seq_len(config: dict[str, Any]) -> int:
    candidates: list[tuple[str, Any]] = [
        ("seq_len", config.get("seq_len")),
        ("training_seq_len", config.get("training_seq_len")),
    ]

    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        candidates.append(("model.seq_len", model_cfg.get("seq_len")))

    train_cfg = config.get("train")
    if isinstance(train_cfg, dict):
        candidates.append(("train.seq_len", train_cfg.get("seq_len")))

    for key, value in candidates:
        if value is not None:
            return safe_int(value, key)

    raise ValueError(
        "Missing model sequence length in config. Expected one of: "
        "seq_len, training_seq_len, model.seq_len, or train.seq_len."
    )


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


def get_wandb_api():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B-backed REPL loading requires the `wandb` package."
        ) from exc
    return wandb.Api()


def wandb_artifact_not_found(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "not found" in message or "404" in message or "commerror" in name


def resolve_wandb_run_reference(
    raw_run: str,
    *,
    api,
    explicit_entity: str | None,
    explicit_project: str | None,
) -> tuple[str, str, str]:
    raw = raw_run.strip()
    if not raw:
        raise ValueError("--wandb-run must be non-empty.")

    if "://" in raw:
        parsed = urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[-2] == "runs":
            return parts[-4], parts[-3], parts[-1]
        raise ValueError(
            f"Unsupported W&B run URL format: {raw}. Expected .../<entity>/<project>/runs/<run_id>."
        )

    parts = [part for part in raw.split("/") if part]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 1:
        entity = explicit_entity or getattr(api, "default_entity", None)
        project = explicit_project or DEFAULT_WANDB_PROJECT
        if not entity:
            raise ValueError(
                "Could not determine W&B entity for bare run ID. "
                "Pass --wandb-entity or configure the local W&B client."
            )
        return str(entity), str(project), parts[0]
    raise ValueError(
        f"Unsupported --wandb-run value: {raw!r}. "
        "Use entity/project/run_id, a bare run_id, or a wandb.ai run URL."
    )


def _single_downloaded_file(download_dir: Path, preferred_filename: str | None = None) -> Path:
    candidates: list[Path] = []
    if preferred_filename:
        candidates.append(download_dir / preferred_filename)
        candidates.extend(
            path for path in download_dir.rglob(preferred_filename) if path.is_file()
        )
    else:
        candidates.extend(path for path in download_dir.rglob("*") if path.is_file())

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    downloaded_files = [path for path in download_dir.rglob("*") if path.is_file()]
    if len(downloaded_files) == 1:
        return downloaded_files[0].resolve()
    raise FileNotFoundError(f"No artifact file found in {download_dir}")


def download_wandb_artifact(
    api,
    *,
    entity: str,
    project: str,
    artifact_name: str,
    alias: str,
    cache_dir: Path,
    preferred_filename: str | None = None,
) -> Path:
    artifact_ref = f"{entity}/{project}/{artifact_name}:{alias}"
    artifact = api.artifact(artifact_ref)
    artifact_cache_dir = cache_dir / f"{artifact_name}__{alias}"
    artifact_cache_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path(artifact.download(root=str(artifact_cache_dir)))
    return _single_downloaded_file(download_dir, preferred_filename=preferred_filename)


def resolve_wandb_artifacts(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    api = get_wandb_api()
    entity, project, run_id = resolve_wandb_run_reference(
        args.wandb_run,
        api=api,
        explicit_entity=args.wandb_entity,
        explicit_project=args.wandb_project,
    )
    run = api.run(f"{entity}/{project}/{run_id}")
    run_name = str(getattr(run, "name", "")).strip()
    if not run_name:
        raise ValueError(
            f"W&B run {entity}/{project}/{run_id} does not have a stable run.name."
        )

    cache_dir = (
        Path(args.wandb_cache_dir).expanduser().resolve()
        if args.wandb_cache_dir
        else (PROJECT_ROOT / "artifacts" / "wandb-repl" / entity / project / run_id).resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_artifact_name = f"{run_name}-model"
    model_aliases = [args.wandb_model_alias]
    if args.wandb_model_alias != "latest":
        model_aliases.append("latest")

    model_path: Path | None = None
    last_error: Exception | None = None
    for alias in model_aliases:
        try:
            model_path = download_wandb_artifact(
                api,
                entity=entity,
                project=project,
                artifact_name=model_artifact_name,
                alias=alias,
                cache_dir=cache_dir,
            )
            break
        except Exception as exc:
            if wandb_artifact_not_found(exc):
                last_error = exc
                continue
            raise
    if model_path is None:
        raise FileNotFoundError(
            f"Could not find W&B model artifact '{model_artifact_name}' "
            f"with aliases {model_aliases} for run {entity}/{project}/{run_id}."
        ) from last_error

    try:
        run_config_path = download_wandb_artifact(
            api,
            entity=entity,
            project=project,
            artifact_name=f"{run_name}-run-config",
            alias="latest",
            cache_dir=cache_dir,
            preferred_filename="run_config.json",
        )
        inference_config_path = download_wandb_artifact(
            api,
            entity=entity,
            project=project,
            artifact_name=f"{run_name}-inference-config",
            alias="latest",
            cache_dir=cache_dir,
            preferred_filename="inference_config.json",
        )
    except Exception as exc:
        if wandb_artifact_not_found(exc):
            raise FileNotFoundError(
                f"Missing required metadata artifact for run {entity}/{project}/{run_id}. "
                "Expected both '<run_name>-run-config:latest' and "
                "'<run_name>-inference-config:latest'."
            ) from exc
        raise

    return model_path, load_json_config(run_config_path), load_json_config(inference_config_path)


def build_runtime_config_from_run_metadata(
    run_config: dict[str, Any],
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = run_config.get("model")
    tokenizer_cfg = run_config.get("tokenizer")
    train_cfg = run_config.get("train")
    if not isinstance(model_cfg, dict):
        raise ValueError("run_config.json is missing a 'model' object.")
    if not isinstance(tokenizer_cfg, dict):
        raise ValueError("run_config.json is missing a 'tokenizer' object.")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError("run_config.json has invalid 'train' object.")

    model_name = str(model_cfg.get("name", inference_config.get("model_name", ""))).strip()
    if model_name and model_name != "baseline_decoder":
        raise ValueError(
            f"W&B REPL currently supports baseline_decoder runs only, got {model_name!r}."
        )
    tokenizer_name = str(
        tokenizer_cfg.get("name", inference_config.get("tokenizer_name", ""))
    ).strip()
    if tokenizer_name and tokenizer_name != "hf_pretrained":
        raise ValueError(
            f"W&B REPL currently supports hf_pretrained tokenizers only, got {tokenizer_name!r}."
        )

    runtime_config = dict(inference_config)
    seq_len_value = (
        model_cfg.get("seq_len")
        if model_cfg.get("seq_len") is not None
        else (
            train_cfg.get("seq_len")
            if isinstance(train_cfg, dict) and train_cfg.get("seq_len") is not None
            else inference_config.get("training_seq_len")
        )
    )
    runtime_config.update(
        {
            "dropout": safe_float(model_cfg.get("dropout", 0.1), "dropout"),
            "norm_placement": str(model_cfg.get("norm_placement", "post")),
            "attention_impl": str(
                model_cfg.get("attention_impl", inference_config.get("attention_impl", "basic"))
            ),
            "enable_weight_tying": bool(model_cfg.get("enable_weight_tying", False)),
            "tokenizer_source": str(
                tokenizer_cfg.get(
                    "pretrained_name_or_path", inference_config.get("tokenizer_source", "")
                )
            ),
            "tokenizer_revision": tokenizer_cfg.get(
                "revision", inference_config.get("tokenizer_revision")
            ),
            "tokenizer_use_fast": bool(tokenizer_cfg.get("use_fast", True)),
            "tokenizer_trust_remote_code": bool(
                tokenizer_cfg.get("trust_remote_code", False)
            ),
            "seq_len": safe_int(seq_len_value, "seq_len"),
            "model_name": model_name or "baseline_decoder",
            "tokenizer_name": tokenizer_name or "hf_pretrained",
        }
    )
    return runtime_config


def validate_hf_tokenizer_bundle(
    runtime_config: dict[str, Any],
    tokenized_bundle,
) -> None:
    special = tokenized_bundle.vocab.special

    expected_pairs = {
        "vocab_size": int(special.vocab_size),
        "base_vocab_size": int(
            special.base_vocab_size if special.base_vocab_size is not None else special.vocab_size
        ),
        "num_special_tokens": int(special.num_special_tokens),
        "eos_id": int(special.eos_id),
        "pad_id": int(special.pad_id),
    }
    if special.unk_id is not None:
        expected_pairs["unk_id"] = int(special.unk_id)

    for key, resolved_value in expected_pairs.items():
        raw_expected = runtime_config.get(key)
        if raw_expected is None:
            continue
        expected_value = safe_int(raw_expected, key)
        if expected_value != resolved_value:
            raise ValueError(
                f"Hugging Face tokenizer metadata mismatch for {key}: "
                f"config={expected_value}, tokenizer={resolved_value}"
            )


def load_model_and_tokenizer_from_wandb(args: argparse.Namespace) -> tuple[
    BaselineModel,
    Any,
    dict[Any, int],
    list[Any],
    dict[str, Any],
    torch.device,
    int | None,
    int | None,
    int | None,
]:
    model_path, run_config, inference_config = resolve_wandb_artifacts(args)
    device = choose_device(args.device)
    state_dict, _ = load_pt_file(model_path, device)
    runtime_config = build_runtime_config_from_run_metadata(run_config, inference_config)

    tokenized_bundle = build_hf_pretrained_tokenizer_bundle(
        HFPretrainedTokenizerConfig(
            pretrained_name_or_path=str(runtime_config["tokenizer_source"]),
            use_fast=bool(runtime_config.get("tokenizer_use_fast", True)),
            revision=runtime_config.get("tokenizer_revision"),
            trust_remote_code=bool(runtime_config.get("tokenizer_trust_remote_code", False)),
        )
    )
    validate_hf_tokenizer_bundle(runtime_config, tokenized_bundle)

    special = tokenized_bundle.vocab.special
    model = BaselineModel(
        vocab_size=safe_int(runtime_config["vocab_size"], "vocab_size"),
        d_model=safe_int(runtime_config["d_model"], "d_model"),
        n_heads=safe_int(runtime_config["n_heads"], "n_heads"),
        layers=safe_int(runtime_config["layers"], "layers"),
        seq_len=resolve_model_seq_len(runtime_config),
        dropout=safe_float(runtime_config.get("dropout", 0.1), "dropout"),
        attention_impl=str(runtime_config.get("attention_impl", "basic")),
        norm_placement=str(runtime_config.get("norm_placement", "post")),
        enable_weight_tying=bool(runtime_config.get("enable_weight_tying", False)),
        pad_id=special.pad_id,
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
    return (
        model,
        tokenized_bundle.tokenizer,
        tokenized_bundle.vocab.token_to_id,
        tokenized_bundle.vocab.id_to_token,
        runtime_config,
        device,
        special.eos_id,
        special.pad_id,
        special.unk_id,
    )


def encode_text(
    text: str,
    tokenizer: Any,
    token_to_id: dict[Any, int],
    unk_id: int | None = None,
) -> list[int]:
    if isinstance(tokenizer, BPETokenizer):
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

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return [int(token_id) for token_id in encoded]


def decode_ids(
    ids: list[int],
    id_to_token: list[Any],
    tokenizer: Any,
    eos_id: int | None = None,
    pad_id: int | None = None,
    unk_id: int | None = None,
) -> str:
    if not isinstance(tokenizer, BPETokenizer):
        filtered_ids = [
            idx
            for idx in ids
            if not (
                (eos_id is not None and idx == eos_id)
                or (pad_id is not None and idx == pad_id)
            )
        ]
        if not filtered_ids:
            return ""
        try:
            return str(
                tokenizer.decode(
                    filtered_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(tokenizer.decode(filtered_ids))

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
    tokenizer: Any,
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


def load_model_and_tokenizer_from_local(args: argparse.Namespace) -> tuple[
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
        seq_len=resolve_model_seq_len(config),
        attention_impl=str(config.get("attention_impl", "basic")),
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


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[
    BaselineModel,
    Any,
    dict[Any, int],
    list[Any],
    dict[str, Any],
    torch.device,
    int | None,
    int | None,
    int | None,
]:
    if args.wandb_run:
        return load_model_and_tokenizer_from_wandb(args)
    return load_model_and_tokenizer_from_local(args)


def run_repl(
    model: BaselineModel,
    tokenizer: Any,
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
            stop = raw_reply.find("\nUser:")
            assistant_raw = raw_reply if stop == -1 else raw_reply[:stop]
        else:
            assistant_raw = raw_reply
        assistant_text = assistant_raw.strip() or "(empty reply)"
        print(f"bot> {assistant_text}\n")

        if not args.no_history:
            history = f"{prompt}{assistant_raw}\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
