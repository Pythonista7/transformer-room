"""
Upload a transformer-room baseline checkpoint to HuggingFace Hub.

Uploads: model.safetensors, config.json, tokenizer/, README.md (model card).

Usage:
    python scripts/hf_upload.py \\
        --artifact-dir artifacts/models/<run_name> \\
        --repo-id Pythonista7/gpt2-124m-fineweb-baseline \\
        [--wandb-run-id 10wy8cqi] \\
        [--wandb-2b5-run-id <full-run-id>] \\
        [--token $HF_TOKEN] \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path


_DEFAULT_WANDB_ENTITY = "ashwin-ms-does-ai"
_DEFAULT_WANDB_PROJECT = "transformer-room-baseline"


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_artifact_metadata(artifact_dir: Path) -> tuple[dict, dict]:
    """Load inference_config.json and run_config.json from the artifact directory."""
    inference_path = artifact_dir / "inference_config.json"
    run_config_path = artifact_dir / "run_config.json"
    if not inference_path.exists():
        raise FileNotFoundError(f"inference_config.json not found in {artifact_dir}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {artifact_dir}")
    with open(inference_path) as f:
        inference_config = json.load(f)
    with open(run_config_path) as f:
        run_config = json.load(f)
    return inference_config, run_config


def build_hf_config_json(inference_config: dict, run_config: dict) -> dict:
    """Build an HF-style config.json from the two metadata files."""
    import transformers  # already in requirements
    model_cfg = run_config.get("model", {})
    return {
        "model_type": "baseline_decoder",
        "architectures": ["BaselineModel"],
        "vocab_size": inference_config["vocab_size"],
        "hidden_size": inference_config["d_model"],
        "num_hidden_layers": inference_config["layers"],
        "num_attention_heads": inference_config["n_heads"],
        "max_position_embeddings": inference_config["training_seq_len"],
        "attention_implementation": inference_config["attention_impl"],
        "norm_placement": model_cfg.get("norm_placement", "pre"),
        "enable_weight_tying": model_cfg.get("enable_weight_tying", False),
        "dropout": model_cfg.get("dropout", 0),
        "pad_token_id": inference_config["pad_id"],
        "eos_token_id": inference_config["eos_id"],
        "bos_token_id": inference_config["eos_id"],
        "torch_dtype": "float32",
        "transformers_version": transformers.__version__,
        "base_vocab_size": inference_config["base_vocab_size"],
        "tokenizer_source": inference_config["tokenizer_source"],
        "added_special_tokens": inference_config.get("tokenizer_added_special_tokens", {}),
    }


# ---------------------------------------------------------------------------
# Weights conversion
# ---------------------------------------------------------------------------

def count_parameters_from_state_dict(state_dict: dict) -> int:
    return sum(t.numel() for t in state_dict.values())


def convert_weights_to_safetensors(state_dict: dict, output_path: Path) -> None:
    """Write a plain state_dict as model.safetensors."""
    from safetensors.torch import save_file
    contiguous = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(contiguous, str(output_path), metadata={"format": "pt"})


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def _is_nan(v: object) -> bool:
    try:
        return math.isnan(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


def extract_local_metrics(artifact_dir: Path) -> dict[str, float]:
    """
    Scan metrics.jsonl and return the last non-NaN value for each metric key.
    Returns {} if the file does not exist.
    """
    metrics_path = artifact_dir / "metrics.jsonl"
    if not metrics_path.exists():
        warnings.warn(
            f"metrics.jsonl not found in {artifact_dir} — train/val metrics will be N/A in model card"
        )
        return {}
    merged: dict[str, float] = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in entry.get("metrics", {}).items():
                if not _is_nan(v):
                    merged[k] = float(v)
    return merged


def fetch_wandb_metrics(run_id: str, entity: str, project: str) -> dict[str, float]:
    """
    Pull final summary metrics from a W&B run.
    Returns {} on any failure (no credentials, run not found, etc.).
    """
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        result: dict[str, float] = {}
        for k, v in run.summary.items():
            if k.startswith("_"):
                continue
            key = k.split("/")[-1]  # strip section prefix like "Loss Curves/"
            if not _is_nan(v):
                try:
                    result[key] = float(v)
                except (TypeError, ValueError):
                    pass
        return result
    except Exception:
        return {}


def _load_checkpoint_state(artifact_dir: Path) -> dict:
    """Extract global_step and tokens_seen_train from the checkpoint if it exists."""
    ckpt_path = artifact_dir / "baseline_checkpoint.pt"
    if not ckpt_path.exists():
        return {}
    try:
        import torch
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        result = {}
        for k in ("global_step", "tokens_seen_train"):
            v = ckpt.get(k)
            if v is not None and not _is_nan(v):
                result[k] = v
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _fmt(v: object, spec: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        if _is_nan(v):
            return "N/A"
        return format(float(v), spec)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"


def build_model_card(
    inference_config: dict,
    run_config: dict,
    repo_id: str,
    param_count: int,
    metrics: dict[str, float],
    wandb_run_id: str | None,
    wandb_run_id_2b5: str | None,
    wandb_entity: str,
    wandb_project: str,
) -> str:
    train_cfg = run_config.get("train", {})
    opt_cfg = train_cfg.get("optimizer", {})
    model_cfg = run_config.get("model", {})

    seq_len = inference_config["training_seq_len"]
    effective_batch = train_cfg.get("effective_batch_size", "N/A")
    tokens_per_step = (
        f"{effective_batch * seq_len:,}" if isinstance(effective_batch, int) else "N/A"
    )
    micro_batch = train_cfg.get("micro_batch_size", "N/A")
    accum_steps = train_cfg.get("accumulation_steps", "N/A")
    lr = opt_cfg.get("learning_rate", "N/A")
    wd = opt_cfg.get("weight_decay", "N/A")

    # Find first val source from metric keys
    val_source: str | None = None
    for k in metrics:
        if "/val_loss" in k:
            val_source = k.split("/val_loss")[0]
            break

    train_loss = metrics.get("train_loss_step") or metrics.get("train_loss_epoch")
    train_bpb = metrics.get("train_bits_per_byte") or metrics.get("train_bits_per_byte_epoch")
    val_loss = metrics.get(f"{val_source}/val_loss") if val_source else None
    val_bpb = metrics.get(f"{val_source}/val_bits_per_byte") if val_source else None
    val_source_display = val_source or "fineweb-cc-2024-10"

    wandb_base = f"https://wandb.ai/{wandb_entity}/{wandb_project}/runs"
    smoke_link = (
        f"[`{wandb_run_id}`]({wandb_base}/{wandb_run_id})"
        if wandb_run_id else "N/A"
    )
    full_link = (
        f"[`{wandb_run_id_2b5}`]({wandb_base}/{wandb_run_id_2b5})"
        if wandb_run_id_2b5 else "TBD — training in progress"
    )

    param_str = f"~{param_count / 1e6:.1f}M" if param_count else "N/A"
    norm = model_cfg.get("norm_placement", "pre")
    weight_tying = model_cfg.get("enable_weight_tying", True)

    frontmatter = """\
---
language: en
license: apache-2.0
tags:
  - gpt2
  - causal-lm
  - pre-norm
  - weight-tying
  - fineweb
  - pytorch
datasets:
  - HuggingFaceFW/fineweb
pipeline_tag: text-generation
---
"""

    body = f"""\
# GPT-2 124M — FineWeb Baseline (transformer-room)

Decoder-only language model trained from scratch on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-10BT` subset).
GPT-2 scale ({param_str} parameters), pre-norm, SDPA attention, weight-tied embeddings.

Part of the **transformer-room** project — public training artifact for Phase 1 / Stage 1 baseline.

## Model Architecture

| Property | Value |
|---|---|
| Model type | `baseline_decoder` (custom, not a `transformers.PreTrainedModel`) |
| Parameters | {param_str} |
| Hidden size | {inference_config['d_model']} |
| Attention heads | {inference_config['n_heads']} |
| Layers | {inference_config['layers']} |
| Max sequence length | {seq_len} |
| Vocabulary size | {inference_config['vocab_size']} (GPT-2 base + `<PAD>` token) |
| Norm placement | `{norm}` (pre-norm — no LR warmup required) |
| Attention | Scaled Dot-Product Attention (PyTorch SDPA) |
| Weight tying | `{weight_tying}` (input embedding = output projection) |
| Positional encoding | Sinusoidal |
| FFN activation | ReLU |

## Training Details

| Property | Value |
|---|---|
| Dataset | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (`sample-10BT`) |
| Validation set | `CC-MAIN-2024-10` (held-out FineWeb CC dump, no train overlap) |
| Token budget | ~2.5B (Chinchilla-optimal for 124M: 20 tokens/param) |
| Optimizer | AdamW (`lr={lr}`, `weight_decay={wd}`) |
| LR schedule | Cosine decay to 0.1× (no warmup — pre-norm stable) |
| Effective batch size | {effective_batch} seqs × {seq_len} tokens = {tokens_per_step} tokens/step |
| Micro batch / accum | {micro_batch} seqs / {accum_steps} gradient accumulation steps |
| Hardware | H100 80GB |
| Precision | bf16 autocast (PyTorch AMP) |
| torch.compile | Yes (default mode, activation memory budget 0.75) |

## Results

| Metric | Smoke run (1k steps) | Full run (~2.5B tokens) |
|---|---|---|
| Train loss (final step) | {_fmt(train_loss)} | TBD |
| Train bits-per-byte | {_fmt(train_bpb)} | TBD |
| Val loss (`{val_source_display}`) | {_fmt(val_loss)} | TBD |
| Val bits-per-byte | {_fmt(val_bpb)} | TBD |
| HellaSwag (0-shot) | TBD | TBD |
| ARC-Easy (0-shot) | TBD | TBD |

_HellaSwag / ARC evaluations via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — planned in ASH-21._

## W&B Training Logs

| Run | Link |
|---|---|
| Smoke run (1k steps, B=512) | {smoke_link} |
| Full run (~2.5B tokens) | {full_link} |

## Loading the Weights

This model is **not** a `transformers.PreTrainedModel` and cannot be loaded with `AutoModelForCausalLM.from_pretrained()`.

```python
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Tokenizer (GPT-2 base + <PAD> token, vocab_size=50258)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Weights
state_dict = load_file("model.safetensors")  # or hf_hub_download("{repo_id}", "model.safetensors")
```

To reconstruct the model, instantiate `BaselineModel` from [`transformer-room/src/components/models/baseline_model.py`](https://github.com/ashwin-ms-does-ai/transformer-room) using the fields in `config.json`, then call `model.load_state_dict(state_dict)`.

## License

Apache 2.0
"""

    return frontmatter + "\n" + body


# ---------------------------------------------------------------------------
# Staging + upload
# ---------------------------------------------------------------------------

def prepare_staging_dir(
    artifact_dir: Path,
    staging_dir: Path,
    hf_config: dict,
    model_card_text: str,
) -> None:
    """Write config.json and README.md; copy tokenizer/ into staging_dir.

    model.safetensors is written directly into staging_dir by the caller.
    """
    (staging_dir / "config.json").write_text(
        json.dumps(hf_config, indent=2), encoding="utf-8"
    )
    (staging_dir / "README.md").write_text(model_card_text, encoding="utf-8")

    tokenizer_src = artifact_dir / "tokenizer"
    if tokenizer_src.is_dir():
        shutil.copytree(tokenizer_src, staging_dir / "tokenizer")
    else:
        warnings.warn(
            f"tokenizer/ directory not found at {tokenizer_src} — skipping tokenizer upload"
        )


def upload_to_hub(
    staging_dir: Path,
    repo_id: str,
    token: str | None,
    commit_message: str,
    dry_run: bool,
    private: bool,
) -> None:
    if dry_run:
        print("\n[dry-run] Files that would be uploaded:")
        for p in sorted(staging_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(staging_dir)
                size_kb = p.stat().st_size / 1024
                print(f"  {rel}  ({size_kb:.1f} KB)")
        print(f"\n[dry-run] Target repo : {repo_id}")
        print(f"[dry-run] Commit msg  : {commit_message!r}")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    print(f"Uploading to https://huggingface.co/{repo_id} ...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging_dir),
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Done: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload a transformer-room baseline checkpoint to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--artifact-dir", required=True, type=Path,
        help="Run artifact directory (contains inference_config.json, run_config.json, "
             "baseline_model.pt, tokenizer/)",
    )
    p.add_argument(
        "--repo-id", required=True,
        help="HF Hub repo id, e.g. Pythonista7/gpt2-124m-fineweb-baseline",
    )
    p.add_argument("--wandb-run-id", default=None, help="W&B run ID for primary/smoke run")
    p.add_argument("--wandb-2b5-run-id", default=None, help="W&B run ID for the 2.5B full run")
    p.add_argument("--wandb-entity", default=_DEFAULT_WANDB_ENTITY)
    p.add_argument("--wandb-project", default=_DEFAULT_WANDB_PROJECT)
    p.add_argument(
        "--token", default=None,
        help="HF API token (falls back to HF_TOKEN env var, then huggingface-cli login cache)",
    )
    p.add_argument("--commit-message", default=None)
    p.add_argument("--private", action="store_true", help="Create the HF repo as private")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without making any network calls",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch

    args = parse_args(argv)
    artifact_dir: Path = args.artifact_dir.expanduser().resolve()

    if not artifact_dir.is_dir():
        print(f"Error: artifact-dir does not exist: {artifact_dir}", file=sys.stderr)
        return 1

    model_pt = artifact_dir / "baseline_model.pt"
    checkpoint_pt = artifact_dir / "baseline_checkpoint.pt"
    if not model_pt.exists() and not checkpoint_pt.exists():
        print(
            f"Error: neither baseline_model.pt nor baseline_checkpoint.pt found in {artifact_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Loading metadata from {artifact_dir} ...")
    try:
        inference_config, run_config = load_artifact_metadata(artifact_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    hf_config = build_hf_config_json(inference_config, run_config)

    print("Loading weights ...")
    if model_pt.exists():
        state_dict = torch.load(model_pt, weights_only=True, map_location="cpu")
    else:
        print(
            f"  baseline_model.pt not found — extracting state dict from baseline_checkpoint.pt"
        )
        ckpt = torch.load(checkpoint_pt, weights_only=False, map_location="cpu")
        state_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k != "pos_encoding.pos_enc_cache"
        }
    param_count = count_parameters_from_state_dict(state_dict)
    print(f"  Parameters: {param_count:,} (~{param_count / 1e6:.1f}M)")

    print("Extracting metrics ...")
    metrics = extract_local_metrics(artifact_dir)
    if args.wandb_run_id:
        wb = fetch_wandb_metrics(args.wandb_run_id, args.wandb_entity, args.wandb_project)
        if wb:
            print(f"  Fetched {len(wb)} metrics from W&B run {args.wandb_run_id}")
            metrics.update(wb)
        else:
            print(f"  W&B metrics unavailable for {args.wandb_run_id} — using local only")
    ckpt_state = _load_checkpoint_state(artifact_dir)
    if "global_step" in ckpt_state:
        print(
            f"  global_step={ckpt_state['global_step']}  "
            f"tokens_seen_train={ckpt_state.get('tokens_seen_train', 'N/A')}"
        )

    commit_message = args.commit_message or f"Upload {artifact_dir.name} baseline checkpoint"

    with tempfile.TemporaryDirectory(prefix="hf_upload_") as tmpdir:
        staging = Path(tmpdir)

        print("Converting baseline_model.pt → model.safetensors ...")
        convert_weights_to_safetensors(state_dict, staging / "model.safetensors")
        del state_dict

        model_card = build_model_card(
            inference_config=inference_config,
            run_config=run_config,
            repo_id=args.repo_id,
            param_count=param_count,
            metrics=metrics,
            wandb_run_id=args.wandb_run_id,
            wandb_run_id_2b5=args.wandb_2b5_run_id,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )

        prepare_staging_dir(
            artifact_dir=artifact_dir,
            staging_dir=staging,
            hf_config=hf_config,
            model_card_text=model_card,
        )

        token = args.token or os.environ.get("HF_TOKEN")
        upload_to_hub(
            staging_dir=staging,
            repo_id=args.repo_id,
            token=token,
            commit_message=commit_message,
            dry_run=args.dry_run,
            private=args.private,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
