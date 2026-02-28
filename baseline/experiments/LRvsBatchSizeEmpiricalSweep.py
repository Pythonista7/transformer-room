from __future__ import annotations
import gc
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    HFTextDatasetConfig,
    LoggingConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from baseline.train import model_pipeline


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


def _format_lr_slug(learning_rate: float) -> str:
    return f"{learning_rate:g}".replace(".", "p")


def _build_sweep_group() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"p0-LRvsBSz-wikitext2_gpt2-sweep-{timestamp}"


def build_config(
    learning_rate: float,
    batch_size: int,
    *,
    sweep_group: str | None = None,
) -> ExperimentConfig:
    project_root = PROJECT_ROOT
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-v1"
    run_prefix = sweep_group or "p0-LRvsBSz-wikitext2_gpt2"
    vocab_path = (
        project_root
        / "baseline"
        / "tokenizers"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )
    base_vocab_size = ensure_wikitext_vocab_file(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        vocab_path=vocab_path,
    )

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            artifacts_root=str(project_root / "baseline" / "models"),
            run_name=(
                f"{run_prefix}-"
                f"lr{_format_lr_slug(learning_rate)}-bs{batch_size}"
            ),
            group_name=sweep_group,
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=200, # steps per run is 200 so we can chk_pt every 200 or 1 per run.
            use_torch_compile=False,
            torch_compile_mode="default",
            torch_compile_fullgraph=False,
            torch_compile_dynamic=False,
            seed=42,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="train",
            text_field="text",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=base_vocab_size,
            num_special_tokens=3,
            vocab_path=str(vocab_path),
        ),
        model=BaselineDecoderConfig(
            d_model=768,
            n_heads=8,
            layers=12,
            dropout=0.1,
        ),
        train=TrainConfig(
            epochs=3,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seq_len=1024,
            stride=1024,
            data_fraction=1.0,
        ),
        split=HoldoutSplitConfig(
            train_fraction=0.9,
            seed=42,
            shuffle=False,
        ),
        logging=LoggingConfig(
            provider="wandb",
            wandb=WandbMetricsConfig(
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=True,
                enable_perplexity=True,
                enable_step_time=True,
                enable_peak_memory=True,
                enable_global_grad_norm=True,
                enable_activation_norms=True,
                enable_ln_grad_norms=True,
                enable_attention_entropy=True,
                watch_model=True,
                log_every_n_steps=10,
                diagnostics_every_n_steps=50,
                val_every_n_steps=250,
                attention_entropy_every_n_steps=200,
                attention_entropy_head_cap=2,
                attention_entropy_token_cap=128,
            ),
        ),
    )


def main() -> int:
    learning_rates = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    batch_sizes = (20,) # TODO: revisit once we have grad-acc in place.
    sweep_group = _build_sweep_group()

    print(f"Starting sweep group: {sweep_group}")

    results = []
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            print(
                "Starting run | "
                f"learning_rate={learning_rate} | "
                f"batch_size={batch_size}"
            )
            config = build_config(
                learning_rate=learning_rate,
                batch_size=batch_size,
                sweep_group=sweep_group,
            )
            result = model_pipeline(config)
            results.append(
                {
                    "sweep_group": sweep_group,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "run_dir": result.run_artifact_dir,
                    "checkpoint": result.checkpoint_path,
                    "final_model": result.final_model_path,
                    "train_loss": result.final_train_loss,
                    "val_loss": result.final_val_loss,
                    "val_ppl": result.final_val_perplexity,
                }
            )
            print(
                "Training complete | "
                f"learning_rate={learning_rate} | "
                f"batch_size={batch_size} | "
                f"run_dir={result.run_artifact_dir} | "
                f"checkpoint={result.checkpoint_path} | "
                f"final_model={result.final_model_path}"
            )

            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                empty_cache = getattr(torch.mps, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()

    print("Sweep summary:")
    for summary in results:
        print(
            f"group={summary['sweep_group']} | "
            f"lr={summary['learning_rate']} | "
            f"batch_size={summary['batch_size']} | "
            f"val_loss={summary['val_loss']:.6f} | "
            f"val_ppl={summary['val_ppl']:.6f} | "
            f"run_dir={summary['run_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
