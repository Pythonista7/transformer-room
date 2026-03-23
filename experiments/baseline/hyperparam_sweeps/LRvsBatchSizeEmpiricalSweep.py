from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    HFTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from src.train import model_pipeline
from src.training import runtime as training_runtime
from src.training import wikitext as training_wikitext


def _format_lr_slug(learning_rate: float) -> str:
    return f"{learning_rate:g}".replace(".", "p")


def _build_sweep_group() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"p0-group-LRvsBSz-wikitext2_gpt2-sweep-{timestamp}"


def _build_run_name(learning_rate: float, effective_batch_size: int) -> str:
    return f"p0-group-LRvsBSz-wikitext2-gpt2-lr{_format_lr_slug(learning_rate)}-bs{effective_batch_size}"


def build_config(
    learning_rate: float,
    effective_batch_size: int,
    *,
    sweep_group: str | None = None,
) -> ExperimentConfig:
    project_root = PROJECT_ROOT
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-v1"
    vocab_path = (
        project_root
        / "src"
        / "vocabs"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )
    base_vocab_size = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        vocab_path=vocab_path,
    )

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            artifacts_root=str(project_root / "artifacts" / "models"),
            run_name=_build_run_name(learning_rate, effective_batch_size),
            group_name=sweep_group,
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=10000, # steps per run is 200 so we can just expect the final model here, no need chkpt for smaller runs.
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
            epochs=5, # we get around 200 steps per run with these settings, so 5 epochs should be enough to see some signal in the results while keeping runtime reasonable.
            optimizer=OptimizerConfig(
                learning_rate=learning_rate,
                weight_decay=0.0,
            ),
            effective_batch_size=effective_batch_size,
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
    effective_batch_sizes = (12, 20)
    sweep_group = _build_sweep_group()

    print(f"Starting sweep group: {sweep_group}")

    results = []
    for learning_rate in learning_rates:
        for effective_batch_size in effective_batch_sizes:
            print(
                "Starting run | "
                f"learning_rate={learning_rate} | "
                f"effective_batch_size={effective_batch_size}"
            )
            config = build_config(
                learning_rate=learning_rate,
                effective_batch_size=effective_batch_size,
                sweep_group=sweep_group,
            )
            result = model_pipeline(config)
            results.append(
                {
                    "sweep_group": sweep_group,
                    "learning_rate": learning_rate,
                    "effective_batch_size": effective_batch_size,
                    "run_dir": result.run_artifact_dir,
                    "checkpoint": result.checkpoint_path,
                    "final_model": result.final_model_path,
                    "checkpoint_artifact_ref": result.checkpoint_artifact_ref,
                    "final_model_artifact_ref": result.final_model_artifact_ref,
                    "train_loss": result.final_train_loss,
                    "val_loss": result.final_val_loss,
                    "val_ppl": result.final_val_perplexity,
                }
            )
            print(
                "Training complete | "
                f"learning_rate={learning_rate} | "
                f"effective_batch_size={effective_batch_size} | "
                f"run_dir={result.run_artifact_dir} | "
                f"checkpoint={result.checkpoint_path} | "
                f"final_model={result.final_model_path} | "
                f"checkpoint_artifact={result.checkpoint_artifact_ref} | "
                f"final_model_artifact={result.final_model_artifact_ref}"
            )

            del result
            training_runtime.clear_runtime_state()

    print("Sweep summary:")
    for summary in results:
        print(
            f"group={summary['sweep_group']} | "
            f"lr={summary['learning_rate']} | "
            f"effective_batch_size={summary['effective_batch_size']} | "
            f"val_loss={summary['val_loss']:.6f} | "
            f"val_ppl={summary['val_ppl']:.6f} | "
            f"run_dir={summary['run_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
