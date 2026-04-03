from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import (
    BaselineDecoderConfig,
    ExperimentConfig,
    HFPretrainedTokenizerConfig,
    HFTextDatasetConfig,
    LRSchedulerChainConfig,
    LRSchedulerStageConfig,
    LoggingConfig,
    OptimizerConfig,
    PreSplitConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from src.train import model_pipeline
from src.training.runtime import clear_runtime_state


SEED = 47

# Model params
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12

# Training params
LEARNING_RATE = 1e-3
LR_END_FACTOR = 0.1
SEQ_LEN = 1024
STRIDE = SEQ_LEN

# For the latency probe we keep micro-batch size close to the phase-1 baseline
# while using a valid integer accumulation plan.
MICRO_BATCH_SZ = 96
ACCUMULATION_STEPS = 5
EFFECTIVE_BATCH_SZ = MICRO_BATCH_SZ * ACCUMULATION_STEPS
TORCH_COMPILE_MEM_BUDGET = 0.75

# Logging / dataset
WANDB_PROJECT_NAME = "transformer-room-baseline"
WANDB_GROUP_NAME = "phase1/stage-1/latency-cadence-probe"
WANDB_RUN_NAME = (
    f"baseline-gpt-2-124M-B-{EFFECTIVE_BATCH_SZ}-MB-{MICRO_BATCH_SZ}-latency-probe"
)
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"

DEFAULT_MAX_TRAIN_STEPS = 16
DISABLED_CADENCE = 1_000


VARIANT_CADENCES: dict[str, dict[str, int]] = {
    # "control": {
    #     "log_every_n_steps": 4,
    #     "diagnostics_every_n_steps": DISABLED_CADENCE,
    #     "layer_grad_norms_every_n_steps": DISABLED_CADENCE,
    #     "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
    #     "attention_entropy_every_n_steps": DISABLED_CADENCE,
    # },
    # "diagnostics": {
    #     "log_every_n_steps": 4,
    #     "diagnostics_every_n_steps": 12,
    #     "layer_grad_norms_every_n_steps": DISABLED_CADENCE,
    #     "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
    #     "attention_entropy_every_n_steps": DISABLED_CADENCE,
    # },
    # "layer_grad": {
    #     "log_every_n_steps": 4,
    #     "diagnostics_every_n_steps": DISABLED_CADENCE,
    #     "layer_grad_norms_every_n_steps": 12,
    #     "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
    #     "attention_entropy_every_n_steps": DISABLED_CADENCE,
    # },
    "param_optim": {
        "log_every_n_steps": 4,
        "diagnostics_every_n_steps": DISABLED_CADENCE,
        "layer_grad_norms_every_n_steps": DISABLED_CADENCE,
        "parameter_optimizer_norms_every_n_steps": 12,
        "attention_entropy_every_n_steps": DISABLED_CADENCE,
    },
    "attention_entropy": {
        "log_every_n_steps": 4,
        "diagnostics_every_n_steps": DISABLED_CADENCE,
        "layer_grad_norms_every_n_steps": DISABLED_CADENCE,
        "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
        "attention_entropy_every_n_steps": 12,
    },
    "collision": {
        "log_every_n_steps": 4,
        "diagnostics_every_n_steps": 12,
        "layer_grad_norms_every_n_steps": 12,
        "parameter_optimizer_norms_every_n_steps": 12,
        "attention_entropy_every_n_steps": 12,
    },
}


def _build_base_config(max_train_steps: int) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(
            project_name=WANDB_PROJECT_NAME,
            group_name=WANDB_GROUP_NAME,
            run_name=WANDB_RUN_NAME,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            resume_from_checkpoint=False,
            persist_local_artifacts=True,
            checkpoint_every_n_steps=0,
            seed=SEED,
            use_torch_compile=True,
            activation_memory_budget=TORCH_COMPILE_MEM_BUDGET,
            compile_warmup_steps=3,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            split="train",
            text_field="text",
            shuffle_buffer_size=5_000,
        ),
        tokenizer=HFPretrainedTokenizerConfig(
            pretrained_name_or_path="gpt2",
            use_fast=True,
            bpb_mode="exact",
        ),
        model=BaselineDecoderConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            layers=N_LAYERS,
            dropout=0,
            norm_placement="pre",
            attention_impl="sdpa",
            enable_weight_tying=True,
        ),
        train=TrainConfig(
            epochs=None,
            optimizer=OptimizerConfig(
                name="adamw",
                learning_rate=LEARNING_RATE,
                weight_decay=0.1,
            ),
            lr_scheduler=LRSchedulerChainConfig(
                stages=[
                    LRSchedulerStageConfig(
                        type="cosine",
                        start_factor=1,
                        end_factor=LR_END_FACTOR,
                        steps=None,
                    )
                ]
            ),
            effective_batch_size=EFFECTIVE_BATCH_SZ,
            micro_batch_size=MICRO_BATCH_SZ,
            accumulation_steps=ACCUMULATION_STEPS,
            lr_scaling="sqrt",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_mode="streaming",
            max_steps=max_train_steps,
            run_validation=False,
        ),
        split=PreSplitConfig(),
        logging=LoggingConfig(
            provider="wandb",
            enable_artifact_io=True,
            wandb=WandbMetricsConfig(
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=False,
                enable_perplexity=True,
                enable_bits_per_byte=True,
                enable_step_time=True,
                enable_peak_memory=True,
                enable_update_to_weight_ratio=True,
                enable_global_param_norm=True,
                enable_global_grad_norm=True,
                enable_activation_norms=True,
                enable_layer_grad_norms=True,
                layer_grad_norm_stride=4,
                enable_attention_entropy=True,
                attention_entropy_head_cap=4,
                attention_entropy_token_cap=256,
                log_every_n_steps=4,
                diagnostics_every_n_steps=DISABLED_CADENCE,
                layer_grad_norms_every_n_steps=DISABLED_CADENCE,
                parameter_optimizer_norms_every_n_steps=DISABLED_CADENCE,
                attention_entropy_every_n_steps=DISABLED_CADENCE,
            ),
        ),
    )


def _build_variant_config(
    *,
    base_config: ExperimentConfig,
    variant_name: str,
) -> ExperimentConfig:
    cadence_overrides = VARIANT_CADENCES[variant_name]
    updated_run = replace(
        base_config.run,
        run_name=f"{base_config.run.run_name}-{variant_name}",
    )
    updated_wandb = replace(
        base_config.logging.wandb,
        **cadence_overrides,
    )
    updated_logging = replace(base_config.logging, wandb=updated_wandb)
    return replace(
        base_config,
        run=updated_run,
        logging=updated_logging,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short cadence-isolation experiments for latency debugging."
    )
    parser.add_argument(
        "--variant",
        choices=("all", *VARIANT_CADENCES.keys()),
        default="all",
        help="Variant to run. Defaults to all variants in sequence.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=DEFAULT_MAX_TRAIN_STEPS,
        help="Short run length used for each latency probe.",
    )
    return parser.parse_args(argv)


def _resolve_variant_names(selected_variant: str) -> list[str]:
    if selected_variant == "all":
        return list(VARIANT_CADENCES.keys())
    return [selected_variant]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    base_config = _build_base_config(args.max_train_steps)
    variant_names = _resolve_variant_names(args.variant)

    print(
        "Running latency cadence probe variants: "
        f"{', '.join(variant_names)} | max_train_steps={args.max_train_steps}"
    )

    for variant_name in variant_names:
        clear_runtime_state()
        variant_config = _build_variant_config(
            base_config=base_config,
            variant_name=variant_name,
        )
        cadence_overrides = VARIANT_CADENCES[variant_name]
        print(
            "\n=== Variant start ===\n"
            f"name={variant_name}\n"
            f"run_name={variant_config.run.run_name}\n"
            f"cadences={cadence_overrides}\n"
            f"effective_batch_size={variant_config.train.effective_batch_size}\n"
            f"micro_batch_size={variant_config.train.micro_batch_size}\n"
            f"accumulation_steps={variant_config.train.accumulation_steps}\n"
        )
        result = model_pipeline(variant_config)
        clear_runtime_state()
        print(
            "Variant complete | "
            f"name={variant_name} | "
            f"run_dir={result.run_artifact_dir} | "
            f"checkpoint={result.checkpoint_path} | "
            f"final_model={result.final_model_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
