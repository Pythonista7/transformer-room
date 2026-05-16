import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import BaselineDecoderConfig, ExperimentConfig, HFPretrainedTokenizerConfig, HFStreamingSourceConfig, HFTextDatasetConfig, HoldoutSplitConfig, LRSchedulerChainConfig, LRSchedulerStageConfig, LoggingConfig, OptimizerConfig, PreSplitConfig, RunConfig, TrainConfig, ValSourceConfig, WandbMetricsConfig
from src.train import model_pipeline

"""
Speed-run variant of ph1-baseline — 20 steps, cadences scaled accordingly.
Use this to validate the full e2e plumbing (data → train loop → val → metrics → artifact save → HF upload)
without waiting for a real training run.
"""


SEED = 47

D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12

EFFECTIVE_BATCH_SZ = 512
MICRO_BATCH_SZ = 64
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SZ // MICRO_BATCH_SZ
TORCH_COMPILE_MEM_BUDGET = 0.75
LEARNING_RATE = 1e-3
LR_END_FACTOR = 0.1

SEQ_LEN = 1024
STRIDE = SEQ_LEN

WANDB_PROJECT_NAME = "transformer-room-baseline"
WANDB_GROUP_NAME = "phase1/stage-1/speed-run"
WANDB_RUN_NAME = f"speed-run-B-{EFFECTIVE_BATCH_SZ}-MB-{MICRO_BATCH_SZ}"

DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"

VAL_DATASET_CONFIG = "CC-MAIN-2024-10"
VAL_MAX_ROWS = 500          # tiny — just enough to exercise the val path
VAL_MAX_EVAL_BATCHES = 5    # 5 × (64 × 1024) ≈ 330K val tokens

# compile_warmup_steps=3, so 20 steps gives 17 real compiled steps
MAX_TRAIN_STEPS = 20

# Log / eval every N steps — scaled so every cadence fires at least once in 20 steps
_LOG_EVERY       = 5
_VAL_EVERY       = 10
_DIAG_EVERY      = 5
_LAYER_GRAD_EVERY = 10
_PARAM_NORM_EVERY = 10
_ATTN_ENT_EVERY   = 10


PHASE_1_STAGE_1_SPEED_RUN_CONFIG = ExperimentConfig(
        run=RunConfig(
            project_name=WANDB_PROJECT_NAME,
            group_name=WANDB_GROUP_NAME,
            run_name=WANDB_RUN_NAME,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            resume_from_checkpoint=False,
            persist_local_artifacts=True,
            checkpoint_every_n_steps=MAX_TRAIN_STEPS,  # one checkpoint at the very end
            seed=SEED,
            use_torch_compile=True,
            activation_memory_budget=TORCH_COMPILE_MEM_BUDGET,
            compile_warmup_steps=3,
            hf_repo_id="Pythonista7/gpt2-124m-fineweb-baseline",  # exercises the upload path too
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
            lr_scaling="none" if ACCUMULATION_STEPS == 1 else "sqrt",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_mode="streaming",
            max_steps=MAX_TRAIN_STEPS,
            run_validation=True,
        ),
        val_sources=[
            ValSourceConfig(
                name="fineweb-cc-2024-10",
                source=HFStreamingSourceConfig(
                    dataset_name=DATASET_NAME,
                    dataset_config=VAL_DATASET_CONFIG,
                    split="train",
                    text_field="text",
                    shuffle_buffer_size=1_000,
                    max_rows=VAL_MAX_ROWS,
                ),
                max_eval_batches=VAL_MAX_EVAL_BATCHES,
            )
        ],
        split=PreSplitConfig(),
        logging=LoggingConfig(
            provider="wandb",
            enable_artifact_io=True,
            wandb=WandbMetricsConfig(
                # Step metrics
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=False,
                enable_perplexity=True,
                enable_bits_per_byte=True,
                enable_step_time=True,
                enable_peak_memory=True,

                # Diagnostics
                enable_update_to_weight_ratio=True,
                enable_global_param_norm=True,
                enable_global_grad_norm=True,
                enable_activation_norms=True,

                enable_layer_grad_norms=True,
                layer_grad_norm_stride=4,

                # Attention entropy
                enable_attention_entropy=True,
                attention_entropy_head_cap=4,
                attention_entropy_token_cap=256,

                # Cadences — scaled to fire at least once in 20 steps
                log_every_n_steps=_LOG_EVERY,
                val_every_n_steps=_VAL_EVERY,
                diagnostics_every_n_steps=_DIAG_EVERY,
                layer_grad_norms_every_n_steps=_LAYER_GRAD_EVERY,
                parameter_optimizer_norms_every_n_steps=_PARAM_NORM_EVERY,
                attention_entropy_every_n_steps=_ATTN_ENT_EVERY,
            ),
        ),
)


def main() -> int:
    result = model_pipeline(PHASE_1_STAGE_1_SPEED_RUN_CONFIG)
    print(
        "Speed run complete | "
        f"run_dir={result.run_artifact_dir} | "
        f"checkpoint={result.checkpoint_path} | "
        f"final_model={result.final_model_path}"
    )
    return 0


if __name__ == "__main__":
    result = main()
    import os
    os._exit(result)
