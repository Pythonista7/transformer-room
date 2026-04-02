import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import BaselineDecoderConfig, ExperimentConfig, HFPretrainedTokenizerConfig, HFTextDatasetConfig, HoldoutSplitConfig, LRSchedulerChainConfig, LRSchedulerStageConfig, LoggingConfig, OptimizerConfig, PreSplitConfig, RunConfig, TrainConfig, WandbMetricsConfig
from src.train import model_pipeline

"""
This is going to be a GPT-2 size model but with pre-norms and only a decay lr-schedule (thanks to pre-norm).
"""

WANDB_PROJECT_NAME = "transformer-room-baseline"
WANDB_GROUP_NAME = "phase1/stage-1"
WANDB_RUN_NAME = "baseline-gpt-2-124M"

SEED = 47



# Model Params
# Using the gpt-2 tokenizer we get around 50k vocab size
# And this model will have around 124M params ( see src/utils/simple_calc.py )
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12


# Training Params
# EPOCHS = 1 we will use MAX_TRAIN_STEPS instead since the dataset is huge.
EFFECTIVE_BATCH_SZ = 128
MICRO_BATCH_SZ = 32
TORCH_COMPILE_MEM_BUDGET = 0.75
LEARNING_RATE = 1e-3
LR_END_FACTOR = 0.1 

SEQ_LEN = 1024
STRIDE = SEQ_LEN


# Dataset

# We are using a 10B Token dataset, with a batch-sz of 64 & seq_len 1024 toks
# which comes to 65,536 per step
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"

# As for chinchilla recommeding 20 tokens per param
# so a 124M model approx should train on 2.5B tokens -> that suggests max_steps = 38k steps 

# While llama recommends a 1000:1 for token:param, which lands us around 125B which is insane! 
# Even for 100k steps, we shouldve seen around 6,553,600,000 ~ 6.5B tokens

MAX_TRAIN_STEPS = 1_000 # Will first test it for 1k before anything else

# On an A100, including torch.compile and final model upload, the train time for 1k steps was 34mins
# The GPU utilization could be better with bigger batches but this is the ball park range.
# So a chinchilla regime would take around 21.5-22hrs on the 40GB-A100


PHASE_1_STAGE_1_BAELINE_CONFIG = ExperimentConfig(
        run=RunConfig(
            project_name=WANDB_PROJECT_NAME,
            group_name=WANDB_GROUP_NAME,
            run_name=WANDB_RUN_NAME,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            resume_from_checkpoint=True,
            persist_local_artifacts=True,
            checkpoint_every_n_steps=30_000, # TODO: Make this a % value of total steps given we know the dataset size.
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
            bpb_mode="exact", # might incur overhead in streaming mode dataloader
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
            epochs=None, # We will use MAX_TRAIN_STEPS instead.
            optimizer=OptimizerConfig(
                name="adamw", 
                learning_rate=LEARNING_RATE,
                weight_decay= 0.1,
            ),
            lr_scheduler= LRSchedulerChainConfig(
                stages=[
                    LRSchedulerStageConfig(
                        type="cosine",
                        start_factor=1,
                        end_factor=LR_END_FACTOR,
                        steps=None, # this should automatically apply this for the entire run
                    )
                ]
            ),
            effective_batch_size=EFFECTIVE_BATCH_SZ,
            micro_batch_size= MICRO_BATCH_SZ,
            accumulation_steps= EFFECTIVE_BATCH_SZ/MICRO_BATCH_SZ,
            lr_scaling= "none" if EFFECTIVE_BATCH_SZ/MICRO_BATCH_SZ == 1 else "sqrt",
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_mode="streaming",
            max_steps= MAX_TRAIN_STEPS,
            run_validation=False
        ),
        split=PreSplitConfig(),
        logging=LoggingConfig(
            provider="wandb",
            enable_artifact_io=True,
            wandb=WandbMetricsConfig(
                # Step metrics
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=False, # No val set
                enable_perplexity=True,
                enable_bits_per_byte=True,
                enable_step_time=True,
                enable_peak_memory=True,
                
                # Diagnostics
                enable_update_to_weight_ratio=True, # Should tell me if lr is in right range.
                enable_global_param_norm=True,
                enable_global_grad_norm=True,
                enable_activation_norms=True,
                
                enable_layer_grad_norms=True,
                layer_grad_norm_stride=4, # 12 layers, stride 4 => start,mid,end metrics
                                
                # Attention entropy
                enable_attention_entropy=True,
                attention_entropy_head_cap= 4, # no of heads is 12, so sampling only 1/4th here for now.
                attention_entropy_token_cap= 256, # SEQ_LEN 1024 // 4
                
                # Cadances
                log_every_n_steps= 25, # for step metrics
                diagnostics_every_n_steps= 100, # default for diagnostics
                layer_grad_norms_every_n_steps= 500,
                parameter_optimizer_norms_every_n_steps=500, # for update/weight ratio freq
                attention_entropy_every_n_steps=250,

            ),
        ),
)


def main() -> int:
    result = model_pipeline(PHASE_1_STAGE_1_BAELINE_CONFIG)
    print(
        "Training complete | "
        f"run_dir={result.run_artifact_dir} | "
        f"checkpoint={result.checkpoint_path} | "
        f"final_model={result.final_model_path}"
    )
    return 0


if __name__ == "__main__":
    result = main()
    import os
    os._exit(result)
