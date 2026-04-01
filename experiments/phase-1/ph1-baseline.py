
from pathlib import Path
import sys

from src.core.config import BPETokenizerConfig, BaselineDecoderConfig, ExperimentConfig, HFPretrainedTokenizerConfig, HFTextDatasetConfig, HoldoutSplitConfig, LoggingConfig, OptimizerConfig, PreSplitConfig, RunConfig, TrainConfig, WandbMetricsConfig
from src.training import wikitext as training_wikitext

"""
This is going to be a GPT-2 size model but with pre-norms and only a decay lr-schedule (thanks to pre-norm).
"""

WANDB_PROJECT_NAME = "transformer-room-baseline"
WANDB_GROUP_NAME = "phase1/stage-1"
WANDB_RUN_NAME = "baseline-gpt-2-124M"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = "1345"



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
LEARNING_RATE = 1e-3
FINAL_LEARNING_RATE = 3e-4
SEQ_LEN = 1024
STRIDE = SEQ_LEN


# Dataset

# We are using a 10B Token dataset, with a batch-sz of 64 & seq_len 1024 toks
# which comes to 65,536 per step
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"


# for 100k steps, we shouldve seen around 6,553,600,000 ~ 6.5B tokens
MAX_TRAIN_STEPS = 100_000



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
            activation_memory_budget=0.75,
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
        ),
        model=BaselineDecoderConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            layers=N_LAYERS,
            norm_placement="pre",
            attention_impl="sdpa",
            enable_weight_tying=True,
        ),
        train=TrainConfig(
            epochs=None, # We will use MAX_TRAIN_STEPS instead.
            optimizer=OptimizerConfig(
                name="adamw", 
                learning_rate=LEARNING_RATE,
                weight_decay= 0.01,
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
                enable_train_loss_vs_tokens=True,
                enable_val_loss_vs_tokens=True,
                enable_perplexity=True,
                enable_bits_per_byte=True,
                enable_step_time=True,
                enable_peak_memory=True,
                enable_global_grad_norm=False,
                enable_layer_grad_norms=False,
                # what is included in diagonistic_every_n_steps? 
            ),
        ),
)