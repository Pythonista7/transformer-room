
from pathlib import Path
import sys

from src.core.config import BPETokenizerConfig, BaselineDecoderConfig, ExperimentConfig, HFTextDatasetConfig, HoldoutSplitConfig, LoggingConfig, OptimizerConfig, RunConfig, TrainConfig, WandbMetricsConfig
from src.training import wikitext as training_wikitext


PROJECT_NAME = "transformer-room-Phase-1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = "1345"

# Dataset
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"

vocab_path = PROJECT_ROOT / "src" / "vocabs" / "wikitext2_v1_hf_vocab_bpe.txt"

BASE_VOCAB_SZ = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        vocab_path=vocab_path,
    )

# Model Params
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12

# Training Params
EPOCHS = 1
EFFECTIVE_BATCH_SZ = 64
MICRO_BATCH_SZ = 32
LEARNING_RATE = 3e-4
SEQ_LEN = 1024
STRIDE = SEQ_LEN
TRAINING_DATA_FRACTION = 0.9

PHASE_1_STAGE_1_BAELINE_CONFIG = ExperimentConfig(
        run=RunConfig(
            project_name=PROJECT_NAME,
            run_name="base",
            group_name="ph1-s1",
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            resume_from_checkpoint=True,
            persist_local_artifacts=True,
            checkpoint_every_n_steps=10_000, # TODO: Make this a % value of total steps given we know the dataset size.
            seed=SEED,
            use_torch_compile=True,
            activation_memory_budget=0.75,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            split="train",
            text_field="text",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=BASE_VOCAB_SZ,
            num_special_tokens=3,
            vocab_path=str(vocab_path),
        ),
        model=BaselineDecoderConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            layers=N_LAYERS,
            norm_placement="pre",
            attention_impl="sdpa"
        ),
        train=TrainConfig(
            epochs=EPOCHS,
            optimizer=OptimizerConfig(
                name="adamw", 
                learning_rate=LEARNING_RATE,
                weight_decay= 0.01
            ),
            effective_batch_size=EFFECTIVE_BATCH_SZ,
            micro_batch_size= MICRO_BATCH_SZ,
            accumulation_steps= EFFECTIVE_BATCH_SZ/MICRO_BATCH_SZ,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_fraction=TRAINING_DATA_FRACTION,
            run_validation=True
        ),
        split=HoldoutSplitConfig(
            train_fraction=0.9,
            seed=SEED,
            shuffle=False,
        ),
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
            ),
        ),
)