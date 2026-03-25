from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HFTextDatasetConfig,
    HoldoutSplitConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from src.train import model_pipeline
from src.training import wikitext as training_wikitext

PROJECT_NAME = "transformer-room-baseline"
RUN_GROUP = "pre-vs-post-layer-norm"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"

D_MODEL = 128
N_HEADS = 8
LAYERS = 20
EFFECTIVE_BATCH_SIZE = 64
SEQ_LEN = 128
STRIDE = 128
EPOCHS = 1
DATA_FRACTION = 1.0
TRAIN_FRACTION = 0.9
SEED = 42

LEARNING_RATE = 1e-3
LR_WARMUP_STEPS = 50
LR_WARMUP_START_FACTOR = 1e-4
LAYER_GRAD_STRIDE = 1
LAYER_GRAD_EVERY_N_STEPS = 10


def _build_variant_config(
    *,
    base_vocab_size: int,
    vocab_path: Path,
    run_name: str,
    norm_placement: str,
    warmup_enabled: bool,
) -> ExperimentConfig:
    warmup_steps = LR_WARMUP_STEPS if warmup_enabled else 0
    warmup_start_factor = LR_WARMUP_START_FACTOR if warmup_enabled else 0.0
    return ExperimentConfig(
        run=RunConfig(
            project_name=PROJECT_NAME,
            run_name=run_name,
            group_name=RUN_GROUP,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            persist_local_artifacts=False,
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=0,
            seed=SEED,
        ),
        dataset=HFTextDatasetConfig(
            dataset_name=DATASET_NAME,
            dataset_config=DATASET_CONFIG,
            split="train",
            text_field="text",
        ),
        tokenizer=BPETokenizerConfig(
            base_vocab_size=base_vocab_size,
            num_special_tokens=3,
            vocab_path=str(vocab_path),
        ),
        model=BaselineDecoderConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            layers=LAYERS,
            norm_placement=norm_placement,
        ),
        train=TrainConfig(
            epochs=EPOCHS,
            optimizer=OptimizerConfig(name="adam", learning_rate=LEARNING_RATE),
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            accumulation_steps=1,
            seq_len=SEQ_LEN,
            stride=STRIDE,
            data_fraction=DATA_FRACTION,
            lr_warmup_steps=warmup_steps,
            lr_warmup_start_factor=warmup_start_factor,
        ),
        split=HoldoutSplitConfig(
            train_fraction=TRAIN_FRACTION,
            seed=SEED,
            shuffle=False,
        ),
        logging=LoggingConfig("console",enable_artifact_io=False)
        # logging=LoggingConfig(
        #     provider="wandb",
        #     enable_artifact_io=False,
        #     wandb=WandbMetricsConfig(
        #         enable_train_loss_vs_tokens=True,
        #         enable_val_loss_vs_tokens=True,
        #         enable_perplexity=True,
        #         enable_bits_per_byte=True,
        #         enable_step_time=True,
        #         enable_peak_memory=True,
        #         enable_global_grad_norm=True,
        #         enable_layer_grad_norms=True,
        #         enable_global_param_norm=False,
        #         enable_layer_param_norms=False,
        #         enable_param_update_norm=False,
        #         enable_update_to_weight_ratio=False,
        #         enable_optimizer_state_norms=False,
        #         enable_activation_norms=False,
        #         enable_ln_grad_norms=False,
        #         enable_attention_entropy=False,
        #         watch_model=False,
        #         log_every_n_steps=10,
        #         diagnostics_every_n_steps=10,
        #         layer_grad_norm_stride=LAYER_GRAD_STRIDE,
        #         layer_grad_norms_every_n_steps=LAYER_GRAD_EVERY_N_STEPS,
        #         val_every_n_steps=250,
        #         attention_entropy_every_n_steps=250,
        #         attention_entropy_head_cap=1,
        #         attention_entropy_token_cap=SEQ_LEN,
        #     ),
        # ),
    )


def build_configs() -> dict[str, ExperimentConfig]:
    vocab_path = PROJECT_ROOT / "src" / "vocabs" / "wikitext2_v1_hf_vocab_bpe.txt"
    base_vocab_size = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        vocab_path=vocab_path,
    )

    return {
        "pre_no_warmup": _build_variant_config(
            base_vocab_size=base_vocab_size,
            vocab_path=vocab_path,
            run_name="pre_no_warmup",
            norm_placement="pre",
            warmup_enabled=False,
        ),
        "pre_warmup": _build_variant_config(
            base_vocab_size=base_vocab_size,
            vocab_path=vocab_path,
            run_name="pre_warmup",
            norm_placement="pre",
            warmup_enabled=True,
        ),
        "post_no_warmup": _build_variant_config(
            base_vocab_size=base_vocab_size,
            vocab_path=vocab_path,
            run_name="post_no_warmup",
            norm_placement="post",
            warmup_enabled=False,
        ),
        "post_warmup": _build_variant_config(
            base_vocab_size=base_vocab_size,
            vocab_path=vocab_path,
            run_name="post_warmup",
            norm_placement="post",
            warmup_enabled=True,
        ),
    }


def main() -> int:
    for variant_name, config in build_configs().items():
        print(f"Running variant: {variant_name}")
        result = model_pipeline(config)
        print(result.run_artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
