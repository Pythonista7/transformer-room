import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import BaselineDecoderConfig, ExperimentConfig, HFPretrainedTokenizerConfig, HFStreamingSourceConfig, HFTextDatasetConfig, LRSchedulerChainConfig, LRSchedulerStageConfig, LoggingConfig, OptimizerConfig, PreSplitConfig, RunConfig, TrainConfig, ValSourceConfig, WandbMetricsConfig

from src.train import model_pipeline

"""
Scaled-down smoke test for ph1-baseline.py.
Run locally before committing to cloud GPUs to verify the full pipeline
(streaming data, multi-source val, metrics, local artifact IO) is healthy.

Knobs scaled down from ph1-baseline:
  d_model:       768  -> 128
  n_heads:       12   -> 4
  n_layers:      12   -> 2
  seq_len:       1024 -> 256
  batch_size:    512  -> 8  (micro=4, accum=2)
  max_steps:     1000 -> 20
  val cadence:   500  -> 10 (2 periodic passes + 1 final)
  val batches:   50   -> 5
  val max_rows:  5000 -> 500
  compile:       on   -> off  (skip compile overhead locally)
  logging:       wandb -> local
"""


SEED = 47

# Model
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2

# Training
EFFECTIVE_BATCH_SZ = 8
MICRO_BATCH_SZ = 4
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SZ // MICRO_BATCH_SZ
LEARNING_RATE = 1e-3
LR_END_FACTOR = 0.1
SEQ_LEN = 256
STRIDE = SEQ_LEN
MAX_TRAIN_STEPS = 20

# Dataset — same as production run
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"
VAL_DATASET_CONFIG = "CC-MAIN-2024-10"
VAL_MAX_ROWS = 500
VAL_MAX_EVAL_BATCHES = 5  # 5 × (4 seqs × 256 toks) = 5k val tokens per pass

RUN_NAME = f"smoke-gpt2-d{D_MODEL}-L{N_LAYERS}-B{EFFECTIVE_BATCH_SZ}"

SMOKE_CONFIG = ExperimentConfig(
    run=RunConfig(
        project_name="transformer-room-baseline",
        group_name="phase1/stage-1/smoke",
        run_name=RUN_NAME,
        artifacts_root=str(PROJECT_ROOT / "artifacts" / "smoke"),
        resume_from_checkpoint=False,
        persist_local_artifacts=True,
        checkpoint_every_n_steps=0,  # disabled — skip disk I/O for smoke
        seed=SEED,
        use_torch_compile=False,
        compile_warmup_steps=0,
    ),
    dataset=HFTextDatasetConfig(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        split="train",
        text_field="text",
        shuffle_buffer_size=500,
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
                shuffle_buffer_size=200,
                max_rows=VAL_MAX_ROWS,
            ),
            max_eval_batches=VAL_MAX_EVAL_BATCHES,
        )
    ],
    split=PreSplitConfig(),
    logging=LoggingConfig(
        provider="local",
        enable_artifact_io=True,
        wandb=WandbMetricsConfig(
            enable_train_loss_vs_tokens=True,
            enable_val_loss_vs_tokens=True,
            enable_perplexity=True,
            enable_bits_per_byte=True,
            enable_step_time=True,
            enable_peak_memory=True,
            enable_update_to_weight_ratio=True,
            enable_global_param_norm=True,
            enable_global_grad_norm=True,
            enable_activation_norms=True,
            enable_layer_grad_norms=True,
            layer_grad_norm_stride=1,
            enable_attention_entropy=True,
            attention_entropy_head_cap=2,
            attention_entropy_token_cap=64,
            log_every_n_steps=5,
            val_every_n_steps=10,
            diagnostics_every_n_steps=10,
            layer_grad_norms_every_n_steps=10,
            parameter_optimizer_norms_every_n_steps=10,
            attention_entropy_every_n_steps=10,
        ),
    ),
)


def main() -> int:
    result = model_pipeline(SMOKE_CONFIG)
    print(
        "Smoke run complete | "
        f"run_dir={result.run_artifact_dir} | "
        f"steps={result.global_step} | "
        f"train_loss={result.final_train_loss:.4f}"
    )
    for source, metrics in result.final_val_metrics_by_source.items():
        print(
            f"  val [{source}] "
            f"loss={metrics['val_loss']:.4f} | "
            f"ppl={metrics['val_perplexity']:.4f} | "
            f"bpb={metrics['val_bits_per_byte']:.4f}"
        )
    return 0


if __name__ == "__main__":
    result = main()
    import os
    os._exit(result)
