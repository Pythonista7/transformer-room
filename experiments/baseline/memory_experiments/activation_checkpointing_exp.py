from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ACEveryNDecoderConfig,
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HFTextDatasetConfig,
    HoldoutSplitConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    SACDecoderConfig,
    TrainConfig,
    WandbMetricsConfig,
)
from src.train import model_pipeline
from src.training import runtime as training_runtime
from src.training import wikitext as training_wikitext


@dataclass(frozen=True, slots=True)
class VariantSpec:
    key: str
    model_cfg: BaselineDecoderConfig | ACEveryNDecoderConfig | SACDecoderConfig
    use_torch_compile: bool
    activation_memory_budget: float | None = None


def build_variant_specs() -> list[VariantSpec]:
    return [
        # VariantSpec(
        #     key="baseline_no_compile",
        #     model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
        #     use_torch_compile=False,
        # ),
        # VariantSpec(
        #     key="ac_every_5_no_compile",
        #     model_cfg=ACEveryNDecoderConfig(
        #         d_model=768,
        #         n_heads=8,
        #         layers=12,
        #         dropout=0.1,
        #         checkpoint_every_n_layers=5,
        #     ),
        #     use_torch_compile=False,
        # ),
        # VariantSpec(
        #     key="ac_every_2_no_compile",
        #     model_cfg=ACEveryNDecoderConfig(
        #         d_model=768,
        #         n_heads=8,
        #         layers=12,
        #         dropout=0.1,
        #         checkpoint_every_n_layers=2,
        #     ),
        #     use_torch_compile=False,
        # ),
        # VariantSpec(
        #     key="ac_every_1_no_compile",
        #     model_cfg=ACEveryNDecoderConfig(
        #         d_model=768,
        #         n_heads=8,
        #         layers=12,
        #         dropout=0.1,
        #         checkpoint_every_n_layers=1,
        #     ),
        #     use_torch_compile=False,
        # ),
        # VariantSpec(
        #     key="sac_no_compile",
        #     model_cfg=SACDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
        #     use_torch_compile=False,
        # ),
        VariantSpec(
            key="baseline_compile",
            model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
            use_torch_compile=True,
        ),
        VariantSpec(
            key="baseline_compile_budget_0p2",
            model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
            use_torch_compile=True,
            activation_memory_budget=0.2,
        ),
        VariantSpec(
            key="baseline_compile_budget_0p4",
            model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
            use_torch_compile=True,
            activation_memory_budget=0.4,
        ),
        VariantSpec(
            key="baseline_compile_budget_0p5",
            model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
            use_torch_compile=True,
            activation_memory_budget=0.5,
        ),
        VariantSpec(
            key="baseline_compile_budget_0p6",
            model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
            use_torch_compile=True,
            activation_memory_budget=0.6,
        ),
        VariantSpec(
            key="baseline_compile_budget_0p8",
            model_cfg=BaselineDecoderConfig(d_model=768, n_heads=8, layers=12, dropout=0.1),
            use_torch_compile=True,
            activation_memory_budget=0.8,
        ),
    ]


def build_config(
    *,
    variant: VariantSpec,
    sweep_group: str,
    base_vocab_size: int,
) -> ExperimentConfig:
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-v1"
    vocab_path = (
        PROJECT_ROOT
        / "src"
        / "vocabs"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )
    run_name = f"{sweep_group}-{variant.key}"

    return ExperimentConfig(
        run=RunConfig(
            project_name="transformer-room-baseline",
            run_name=run_name,
            group_name=sweep_group,
            artifacts_root=str(PROJECT_ROOT / "artifacts" / "models"),
            resume_from_checkpoint=False,
            checkpoint_every_n_steps=0,
            use_torch_compile=variant.use_torch_compile,
            torch_compile_mode="default",
            torch_compile_fullgraph=False,
            torch_compile_dynamic=False,
            activation_memory_budget=variant.activation_memory_budget,
            compile_warmup_steps=10 if variant.use_torch_compile else 0,
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
        model=variant.model_cfg,
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(
                name="adam",
                learning_rate=3e-5,
                weight_decay=0.0,
            ),
            effective_batch_size=20,
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
                enable_global_grad_norm=False,
                enable_global_param_norm=False,
                enable_layer_param_norms=False,
                enable_param_update_norm=False,
                enable_update_to_weight_ratio=False,
                enable_optimizer_state_norms=False,
                enable_activation_norms=False,
                enable_ln_grad_norms=False,
                enable_attention_entropy=False,
                watch_model=False,
                log_every_n_steps=1,
                diagnostics_every_n_steps=1000,
                parameter_optimizer_norms_every_n_steps=1000,
                val_every_n_steps=0,
                attention_entropy_every_n_steps=200,
                attention_entropy_head_cap=1,
                attention_entropy_token_cap=64,
            ),
        ),
    )


def _build_sweep_group() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"torch-memory-budget-exp-wikitext2-{timestamp}"


def main() -> int:
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-v1"
    vocab_path = (
        PROJECT_ROOT
        / "src"
        / "vocabs"
        / "wikitext2_v1_hf_vocab_bpe.txt"
    )
    base_vocab_size = training_wikitext.ensure_wikitext_vocab_file(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        vocab_path=vocab_path,
    )
    variants = build_variant_specs()
    training_runtime.preflight_dynamo_activation_memory_budget_api(
        [variant.activation_memory_budget for variant in variants]
    )

    sweep_group = _build_sweep_group()
    print(f"Starting activation memory experiment group: {sweep_group}")

    results: list[dict[str, object]] = []
    for variant in variants:
        print(
            "Starting variant | "
            f"key={variant.key} | "
            f"compile={variant.use_torch_compile} | "
            f"budget={variant.activation_memory_budget}"
        )
        cfg = build_config(
            variant=variant,
            sweep_group=sweep_group,
            base_vocab_size=base_vocab_size,
        )
        # Important! We need to reset the torch compiler state between runs to ensure that memory budgets are properly applied and that we don't reuse compiled graphs across runs in a way that would invalidate the experiment. This is especially important for the variants that use torch.compile with different activation memory budgets, as we want to make sure each run is compiled with the correct budget and that there is no cross-run contamination of compiled graphs.
        torch.compiler.reset()
        
        run_result = model_pipeline(cfg)
        results.append(
            {
                "variant": variant.key,
                "compile": variant.use_torch_compile,
                "budget": variant.activation_memory_budget,
                "run_dir": run_result.run_artifact_dir,
                "train_loss": run_result.final_train_loss,
                "val_loss": run_result.final_val_loss,
                "val_ppl": run_result.final_val_perplexity,
            }
        )
        print(
            "Variant complete | "
            f"variant={variant.key} | "
            f"run_dir={run_result.run_artifact_dir} | "
            f"train_loss={run_result.final_train_loss:.6f} | "
            f"val_loss={run_result.final_val_loss:.6f} | "
            f"val_ppl={run_result.final_val_perplexity:.6f}"
        )

        del run_result
        training_runtime.clear_runtime_state()

    print("Experiment summary:")
    for summary in results:
        print(
            f"group={sweep_group} | "
            f"variant={summary['variant']} | "
            f"compile={summary['compile']} | "
            f"budget={summary['budget']} | "
            f"train_loss={summary['train_loss']:.6f} | "
            f"val_loss={summary['val_loss']:.6f} | "
            f"val_ppl={summary['val_ppl']:.6f} | "
            f"run_dir={summary['run_dir']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
