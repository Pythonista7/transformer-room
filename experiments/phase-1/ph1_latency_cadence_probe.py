from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import ExperimentConfig
from src.train import model_pipeline


BASELINE_SCRIPT_PATH = PROJECT_ROOT / "experiments" / "phase-1" / "ph1-baseline.py"
BASELINE_MODULE_NAME = "ph1_baseline_module"
DEFAULT_MAX_TRAIN_STEPS = 16
DISABLED_CADENCE = 1_000

assert(
    DISABLED_CADENCE > DEFAULT_MAX_TRAIN_STEPS
)

VARIANT_CADENCES: dict[str, dict[str, int]] = {
    "control": {
        "log_every_n_steps": 4,
        "diagnostics_every_n_steps": DISABLED_CADENCE,
        "layer_grad_norms_every_n_steps": DISABLED_CADENCE,
        "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
        "attention_entropy_every_n_steps": DISABLED_CADENCE,
    },
    "diagnostics": {
        "log_every_n_steps": 4,
        "diagnostics_every_n_steps": 12,
        "layer_grad_norms_every_n_steps": DISABLED_CADENCE,
        "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
        "attention_entropy_every_n_steps": DISABLED_CADENCE,
    },
    "layer_grad": {
        "log_every_n_steps": 4,
        "diagnostics_every_n_steps": DISABLED_CADENCE,
        "layer_grad_norms_every_n_steps": 12,
        "parameter_optimizer_norms_every_n_steps": DISABLED_CADENCE,
        "attention_entropy_every_n_steps": DISABLED_CADENCE,
    },
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


def _load_baseline_config() -> ExperimentConfig:
    spec = importlib.util.spec_from_file_location(
        BASELINE_MODULE_NAME,
        BASELINE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline experiment from {BASELINE_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "PHASE_1_STAGE_1_BAELINE_CONFIG", None)
    if config is None:
        raise RuntimeError(
            "Baseline experiment script does not expose "
            "`PHASE_1_STAGE_1_BAELINE_CONFIG`."
        )
    return config


def _build_variant_config(
    *,
    base_config: ExperimentConfig,
    variant_name: str,
    max_train_steps: int,
) -> ExperimentConfig:
    cadence_overrides = VARIANT_CADENCES[variant_name]
    run_name_base = base_config.run.run_name or "phase1-baseline"
    variant_run_name = f"{run_name_base}-latency-{variant_name}-{max_train_steps}steps"
    variant_group_name = (
        f"{base_config.run.group_name}/latency-cadence-probe"
        if base_config.run.group_name
        else "latency-cadence-probe"
    )

    updated_run = replace(
        base_config.run,
        run_name=variant_run_name,
        group_name=variant_group_name,
        resume_from_checkpoint=False,
        checkpoint_every_n_steps=0,
    )
    updated_train = replace(
        base_config.train,
        max_steps=max_train_steps,
        run_validation=False,
    )
    updated_wandb = replace(
        base_config.logging.wandb,
        **cadence_overrides,
    )
    updated_logging = replace(
        base_config.logging,
        wandb=updated_wandb,
    )
    return replace(
        base_config,
        run=updated_run,
        train=updated_train,
        logging=updated_logging,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run short cadence-isolation experiments derived from the phase-1 baseline."
        )
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
    base_config = _load_baseline_config()
    variant_names = _resolve_variant_names(args.variant)

    print(
        "Running latency cadence probe variants: "
        f"{', '.join(variant_names)} | max_train_steps={args.max_train_steps}"
    )

    for variant_name in variant_names:
        variant_config = _build_variant_config(
            base_config=base_config,
            variant_name=variant_name,
            max_train_steps=args.max_train_steps,
        )
        cadence_overrides = VARIANT_CADENCES[variant_name]
        print(
            "\n=== Variant start ===\n"
            f"name={variant_name}\n"
            f"run_name={variant_config.run.run_name}\n"
            f"cadences={cadence_overrides}\n"
            f"resume_from_checkpoint={variant_config.run.resume_from_checkpoint}\n"
            f"checkpoint_every_n_steps={variant_config.run.checkpoint_every_n_steps}\n"
        )
        result = model_pipeline(variant_config)
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
