from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..adapters.loggers import sanitize_wandb_name
from ..core.config import ExperimentConfig
from ..core.types import TokenizedCorpus


def find_latest_artifact_dir_with_checkpoint(
    models_root: Path,
    checkpoint_filename: str,
) -> Path | None:
    latest_dir: Path | None = None
    latest_mtime = float("-inf")

    if not models_root.exists():
        return None

    for entry in models_root.iterdir():
        if not entry.is_dir():
            continue
        checkpoint_path = entry / checkpoint_filename
        if not checkpoint_path.exists():
            continue

        checkpoint_mtime = checkpoint_path.stat().st_mtime
        if checkpoint_mtime > latest_mtime:
            latest_mtime = checkpoint_mtime
            latest_dir = entry

    return latest_dir


def prepare_run_artifact_paths(config: ExperimentConfig) -> dict[str, Path]:
    models_root = Path(config.run.artifacts_root).expanduser().resolve()
    models_root.mkdir(parents=True, exist_ok=True)

    if config.run.run_name:
        run_dir = models_root / config.run.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    elif config.run.resume_from_checkpoint:
        run_dir = find_latest_artifact_dir_with_checkpoint(
            models_root=models_root,
            checkpoint_filename=config.run.checkpoint_filename,
        )
        if run_dir is None:
            run_dir = models_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Resuming artifacts from: {run_dir}")
    else:
        run_dir = models_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_artifact_dir": run_dir,
        "checkpoint_path": run_dir / config.run.checkpoint_filename,
        "final_model_path": run_dir / config.run.final_model_filename,
        "model_diagram_path": run_dir / "baseline_model_architecture",
        "run_config_path": run_dir / "run_config.json",
        "inference_config_path": run_dir / "inference_config.json",
        "tokenizer_dir": run_dir / "tokenizer",
    }
    print(f"Run artifacts will be saved to: {run_dir}")
    return paths


def build_checkpoint_artifact_name(run_name: str) -> str:
    return f"{run_name}-checkpoint"


def build_final_model_artifact_name(run_name: str) -> str:
    return f"{run_name}-model"


def write_run_metadata(
    config: ExperimentConfig,
    tokenized: TokenizedCorpus,
    run_paths: dict[str, Path],
) -> None:
    run_paths["run_config_path"].write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    special = tokenized.vocab.special
    tokenizer_artifact_path: str | None = None
    if hasattr(tokenized.tokenizer, "save_pretrained"):
        tokenizer_dir = run_paths["tokenizer_dir"]
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenized.tokenizer.save_pretrained(str(tokenizer_dir))
        tokenizer_artifact_path = str(tokenizer_dir.resolve())

    vocab_path = None
    if hasattr(config.tokenizer, "vocab_path"):
        raw_vocab_path = getattr(config.tokenizer, "vocab_path", "")
        if raw_vocab_path:
            vocab_path = str(Path(raw_vocab_path).expanduser().resolve())

    inference_config = {
        "model_name": config.model.name,
        "tokenizer_name": config.tokenizer.name,
        "tokenizer_source": tokenized.tokenizer_source,
        "tokenizer_revision": tokenized.tokenizer_revision,
        "tokenizer_artifact_path": tokenizer_artifact_path,
        "tokenizer_added_special_tokens": dict(tokenized.added_special_tokens),
        "base_vocab_size": special.base_vocab_size,
        "num_special_tokens": special.num_special_tokens,
        "vocab_size": special.vocab_size,
        "pad_id": special.pad_id,
        "eos_id": special.eos_id,
        "unk_id": special.unk_id,
        "d_model": config.model.d_model,
        "n_heads": config.model.n_heads,
        "layers": config.model.layers,
        "attention_impl": config.model.attention_impl,
        "training_seq_len": config.train.seq_len,
        "tokenizer_vocab_path": vocab_path,
    }
    run_paths["inference_config_path"].write_text(
        json.dumps(inference_config, indent=2),
        encoding="utf-8",
    )


def clone_config_with_run_settings(
    config: ExperimentConfig,
    *,
    run_name: str,
    resume_from_checkpoint: bool,
) -> ExperimentConfig:
    return replace(
        config,
        run=replace(
            config.run,
            run_name=run_name,
            resume_from_checkpoint=resume_from_checkpoint,
        ),
    )


def stdin_is_interactive() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def resolve_wandb_lineage(
    config: ExperimentConfig,
    logger_adapter,
    *,
    input_fn: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> ExperimentConfig:
    if config.logging.provider != "wandb":
        return config
    if not config.logging.enable_artifact_io:
        return config

    has_remote_artifact = getattr(logger_adapter, "has_remote_artifact", None)
    if not callable(has_remote_artifact):
        return config

    base_run_name = config.run.run_name
    if base_run_name is None:
        return config

    def remote_checkpoint_exists(run_name: str) -> bool:
        return bool(
            has_remote_artifact(
                project_name=config.run.project_name,
                artifact_name=build_checkpoint_artifact_name(run_name),
                alias="latest",
            )
        )

    if not remote_checkpoint_exists(base_run_name):
        return config

    interactive_mode = stdin_is_interactive() if interactive is None else interactive
    if not interactive_mode:
        if config.run.resume_from_checkpoint:
            print(
                f"Remote checkpoint already exists for run_name={base_run_name}; "
                "resuming latest lineage."
            )
            return clone_config_with_run_settings(
                config,
                run_name=base_run_name,
                resume_from_checkpoint=True,
            )
        raise ValueError(
            f"Remote checkpoint already exists for run_name={base_run_name}. "
            "Re-run interactively to resume or provide a distinct run.run_name."
        )

    print(f"Remote checkpoint already exists for run_name={base_run_name}.")
    while True:
        print("1. Resume from the existing latest remote checkpoint.")
        print("2. Start a new lineage with a manual suffix.")
        choice = input_fn("Select 1 or 2: ").strip()
        if choice == "1":
            return clone_config_with_run_settings(
                config,
                run_name=base_run_name,
                resume_from_checkpoint=True,
            )
        if choice != "2":
            print("Please enter 1 or 2.")
            continue

        while True:
            raw_suffix = input_fn("Enter a new lineage suffix: ").strip()
            if not raw_suffix:
                print("Suffix must be non-empty.")
                continue

            suffix = sanitize_wandb_name(raw_suffix)
            candidate_run_name = f"{base_run_name}-{suffix}"
            if remote_checkpoint_exists(candidate_run_name):
                print(
                    f"Remote checkpoint already exists for run_name={candidate_run_name}. "
                    "Enter a different suffix."
                )
                continue

            return clone_config_with_run_settings(
                config,
                run_name=candidate_run_name,
                resume_from_checkpoint=False,
            )
