from __future__ import annotations

import gc
import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch

from ..adapters.loggers import sanitize_wandb_name
from ..core.config import ExperimentConfig

DEFAULT_TORCH_TRACE_ROOT = Path("/tmp/tracedir")
DEFAULT_TLPARSE_OUTPUT_DIR = Path("tl_out")


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("TORCH IS NOW USING MPS DEVICE")
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_enable_bf16_autocast(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker):
        return False
    return bool(checker())


# TODO: @Ash remove this if possible or atleast make it optional only when specific timing metrics are essential.
def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_autocast_context(device: torch.device, use_bf16: bool):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_uncompiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _set_activation_memory_budget_if_configured(config: ExperimentConfig) -> None:
    budget = config.run.activation_memory_budget
    if budget is None:
        return

    dynamo_module = getattr(torch, "_dynamo", None)
    if dynamo_module is None:
        raise RuntimeError(
            "run.activation_memory_budget is set, but torch._dynamo is unavailable "
            "on this PyTorch build."
        )
    _ = dynamo_module

    functorch_module = getattr(torch, "_functorch", None)
    functorch_config = getattr(functorch_module, "config", None)
    if functorch_config is None or not hasattr(
        functorch_config,
        "activation_memory_budget",
    ):
        raise RuntimeError(
            "run.activation_memory_budget is set, but "
            "torch._functorch.config.activation_memory_budget is unavailable on this "
            "PyTorch build."
        )
    functorch_config.activation_memory_budget = float(budget)


def resolve_torch_compile_trace_dir(config: ExperimentConfig, logger) -> Path | None:
    if not config.run.torch_compile_trace:
        return None

    trace_root = DEFAULT_TORCH_TRACE_ROOT
    if config.logging.provider != "wandb":
        return trace_root

    get_run_id = getattr(logger, "get_run_id", None)
    run_id = get_run_id() if callable(get_run_id) else None
    if run_id is None or not str(run_id).strip():
        raise RuntimeError(
            "run.torch_compile_trace requires an active W&B run ID before compile."
        )
    return trace_root / sanitize_wandb_name(str(run_id))


def _collect_trace_files(trace_dir: Path) -> list[Path]:
    return sorted(candidate for candidate in trace_dir.rglob("*") if candidate.is_file())


def _find_latest_trace_log(trace_dir: Path) -> Path | None:
    log_files = sorted(
        (candidate for candidate in trace_dir.rglob("*.log") if candidate.is_file()),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not log_files:
        return None
    return log_files[-1]


def finalize_torch_compile_trace(
    *,
    config: ExperimentConfig,
    logger,
    trace_dir: Path | None,
) -> None:
    if trace_dir is None:
        return

    if not trace_dir.exists():
        raise RuntimeError(
            f"run.torch_compile_trace is enabled, but no trace directory was created at {trace_dir}."
        )

    if config.logging.provider == "wandb":
        trace_files = _collect_trace_files(trace_dir)
        if not trace_files:
            raise RuntimeError(
                f"run.torch_compile_trace is enabled, but no trace files were found under {trace_dir}."
            )
        upload_run_files = getattr(logger, "upload_run_files", None)
        if not callable(upload_run_files):
            raise RuntimeError(
                "Active logger does not support uploading torch trace run files."
            )
        upload_run_files(
            [str(path) for path in trace_files],
            base_path=str(trace_dir.parent),
        )
        return

    tlparse_path = shutil.which("tlparse")
    if tlparse_path is None:
        raise RuntimeError(
            "run.torch_compile_trace requires `tlparse` to be installed when logging.provider='console'."
        )

    trace_log = _find_latest_trace_log(trace_dir)
    if trace_log is None:
        raise RuntimeError(
            f"run.torch_compile_trace is enabled, but no torch trace log was found under {trace_dir}."
        )

    try:
        subprocess.run(
            [
                tlparse_path,
                str(trace_log),
                "-o",
                str(DEFAULT_TLPARSE_OUTPUT_DIR),
                "--overwrite",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"tlparse failed for torch trace log {trace_log} with exit code {exc.returncode}."
        ) from exc


def maybe_compile_model(
    model: torch.nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    *,
    trace_dir: Path | None = None,
) -> tuple[torch.nn.Module, bool, str]:
    if not config.run.use_torch_compile:
        return model, False, "disabled"
    if not hasattr(torch, "compile"):
        return model, False, "torch.compile unavailable"
    if device.type == "cpu":
        return model, False, f"skipped on {device.type}"

    _set_activation_memory_budget_if_configured(config)

    try:
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_TRACE"] = str(trace_dir)
        compiled_model = torch.compile(
            model,
            mode=config.run.torch_compile_mode,
            fullgraph=bool(config.run.torch_compile_fullgraph),
            dynamic=bool(config.run.torch_compile_dynamic),
        )
        return compiled_model, True, "enabled"
    except Exception as exc:  # pragma: no cover - backend-specific failure paths.
        return model, False, f"failed: {exc}"


def classify_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
    )


def clear_runtime_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        reset_peak_memory_stats = getattr(torch.cuda, "reset_peak_memory_stats", None)
        if callable(reset_peak_memory_stats):
            try:
                reset_peak_memory_stats()
            except Exception:
                pass
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    reset_compiler = getattr(getattr(torch, "compiler", None), "reset", None)
    if callable(reset_compiler):
        reset_compiler()


def preflight_dynamo_activation_memory_budget_api(
    activation_memory_budgets: Iterable[float | None],
) -> None:
    needs_budget = any(budget is not None for budget in activation_memory_budgets)
    if not needs_budget:
        return

    dynamo_module = getattr(torch, "_dynamo", None)
    if dynamo_module is None:
        raise RuntimeError(
            "Budgeted compile variants were requested, but torch._dynamo is unavailable."
        )
    _ = dynamo_module

    functorch_module = getattr(torch, "_functorch", None)
    functorch_config = getattr(functorch_module, "config", None)
    if functorch_config is None or not hasattr(
        functorch_config,
        "activation_memory_budget",
    ):
        raise RuntimeError(
            "Budgeted compile variants were requested, but "
            "torch._functorch.config.activation_memory_budget is unavailable. "
            "This experiment is configured to fail early in this case."
        )
