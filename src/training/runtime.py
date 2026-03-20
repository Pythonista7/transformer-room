from __future__ import annotations

from contextlib import nullcontext

import torch

from ..core.config import ExperimentConfig


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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


def maybe_compile_model(
    model: torch.nn.Module, device: torch.device, config: ExperimentConfig
) -> tuple[torch.nn.Module, bool, str]:
    if not config.run.use_torch_compile:
        return model, False, "disabled"
    if not hasattr(torch, "compile"):
        return model, False, "torch.compile unavailable"
    if device.type != "cuda":
        return model, False, f"skipped on {device.type}"

    _set_activation_memory_budget_if_configured(config)

    try:
        compiled_model = torch.compile(
            model,
            mode=config.run.torch_compile_mode,
            fullgraph=bool(config.run.torch_compile_fullgraph),
            dynamic=bool(config.run.torch_compile_dynamic),
        )
        return compiled_model, True, "enabled"
    except Exception as exc:  # pragma: no cover - backend-specific failure paths.
        return model, False, f"failed: {exc}"
