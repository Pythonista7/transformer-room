from __future__ import annotations

import torch
from torch import optim

from ..core.config import ExperimentConfig


def move_optimizer_state_to_device(
    optimizer: optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_optimizer(
    model: torch.nn.Module,
    config: ExperimentConfig,
    *,
    learning_rate: float | None = None,
) -> optim.Optimizer:
    optimizer_cfg = config.train.optimizer
    resolved_learning_rate = (
        float(optimizer_cfg.learning_rate)
        if learning_rate is None
        else float(learning_rate)
    )
    optimizer_kwargs = {
        "lr": resolved_learning_rate,
        "weight_decay": optimizer_cfg.weight_decay,
    }
    if optimizer_cfg.name == "adam":
        return optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_cfg.name == "adamw":
        return optim.AdamW(model.parameters(), **optimizer_kwargs)
    if optimizer_cfg.name == "sgd":
        return optim.SGD(model.parameters(), **optimizer_kwargs)
    raise ValueError(
        f"Unsupported train.optimizer.name '{optimizer_cfg.name}'. "
        "Expected one of: adam, adamw, sgd."
    )


def scale_gradients_by_token_count(
    model: torch.nn.Module,
    token_count: int,
) -> None:
    if token_count <= 0:
        raise ValueError(f"token_count must be > 0, got {token_count}")

    scale = 1.0 / float(token_count)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
