from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import optim
from torch.optim.lr_scheduler import LRScheduler, LambdaLR

from ..core.config import (
    ExperimentConfig,
    LRSchedulerChainConfig,
    validate_train_lr_scheduler_config,
)


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


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    config: ExperimentConfig,
    *,
    total_optimizer_steps: int | None = None,
) -> LRScheduler | None:
    """Build a chained LR scheduler from `train.lr_scheduler`.

    Important semantics:
    - `start_factor` / `end_factor` are multiplicative LR factors, not absolute LR values.
    - They are applied to each param-group's current optimizer LR (which already includes
      any upstream LR scaling such as `train.lr_scaling`).
    - Effective step LR is therefore: `effective_lr = optimizer_base_lr * factor`.
    """
    resolved_stages = _resolve_lr_scheduler_stages(
        config.train.lr_scheduler,
        total_optimizer_steps=total_optimizer_steps,
    )
    if not resolved_stages:
        return None
    return LambdaLR(
        optimizer,
        lr_lambda=_build_stage_factor_fn(resolved_stages),
    )


@dataclass(frozen=True, slots=True)
class _ResolvedLRSchedulerStage:
    type: str
    start_factor: float
    end_factor: float
    steps: int
    start_step: int
    end_step_exclusive: int


def _resolve_lr_scheduler_stages(
    scheduler_cfg: LRSchedulerChainConfig | None,
    *,
    total_optimizer_steps: int | None,
) -> list[_ResolvedLRSchedulerStage]:
    validate_train_lr_scheduler_config(scheduler_cfg)
    if scheduler_cfg is None:
        return []
    if total_optimizer_steps is not None and int(total_optimizer_steps) <= 0:
        raise ValueError("total_optimizer_steps must be > 0 when provided.")

    stages = scheduler_cfg.stages
    explicit_steps_sum = sum(
        int(stage.steps)
        for stage in stages
        if stage.steps is not None
    )
    final_stage = stages[-1]
    if final_stage.steps is None:
        if total_optimizer_steps is None:
            raise ValueError(
                "train.lr_scheduler final stage with steps=None requires known total optimizer steps."
            )
        resolved_final_steps = int(total_optimizer_steps) - explicit_steps_sum
        if resolved_final_steps <= 0:
            raise ValueError(
                "Resolved final scheduler stage steps must be > 0. "
                "Reduce earlier stage durations or increase total optimizer steps."
            )
    else:
        resolved_final_steps = int(final_stage.steps)
        if (
            total_optimizer_steps is not None
            and explicit_steps_sum > int(total_optimizer_steps)
        ):
            raise ValueError(
                "Sum of train.lr_scheduler stage steps cannot exceed total optimizer steps."
            )

    resolved_stages: list[_ResolvedLRSchedulerStage] = []
    prev_end_factor = 1.0
    cursor = 0
    for idx, stage in enumerate(stages):
        start_factor = (
            prev_end_factor
            if stage.start_factor is None
            else float(stage.start_factor)
        )
        steps = (
            resolved_final_steps
            if idx == len(stages) - 1 and stage.steps is None
            else int(stage.steps)
        )
        resolved_stage = _ResolvedLRSchedulerStage(
            type=str(stage.type),
            start_factor=start_factor,
            end_factor=float(stage.end_factor),
            steps=steps,
            start_step=cursor,
            end_step_exclusive=cursor + steps,
        )
        resolved_stages.append(resolved_stage)
        prev_end_factor = resolved_stage.end_factor
        cursor = resolved_stage.end_step_exclusive
    return resolved_stages


def _build_stage_factor_fn(
    resolved_stages: list[_ResolvedLRSchedulerStage],
):
    final_factor = float(resolved_stages[-1].end_factor)

    def _factor(step: int) -> float:
        step_idx = max(0, int(step))
        for stage in resolved_stages:
            if step_idx < stage.end_step_exclusive:
                local_step = step_idx - stage.start_step
                return _stage_factor(stage, local_step)
        return final_factor

    return _factor


def _stage_factor(stage: _ResolvedLRSchedulerStage, local_step: int) -> float:
    if stage.steps <= 1:
        return float(stage.end_factor)

    t = float(local_step) / float(stage.steps - 1)
    if stage.type == "linear":
        return float(stage.start_factor + (stage.end_factor - stage.start_factor) * t)
    if stage.type == "cosine":
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        return float(stage.start_factor + (stage.end_factor - stage.start_factor) * alpha)
    raise ValueError(
        f"Unsupported train.lr_scheduler stage type '{stage.type}'. "
        "Expected one of: linear, cosine."
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
