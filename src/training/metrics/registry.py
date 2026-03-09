from __future__ import annotations

from typing import Sequence

import torch
from torch import optim

from src.core.config import ExperimentConfig

from .contracts import MetricPlugin
from .plugins import (
    ForwardHookMetricsPlugin,
    GlobalGradNormPlugin,
    LayerNormGradNormPlugin,
    LossPerplexityPlugin,
    ParameterOptimizerNormsPlugin,
    StepTimingAndMemoryPlugin,
)


def build_default_metric_plugins(
    *,
    config: ExperimentConfig,
    checkpoint_model: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    layer_labels: dict[int, list[str]],
    wandb_enabled: bool,
    extra_plugins: Sequence[MetricPlugin] | None = None,
) -> list[MetricPlugin]:
    wandb_cfg = config.logging.wandb
    plugins: list[MetricPlugin] = [
        LossPerplexityPlugin(
            wandb_enabled=wandb_enabled,
            wandb_cfg=wandb_cfg,
        ),
        StepTimingAndMemoryPlugin(
            wandb_cfg=wandb_cfg,
            device=device,
        ),
        GlobalGradNormPlugin(
            wandb_cfg=wandb_cfg,
            model=checkpoint_model,
        ),
        LayerNormGradNormPlugin(
            wandb_cfg=wandb_cfg,
            model=checkpoint_model,
            layer_labels=layer_labels,
        ),
        ParameterOptimizerNormsPlugin(
            wandb_cfg=wandb_cfg,
            model=checkpoint_model,
            optimizer=optimizer,
            layer_labels=layer_labels,
        ),
        ForwardHookMetricsPlugin(
            wandb_enabled=wandb_enabled,
            wandb_cfg=wandb_cfg,
            model=checkpoint_model,
            layer_labels=layer_labels,
        ),
    ]
    if extra_plugins:
        plugins.extend(extra_plugins)
    return plugins
