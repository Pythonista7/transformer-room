from __future__ import annotations

from typing import Any, Mapping

import torch

from baseline.core.config import LoggingConfig
from baseline.core.registry import register_logger_adapter
from baseline.core.types import LoggerSession


class ConsoleLoggerSession:
    def __init__(self, run_name: str | None):
        self.run_name = run_name or "console-run"

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        # Intentionally quiet by default; training loop already prints summaries.
        _ = metrics
        _ = step

    def save(self, path: str) -> None:
        _ = path

    def watch(self, model: torch.nn.Module, loss_fn: torch.nn.Module) -> None:
        _ = model
        _ = loss_fn

    def close(self) -> None:
        return


class ConsoleLoggerAdapter:
    def start(
        self,
        cfg: LoggingConfig,
        project_name: str,
        run_name: str | None,
        config_payload: dict[str, Any],
    ) -> LoggerSession:
        _ = cfg
        _ = project_name
        _ = config_payload
        return ConsoleLoggerSession(run_name=run_name)


class WandbLoggerSession:
    def __init__(self, run: Any):
        self._run = run

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._run.log(dict(metrics), step=step)

    def save(self, path: str) -> None:
        self._run.save(path)

    def watch(self, model: torch.nn.Module, loss_fn: torch.nn.Module) -> None:
        self._run.watch(model, loss_fn, log="all", log_freq=10)

    def close(self) -> None:
        self._run.finish()


class WandbLoggerAdapter:
    def start(
        self,
        cfg: LoggingConfig,
        project_name: str,
        run_name: str | None,
        config_payload: dict[str, Any],
    ) -> LoggerSession:
        _ = cfg
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "logging.provider='wandb' requires the `wandb` package."
            ) from exc

        run = wandb.init(project=project_name, name=run_name, config=config_payload)
        return WandbLoggerSession(run=run)


register_logger_adapter("console", ConsoleLoggerAdapter())
register_logger_adapter("wandb", WandbLoggerAdapter())
