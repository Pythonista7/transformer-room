from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

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

    def save(
        self,
        path: str,
        *,
        artifact_name: str | None = None,
        artifact_type: str | None = None,
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        _ = path
        _ = artifact_name
        _ = artifact_type
        _ = aliases
        _ = metadata

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
        group_name: str | None,
        config_payload: dict[str, Any],
    ) -> LoggerSession:
        _ = cfg
        _ = project_name
        _ = group_name
        _ = config_payload
        return ConsoleLoggerSession(run_name=run_name)


class WandbLoggerSession:
    def __init__(self, run: Any, wandb_module: Any):
        self._run = run
        self._wandb = wandb_module

    @staticmethod
    def _sanitize_name(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
        return sanitized or "artifact"

    def _default_artifact_name(self, path: Path, artifact_type: str) -> str:
        run_label = self._sanitize_name(getattr(self._run, "name", None) or "run")
        run_id = self._sanitize_name(getattr(self._run, "id", None) or "session")
        stem = self._sanitize_name(path.stem)
        return f"{run_label}-{run_id}-{artifact_type}-{stem}"

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._run.log(dict(metrics), step=step)

    def save(
        self,
        path: str,
        *,
        artifact_name: str | None = None,
        artifact_type: str | None = None,
        aliases: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        artifact_path = Path(path)
        resolved_path = artifact_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {resolved_path}")

        resolved_type = artifact_type or "file"
        resolved_name = self._sanitize_name(
            artifact_name or self._default_artifact_name(resolved_path, resolved_type)
        )
        artifact = self._wandb.Artifact(
            name=resolved_name,
            type=resolved_type,
            metadata=dict(metadata) if metadata is not None else None,
        )
        artifact.add_file(str(resolved_path), name=resolved_path.name)
        alias_list = list(dict.fromkeys(aliases or ("latest",)))
        self._run.log_artifact(artifact, aliases=alias_list)

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
        group_name: str | None,
        config_payload: dict[str, Any],
    ) -> LoggerSession:
        _ = cfg
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "logging.provider='wandb' requires the `wandb` package."
            ) from exc

        run = wandb.init(
            project=project_name,
            name=run_name,
            group=group_name,
            config=config_payload,
        )
        return WandbLoggerSession(run=run, wandb_module=wandb)


register_logger_adapter("console", ConsoleLoggerAdapter())
register_logger_adapter("wandb", WandbLoggerAdapter())
