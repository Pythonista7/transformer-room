from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from baseline.core.config import LoggingConfig
from baseline.core.registry import register_logger_adapter
from baseline.core.types import LoggerSession

EPHEMERAL_ARTIFACT_TYPES = {"checkpoint", "model"}


def sanitize_wandb_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return sanitized or "artifact"


def _wandb_artifact_not_found(exc: Exception) -> bool:
    message = str(exc).lower()
    class_name = exc.__class__.__name__.lower()
    return (
        isinstance(exc, FileNotFoundError)
        or "not found" in message
        or "404" in message
        or "not found" in class_name
    )


def _extract_alias_names(aliases: Sequence[Any] | None) -> set[str]:
    alias_names: set[str] = set()
    for alias in aliases or ():
        if isinstance(alias, str):
            alias_names.add(alias)
            continue
        name = getattr(alias, "name", None) or getattr(alias, "alias", None)
        if isinstance(name, str) and name:
            alias_names.add(name)
    return alias_names


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
    ) -> str | None:
        _ = path
        _ = artifact_name
        _ = artifact_type
        _ = aliases
        _ = metadata
        return None

    def restore(
        self,
        path: str,
        *,
        artifact_name: str,
        artifact_type: str | None = None,
        alias: str = "latest",
    ) -> bool:
        _ = path
        _ = artifact_name
        _ = artifact_type
        _ = alias
        return False

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

    def _api(self) -> Any:
        return self._wandb.Api()

    def _project_name(self) -> str | None:
        return getattr(self._run, "project", None) or getattr(self._run, "project_name", None)

    def _artifact_ref(self, artifact_name: str, alias: str) -> str:
        project_name = self._project_name()
        entity = (
            getattr(self._run, "entity", None)
            or os.environ.get("WANDB_ENTITY")
            or getattr(getattr(self._run, "settings", None), "entity", None)
        )
        if entity and project_name:
            return f"{entity}/{project_name}/{artifact_name}:{alias}"
        if project_name:
            return f"{project_name}/{artifact_name}:{alias}"
        return f"{artifact_name}:{alias}"

    def _default_artifact_name(self, path: Path, artifact_type: str) -> str:
        run_label = sanitize_wandb_name(getattr(self._run, "name", None) or "run")
        run_id = sanitize_wandb_name(getattr(self._run, "id", None) or "session")
        stem = sanitize_wandb_name(path.stem)
        return f"{run_label}-{run_id}-{artifact_type}-{stem}"

    def _prune_checkpoint_versions(self, artifact_name: str) -> None:
        versions = list(self._api().artifact_versions("checkpoint", artifact_name))
        for version in versions:
            if _extract_alias_names(getattr(version, "aliases", None)) & {"latest", "final"}:
                continue
            deleter = getattr(version, "delete", None)
            if callable(deleter):
                deleter(delete_aliases=True)

    def _wait_for_artifact(self, artifact: Any) -> None:
        waiter = getattr(artifact, "wait", None)
        if callable(waiter):
            waiter()

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
    ) -> str | None:
        artifact_path = Path(path)
        resolved_path = artifact_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {resolved_path}")

        resolved_type = artifact_type or "file"
        resolved_name = sanitize_wandb_name(
            artifact_name or self._default_artifact_name(resolved_path, resolved_type)
        )
        artifact = self._wandb.Artifact(
            name=resolved_name,
            type=resolved_type,
            metadata=dict(metadata) if metadata is not None else None,
        )
        alias_list = list(dict.fromkeys(aliases or ("latest",)))
        add_file_kwargs: dict[str, Any] = {"name": resolved_path.name}
        if resolved_type in EPHEMERAL_ARTIFACT_TYPES:
            add_file_kwargs["policy"] = "immutable"
            add_file_kwargs["skip_cache"] = True
        artifact.add_file(str(resolved_path), **add_file_kwargs)
        logged_artifact = self._run.log_artifact(artifact, aliases=alias_list)
        if logged_artifact is not None:
            self._wait_for_artifact(logged_artifact)
        else:
            self._wait_for_artifact(artifact)

        if resolved_type == "checkpoint":
            self._prune_checkpoint_versions(resolved_name)
        if resolved_type in EPHEMERAL_ARTIFACT_TYPES and resolved_path.exists():
            resolved_path.unlink()

        primary_alias = alias_list[0] if alias_list else "latest"
        return self._artifact_ref(resolved_name, primary_alias)

    def restore(
        self,
        path: str,
        *,
        artifact_name: str,
        artifact_type: str | None = None,
        alias: str = "latest",
    ) -> bool:
        _ = artifact_type
        target_path = Path(path).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_ref = self._artifact_ref(artifact_name, alias)

        try:
            use_artifact = getattr(self._run, "use_artifact", None)
            if callable(use_artifact):
                artifact = use_artifact(artifact_ref)
            else:
                artifact = self._api().artifact(artifact_ref)
        except Exception as exc:
            if _wandb_artifact_not_found(exc):
                return False
            raise

        with tempfile.TemporaryDirectory(
            dir=str(target_path.parent),
            prefix=".wandb-restore-",
        ) as tmpdir:
            download_root = Path(tmpdir)
            download_dir = Path(artifact.download(root=str(download_root)))
            candidates = [download_dir / target_path.name]
            candidates.extend(
                candidate for candidate in download_dir.rglob(target_path.name) if candidate.is_file()
            )
            source_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if source_path is None:
                downloaded_files = [candidate for candidate in download_dir.rglob("*") if candidate.is_file()]
                if len(downloaded_files) == 1:
                    source_path = downloaded_files[0]
            if source_path is None:
                return False
            shutil.copyfile(source_path, target_path)
        return True

    def watch(self, model: torch.nn.Module, loss_fn: torch.nn.Module) -> None:
        self._run.watch(model, loss_fn, log="all", log_freq=10)

    def close(self) -> None:
        self._run.finish()


class WandbLoggerAdapter:
    def _artifact_ref(self, project_name: str, artifact_name: str, alias: str) -> str:
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            return f"{entity}/{project_name}/{artifact_name}:{alias}"
        return f"{project_name}/{artifact_name}:{alias}"

    def has_remote_artifact(
        self,
        *,
        project_name: str,
        artifact_name: str,
        alias: str = "latest",
    ) -> bool:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "logging.provider='wandb' requires the `wandb` package."
            ) from exc

        try:
            wandb.Api().artifact(self._artifact_ref(project_name, artifact_name, alias))
            return True
        except Exception as exc:
            if _wandb_artifact_not_found(exc):
                return False
            raise

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
