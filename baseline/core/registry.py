from __future__ import annotations

from typing import TypeVar

from .types import (
    DatasetAdapter,
    LoggerAdapter,
    ModelAdapter,
    SplitAdapter,
    TokenizerAdapter,
)

DATASET_ADAPTERS: dict[str, DatasetAdapter] = {}
TOKENIZER_ADAPTERS: dict[str, TokenizerAdapter] = {}
MODEL_ADAPTERS: dict[str, ModelAdapter] = {}
SPLIT_ADAPTERS: dict[str, SplitAdapter] = {}
LOGGER_ADAPTERS: dict[str, LoggerAdapter] = {}


def register_dataset_adapter(name: str, adapter: DatasetAdapter) -> None:
    DATASET_ADAPTERS[name] = adapter


def register_tokenizer_adapter(name: str, adapter: TokenizerAdapter) -> None:
    TOKENIZER_ADAPTERS[name] = adapter


def register_model_adapter(name: str, adapter: ModelAdapter) -> None:
    MODEL_ADAPTERS[name] = adapter


def register_split_adapter(name: str, adapter: SplitAdapter) -> None:
    SPLIT_ADAPTERS[name] = adapter


def register_logger_adapter(name: str, adapter: LoggerAdapter) -> None:
    LOGGER_ADAPTERS[name] = adapter


T = TypeVar("T")


def _get_adapter(kind: str, name: str, registry: dict[str, T]) -> T:
    if name in registry:
        return registry[name]
    available = ", ".join(sorted(registry)) or "<none>"
    raise KeyError(
        f"Unknown {kind} adapter '{name}'. Available {kind} adapters: {available}"
    )


def get_dataset_adapter(name: str) -> DatasetAdapter:
    return _get_adapter("dataset", name, DATASET_ADAPTERS)


def get_tokenizer_adapter(name: str) -> TokenizerAdapter:
    return _get_adapter("tokenizer", name, TOKENIZER_ADAPTERS)


def get_model_adapter(name: str) -> ModelAdapter:
    return _get_adapter("model", name, MODEL_ADAPTERS)


def get_split_adapter(name: str) -> SplitAdapter:
    return _get_adapter("split", name, SPLIT_ADAPTERS)


def get_logger_adapter(name: str) -> LoggerAdapter:
    return _get_adapter("logger", name, LOGGER_ADAPTERS)
