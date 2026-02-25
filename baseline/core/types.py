from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .config import (
        BaselineDecoderConfig,
        BPETokenizerConfig,
        HFTextDatasetConfig,
        HoldoutSplitConfig,
        LocalTextDatasetConfig,
        LoggingConfig,
    )


@dataclass(slots=True)
class SpecialTokenIds:
    base_vocab_size: int
    num_special_tokens: int
    eos_id: int
    pad_id: int
    unk_id: int | None

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + self.num_special_tokens


@dataclass(slots=True)
class VocabInfo:
    token_to_id: dict[Any, int]
    id_to_token: list[Any]
    special: SpecialTokenIds

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)


@dataclass(slots=True)
class TextCorpus:
    full_text: str
    segments: list[str]
    source_description: str


@dataclass(slots=True)
class TokenizedCorpus:
    token_stream: list[int]
    vocab: VocabInfo
    tokenizer: Any
    eos_inserted: int
    unk_replacements: int


@dataclass(slots=True)
class RunResult:
    model: torch.nn.Module
    device: torch.device
    run_artifact_dir: str
    checkpoint_path: str
    final_model_path: str
    global_step: int
    final_train_loss: float
    final_val_loss: float
    final_val_perplexity: float


@runtime_checkable
class DatasetAdapter(Protocol):
    def load(self, cfg: LocalTextDatasetConfig | HFTextDatasetConfig) -> TextCorpus:
        """Load text corpus and normalized text segments."""


@runtime_checkable
class TokenizerAdapter(Protocol):
    def build(self, corpus: TextCorpus, cfg: BPETokenizerConfig) -> TokenizedCorpus:
        """Build/load tokenizer and return tokenized corpus metadata."""


@runtime_checkable
class ModelAdapter(Protocol):
    def build(
        self,
        cfg: BaselineDecoderConfig,
        vocab: VocabInfo,
        special: SpecialTokenIds,
    ) -> torch.nn.Module:
        """Build a model for the given vocab and special-token layout."""


@runtime_checkable
class SplitAdapter(Protocol):
    def split(
        self,
        dataset: Dataset,
        cfg: HoldoutSplitConfig,
    ) -> tuple[Dataset, Dataset]:
        """Split a dataset into train and validation subsets."""


@runtime_checkable
class LoggerSession(Protocol):
    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log scalar metrics."""

    def save(self, path: str) -> None:
        """Track a saved artifact path."""

    def watch(self, model: torch.nn.Module, loss_fn: torch.nn.Module) -> None:
        """Optionally watch model gradients/weights."""

    def close(self) -> None:
        """Close logging session."""


@runtime_checkable
class LoggerAdapter(Protocol):
    def start(
        self,
        cfg: LoggingConfig,
        project_name: str,
        run_name: str | None,
        config_payload: dict[str, Any],
    ) -> LoggerSession:
        """Start a new logging session."""
