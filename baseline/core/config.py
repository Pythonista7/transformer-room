from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .types import SpecialTokenIds


@dataclass(slots=True)
class RunConfig:
    project_name: str
    run_name: str | None = None
    artifacts_root: str = "baseline/models"
    resume_from_checkpoint: bool = True
    checkpoint_every_n_steps: int = 250
    checkpoint_filename: str = "baseline_checkpoint.pt"
    final_model_filename: str = "baseline_model.pt"
    use_torch_compile: bool = False
    torch_compile_mode: str = "default"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False


@dataclass(slots=True)
class LocalTextDatasetConfig:
    name: Literal["local_text"] = "local_text"
    path: str = ""
    segment_delimiter: str = "\n\n"


@dataclass(slots=True)
class HFTextDatasetConfig:
    name: Literal["hf_text"] = "hf_text"
    dataset_name: str = ""
    dataset_config: str | None = None
    split: str = "train"
    text_field: str | None = None
    streaming: bool = False
    max_rows: int = 0


DatasetConfig = LocalTextDatasetConfig | HFTextDatasetConfig


@dataclass(slots=True)
class BPETokenizerConfig:
    name: Literal["bpe"] = "bpe"
    base_vocab_size: int = 10_000
    num_special_tokens: int = 3
    vocab_path: str = ""


TokenizerConfig = BPETokenizerConfig


@dataclass(slots=True)
class BaselineDecoderConfig:
    name: Literal["baseline_decoder"] = "baseline_decoder"
    d_model: int = 128
    n_heads: int = 8
    layers: int = 2


ModelConfig = BaselineDecoderConfig


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 256
    seq_len: int = 128
    stride: int = 128
    data_fraction: float = 1.0


@dataclass(slots=True)
class HoldoutSplitConfig:
    name: Literal["holdout"] = "holdout"
    train_fraction: float = 0.9
    seed: int = 42
    shuffle: bool = False


SplitConfig = HoldoutSplitConfig


@dataclass(slots=True)
class LoggingConfig:
    provider: Literal["console", "wandb"] = "console"


@dataclass(slots=True)
class ExperimentConfig:
    run: RunConfig
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    train: TrainConfig
    split: SplitConfig = field(default_factory=HoldoutSplitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_special_token_ids(tokenizer_cfg: BPETokenizerConfig) -> SpecialTokenIds:
    base_vocab_size = int(tokenizer_cfg.base_vocab_size)
    num_special_tokens = int(tokenizer_cfg.num_special_tokens)

    if base_vocab_size <= 0:
        raise ValueError(f"base_vocab_size must be > 0, got {base_vocab_size}")
    if num_special_tokens < 2:
        raise ValueError(
            "This pipeline expects at least 2 special tokens (EOS and PAD)."
        )

    eos_id = base_vocab_size
    pad_id = base_vocab_size + 1
    unk_id = base_vocab_size + 2 if num_special_tokens >= 3 else None
    return SpecialTokenIds(
        base_vocab_size=base_vocab_size,
        num_special_tokens=num_special_tokens,
        eos_id=eos_id,
        pad_id=pad_id,
        unk_id=unk_id,
    )


def validate_experiment_config(config: ExperimentConfig) -> None:
    if not config.run.project_name.strip():
        raise ValueError("run.project_name must be non-empty.")
    if not config.run.artifacts_root.strip():
        raise ValueError("run.artifacts_root must be non-empty.")
    if config.run.checkpoint_every_n_steps < 0:
        raise ValueError("run.checkpoint_every_n_steps must be >= 0.")
    if not config.run.checkpoint_filename.strip():
        raise ValueError("run.checkpoint_filename must be non-empty.")
    if not config.run.final_model_filename.strip():
        raise ValueError("run.final_model_filename must be non-empty.")

    if config.dataset.name == "local_text":
        if not config.dataset.path.strip():
            raise ValueError("dataset.path is required for local_text dataset.")
    elif config.dataset.name == "hf_text":
        if not config.dataset.dataset_name.strip():
            raise ValueError("dataset.dataset_name is required for hf_text dataset.")
        if config.dataset.max_rows < 0:
            raise ValueError("dataset.max_rows must be >= 0.")
        if not config.dataset.split.strip():
            raise ValueError("dataset.split must be non-empty for hf_text dataset.")
    else:
        raise ValueError(
            f"Unsupported dataset.name '{config.dataset.name}'. "
            "Expected one of: local_text, hf_text."
        )

    if config.tokenizer.name != "bpe":
        raise ValueError(
            f"Unsupported tokenizer.name '{config.tokenizer.name}'. Expected: bpe."
        )
    if not config.tokenizer.vocab_path.strip():
        raise ValueError("tokenizer.vocab_path must be non-empty.")
    resolve_special_token_ids(config.tokenizer)

    if config.model.name != "baseline_decoder":
        raise ValueError(
            f"Unsupported model.name '{config.model.name}'. Expected: baseline_decoder."
        )
    if config.model.d_model <= 0:
        raise ValueError("model.d_model must be > 0.")
    if config.model.n_heads <= 0:
        raise ValueError("model.n_heads must be > 0.")
    if config.model.layers <= 0:
        raise ValueError("model.layers must be > 0.")
    if config.model.d_model % config.model.n_heads != 0:
        raise ValueError(
            "model.d_model must be divisible by model.n_heads "
            f"(got d_model={config.model.d_model}, n_heads={config.model.n_heads})."
        )

    if config.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0.")
    if config.train.learning_rate <= 0:
        raise ValueError("train.learning_rate must be > 0.")
    if config.train.batch_size <= 0:
        raise ValueError("train.batch_size must be > 0.")
    if config.train.seq_len <= 0:
        raise ValueError("train.seq_len must be > 0.")
    if config.train.stride <= 0:
        raise ValueError("train.stride must be > 0.")
    if not 0 < config.train.data_fraction <= 1:
        raise ValueError("train.data_fraction must be in (0, 1].")

    if config.split.name != "holdout":
        raise ValueError(
            f"Unsupported split.name '{config.split.name}'. Expected: holdout."
        )
    if not 0 < config.split.train_fraction < 1:
        raise ValueError("split.train_fraction must be in (0, 1).")

    if config.logging.provider not in {"console", "wandb"}:
        raise ValueError(
            f"Unsupported logging.provider '{config.logging.provider}'. "
            "Expected one of: console, wandb."
        )
