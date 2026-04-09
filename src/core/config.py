from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from .types import SpecialTokenIds

AttentionImplementation = Literal["basic", "sdpa"]
NormPlacement = Literal["pre","post"]
DataMode = Literal["materialized", "streaming"]
LRSchedulerStageType = Literal["linear", "cosine"]
HFPretrainedBpbMode = Literal["off", "approx", "exact"]


@dataclass(slots=True)
class RunConfig:
    """Run identity, artifact behavior, and compile/runtime controls.

    Parameters:
    - `project_name`: logger project namespace.
    - `run_name`: stable run identifier. Required for `logging.provider="wandb"`.
    - `group_name`: optional grouping label (for example, sweep or phase group).
    - `artifacts_root`: local directory root for checkpoints/models/metadata.
    - `persist_local_artifacts`: when `False`, skip local checkpoint/final-model writes.
    - `resume_from_checkpoint`: attempt startup restore from local/remote checkpoint.
    - `checkpoint_every_n_steps`: in-loop checkpoint cadence; `0` disables periodic saves.
    - `checkpoint_filename`: local checkpoint filename under run artifact dir.
    - `final_model_filename`: local final model filename under run artifact dir.
    - `use_torch_compile`: enable `torch.compile` when supported.
    - `torch_compile_mode`: compile mode string passed to `torch.compile(...)`.
    - `torch_compile_fullgraph`: forward `fullgraph` flag to `torch.compile(...)`.
    - `torch_compile_dynamic`: forward `dynamic` flag to `torch.compile(...)`.
    - `torch_compile_trace`: set `TORCH_TRACE` before calling `torch.compile(...)`.
    - `activation_memory_budget`: optional Torch functorch activation memory budget in
      `[0, 1]`; applied only when supported by the installed torch build.
    - `compile_warmup_steps`: number of initial compiled steps excluded from perf
      aggregate metrics (helps avoid compile warmup skew).
    - `seed`: global RNG seed.
    """
    project_name: str
    run_name: str | None = None
    group_name: str | None = None
    artifacts_root: str = "src/models"
    persist_local_artifacts: bool = True
    resume_from_checkpoint: bool = True
    checkpoint_every_n_steps: int = 10_000
    checkpoint_filename: str = "baseline_checkpoint.pt"
    final_model_filename: str = "baseline_model.pt"
    use_torch_compile: bool = False
    torch_compile_mode: str = "default"
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
    torch_compile_trace: bool = False
    activation_memory_budget: float | None = None
    compile_warmup_steps: int = 3 if use_torch_compile else 0
    seed: int = 42


@dataclass(slots=True)
class LocalTextDatasetConfig:
    name: Literal["local_text"] = "local_text"
    path: str = ""
    segment_delimiter: str = "\n\n"


@dataclass(slots=True)
class HFTextDatasetConfig:
    """Hugging Face dataset selection and row-to-text extraction options.

    Parameters:
    - `dataset_name`: HF dataset path/name passed to `load_dataset`.
    - `dataset_config`: optional HF config/subset name.
    - `split`: training split name.
    - `validation_split`: optional validation split name.
      Required when `train.run_validation=True` in streaming mode.
    - `text_field`: row field containing text. If `None`, inferred from sample rows.
    - `shuffle_buffer_size`: streaming shuffle buffer size.
    - `max_rows`: optional row cap. `0` means "no cap".
    """
    name: Literal["hf_text"] = "hf_text"
    dataset_name: str = ""
    dataset_config: str | None = None
    split: str = "train"
    validation_split: str | None = None
    text_field: str | None = None
    shuffle_buffer_size: int = 10_000
    max_rows: int = 0


DatasetConfig = LocalTextDatasetConfig | HFTextDatasetConfig


@dataclass(slots=True)
class BPETokenizerConfig:
    """Local BPE tokenizer config.

    Parameters:
    - `base_vocab_size`: learned non-special vocab size.
    - `num_special_tokens`: special-token count appended after base vocab.
      Must be at least 2 (EOS + PAD); 3 enables UNK.
    - `vocab_path`: path used for tokenizer vocab persistence/loading.
    """
    name: Literal["bpe"] = "bpe"
    base_vocab_size: int = 10_000
    num_special_tokens: int = 3
    vocab_path: str = ""


@dataclass(slots=True)
class HFPretrainedTokenizerConfig:
    """Hugging Face tokenizer options, including BPB behavior for streaming.

    Parameters:
    - `pretrained_name_or_path`: tokenizer model id or local path.
    - `use_fast`: request Rust-backed fast tokenizer implementation.
    - `bpb_mode`: byte-accounting mode used by HF streaming datasets when
      `logging.wandb.enable_bits_per_byte=True`.
      - `"off"`: no byte accounting from tokenizer outputs; streaming BPB metrics
        will typically be `NaN`.
      - `"approx"`: estimate bytes via token byte-length lookup table.
      - `"exact"`: exact bytes via offset mapping from tokenizer outputs.
        Requires `use_fast=True` and tokenizer offset support.
      This setting is currently used by the HF streaming data path. In the
      materialized HF path, BPB may still be `NaN` because token byte lengths are
      not currently precomputed there.
    - `revision`: optional HF revision/tag/commit.
    - `trust_remote_code`: allow execution of remote tokenizer code from HF repos.
      Enable only for trusted sources.
    """
    name: Literal["hf_pretrained"] = "hf_pretrained"
    pretrained_name_or_path: str = ""
    use_fast: bool = True
    bpb_mode: HFPretrainedBpbMode = "off"
    revision: str | None = None
    trust_remote_code: bool = False


TokenizerConfig = BPETokenizerConfig | HFPretrainedTokenizerConfig


@dataclass(slots=True)
class BaselineDecoderConfig:
    """Baseline decoder architecture flags.

    Notes:
    - `norm_placement` selects pre-norm vs post-norm block layout.
    - `enable_weight_tying` ties output projection weights to token embeddings.
    """
    name: Literal["baseline_decoder"] = "baseline_decoder"
    d_model: int = 128
    n_heads: int = 8
    layers: int = 2
    dropout: float = 0.1
    attention_impl: AttentionImplementation = "basic"
    norm_placement: NormPlacement = "post"
    enable_weight_tying: bool = False

@dataclass(slots=True)
class ACEveryNDecoderConfig:
    """Activation-checkpointed decoder variant.

    Notes:
    - `checkpoint_every_n_layers` sets checkpointing stride across decoder layers.
      Lower values reduce activation memory but increase compute overhead.
    """
    name: Literal["ac_every_n_decoder"] = "ac_every_n_decoder"
    d_model: int = 128
    n_heads: int = 8
    layers: int = 2
    dropout: float = 0.1
    attention_impl: AttentionImplementation = "basic"
    checkpoint_every_n_layers: int = 1


@dataclass(slots=True)
class SACDecoderConfig:
    name: Literal["sac_decoder"] = "sac_decoder"
    d_model: int = 128
    n_heads: int = 8
    layers: int = 2
    dropout: float = 0.1
    attention_impl: AttentionImplementation = "basic"


ModelConfig = BaselineDecoderConfig | ACEveryNDecoderConfig | SACDecoderConfig


@dataclass(slots=True)
class OptimizerConfig:
    name: Literal["adam", "adamw", "sgd"] = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0


@dataclass(slots=True)
class LRSchedulerStageConfig:
    """One scheduler stage defined by LR factors relative to base LR.

    Notes:
    - `steps=None` is only allowed for the final stage in the chain.
    - `start_factor=None` means "continue from previous stage end".
    """
    type: LRSchedulerStageType
    end_factor: float
    steps: int | None = None
    start_factor: float | None = None


@dataclass(slots=True)
class LRSchedulerChainConfig:
    """Sequential LR scheduler stages applied in order."""
    stages: list[LRSchedulerStageConfig] = field(default_factory=list)


@dataclass(slots=True)
class TrainConfig:
    """Training loop controls for batching, scheduling, and data iteration.

    Parameters:
    - `effective_batch_size`: optimizer-step batch size target.
    - `epochs`: epoch budget. Optional only when `max_steps` is set.
    - `optimizer`: optimizer settings.
    - `micro_batch_size`: per-microbatch loader batch size. Auto-resolved when omitted.
    - `accumulation_steps`: gradient accumulation factor. Auto-resolved when omitted.
    - `lr_scaling`: LR scaling mode. Must be `"sqrt"` when accumulation is active.
    - `lr_scheduler`: optional chained LR scheduler config.
    - `seq_len`: language-model context length.
    - `stride`: window stride for dataset chunking.
    - `data_fraction`: fraction of materialized dataset to use.
      Not supported in streaming mode (must be `1.0`).
    - `data_mode`: `"materialized"` or `"streaming"`.
      Streaming requires `dataset.name="hf_text"`,
      `tokenizer.name="hf_pretrained"`, and `split.name="pre_split"`.
    - `max_steps`: optional optimizer-step cap. Supported only in streaming mode.
    - `run_validation`: enable validation. Streaming validation additionally requires
      `dataset.validation_split`.

    Behavior:
    - `effective_batch_size == micro_batch_size * accumulation_steps` is enforced.
    """
    effective_batch_size: int
    epochs: int | None = 3
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    micro_batch_size: int | None = None
    accumulation_steps: int | None = None
    lr_scaling: Literal["none", "sqrt"] = "none"
    lr_scheduler: LRSchedulerChainConfig | None = None
    seq_len: int = 128
    stride: int = 128
    data_fraction: float = 1.0
    data_mode: DataMode = "materialized"
    max_steps: int | None = None
    run_validation: bool = True

    def __post_init__(self) -> None:
        resolved = resolve_train_batching(self)
        self.effective_batch_size = resolved.effective_batch_size
        self.micro_batch_size = resolved.micro_batch_size
        self.accumulation_steps = resolved.accumulation_steps
        resolve_train_learning_rate(self)
        validate_train_lr_scheduler_config(self.lr_scheduler)


@dataclass(frozen=True, slots=True)
class ResolvedTrainBatchingConfig:
    effective_batch_size: int
    micro_batch_size: int
    accumulation_steps: int

    @property
    def loader_batch_size(self) -> int:
        return int(self.micro_batch_size)


@dataclass(frozen=True, slots=True)
class ResolvedTrainLearningRateConfig:
    base_learning_rate: float
    scale_factor: float
    applied_learning_rate: float
    scaling_active: bool
    scaling_mode: Literal["none", "sqrt"]


@dataclass(slots=True)
class HoldoutSplitConfig:
    """Random holdout split generated from one dataset source.

    Parameters:
    - `train_fraction`: fraction of examples routed to train set.
    - `seed`: split RNG seed.
    - `shuffle`: whether to shuffle before splitting.
    """
    name: Literal["holdout"] = "holdout"
    train_fraction: float = 0.9
    seed: int = 42
    shuffle: bool = False


@dataclass(slots=True)
class PreSplitConfig:
    """Use dataset-provided train/validation splits (no random split step).

    Note:
    - Supported only when `train.data_mode="streaming"`.
    """
    name: Literal["pre_split"] = "pre_split"


SplitConfig = HoldoutSplitConfig | PreSplitConfig


@dataclass(slots=True)
class WandbMetricsConfig:
    """W&B metric toggles, cadences, and sampling controls.

    Metric toggles:
    - `enable_train_loss_vs_tokens`: log step/epoch train loss with `tokens_seen_train`.
    - `enable_val_loss_vs_tokens`: log periodic/epoch val loss with `tokens_seen_train`.
    - `enable_perplexity`: log train/val perplexity metrics.
    - `enable_bits_per_byte`: log train/val BPB metrics.
      For HF streaming runs, meaningful BPB additionally requires
      `tokenizer.bpb_mode` to be `"approx"` or `"exact"`; `"off"` typically yields `NaN`.
      In general, BPB is meaningful only when byte lengths are available in the
      batch/runtime path (for example BPE materialized or HF streaming with BPB mode).
    - `enable_step_time`: log step and pass timing metrics.
    - `enable_peak_memory`: log CUDA peak allocated/reserved memory metrics.
    - `enable_global_grad_norm`: log `global_grad_norm` on diagnostics cadence.
    - `enable_layer_grad_norms`: log sampled per-layer gradient norms.
    - `enable_global_param_norm`: log global parameter L2 norm.
    - `enable_layer_param_norms`: log first/middle/last layer parameter norms.
    - `enable_param_update_norm`: log parameter update norm per optimizer step.
    - `enable_update_to_weight_ratio`: log update-to-weight ratio.
    - `enable_optimizer_state_norms`: log Adam/AdamW moment and variance norms.
    - `enable_activation_norms`: log activation norms for first/middle/last decoder layers.
    - `enable_ln_grad_norms`: log LayerNorm weight/bias grad norms for first/middle/last.
    - `enable_attention_entropy`: log sampled attention entropy metrics.
    - `watch_model`: enable `wandb.watch(...)` model tracking.

    Cadence controls:
    - `log_every_n_steps`: cadence for step metrics (loss/tokens/perplexity/BPB/time/memory).
    - `diagnostics_every_n_steps`: general diagnostics cadence (`should_log_diagnostics`),
      gating global grad norm, activation norms, and LayerNorm grad norms.
    - `layer_grad_norms_every_n_steps`: optional cadence override for layer grad norms.
      Defaults to `diagnostics_every_n_steps` when `None`.
    - `parameter_optimizer_norms_every_n_steps`: optional cadence override for
      parameter/optimizer norms. Defaults to `diagnostics_every_n_steps` when `None`.
    - `val_every_n_steps`: periodic validation cadence. `0` disables periodic val
      (epoch-end validation still depends on `train.run_validation`).
    - `attention_entropy_every_n_steps`: attention entropy cadence.

    Sampling controls:
    - `layer_grad_norm_stride`: layer-index stride used when sampling layer grad norms.
    - `attention_entropy_head_cap`: number of attention heads sampled.
    - `attention_entropy_token_cap`: token cap per axis for entropy computation.
    """
    enable_train_loss_vs_tokens: bool = True
    enable_val_loss_vs_tokens: bool = True
    enable_perplexity: bool = True
    enable_bits_per_byte: bool = True
    enable_step_time: bool = True
    enable_peak_memory: bool = True
    enable_global_grad_norm: bool = True
    enable_layer_grad_norms: bool = False
    enable_global_param_norm: bool = False
    enable_layer_param_norms: bool = False
    enable_param_update_norm: bool = False
    enable_update_to_weight_ratio: bool = False
    enable_optimizer_state_norms: bool = False
    enable_activation_norms: bool = False
    enable_ln_grad_norms: bool = False
    enable_attention_entropy: bool = False
    watch_model: bool = False
    log_every_n_steps: int = 25
    diagnostics_every_n_steps: int = 25
    layer_grad_norm_stride: int = 4
    layer_grad_norms_every_n_steps: int | None = None
    parameter_optimizer_norms_every_n_steps: int | None = None
    val_every_n_steps: int = 250
    attention_entropy_every_n_steps: int = 200
    attention_entropy_head_cap: int = 2
    attention_entropy_token_cap: int = 128


@dataclass(slots=True)
class LoggingConfig:
    """Logger backend and artifact I/O controls.

    Parameters:
    - `provider`: logging backend.
    - `enable_artifact_io`: enables remote artifact save/restore.
    - `wandb`: W&B metric toggles/cadences.

    Behavior:
    - `provider="wandb"` requires `run.run_name` to be set.
    - `enable_artifact_io=False` disables remote artifact save/restore operations.
      Local file writes still depend on `run.persist_local_artifacts`.
    """
    provider: Literal["console", "local", "wandb"] = "console"
    enable_artifact_io: bool = True
    wandb: WandbMetricsConfig = field(default_factory=WandbMetricsConfig)


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


def resolve_train_batching(train_cfg: TrainConfig) -> ResolvedTrainBatchingConfig:
    effective_batch_size = int(train_cfg.effective_batch_size)
    if effective_batch_size <= 0:
        raise ValueError("train.effective_batch_size must be > 0.")

    micro_batch_size = (
        effective_batch_size
        if train_cfg.micro_batch_size is None
        else int(train_cfg.micro_batch_size)
    )
    if micro_batch_size <= 0:
        raise ValueError("train.micro_batch_size must be > 0 when provided.")

    accumulation_steps = (
        1
        if train_cfg.accumulation_steps is None
        else int(train_cfg.accumulation_steps)
    )
    if accumulation_steps <= 0:
        raise ValueError("train.accumulation_steps must be > 0 when provided.")

    if effective_batch_size != micro_batch_size * accumulation_steps:
        raise ValueError(
            "train.effective_batch_size must equal "
            "train.micro_batch_size * train.accumulation_steps."
        )

    return ResolvedTrainBatchingConfig(
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
    )


def resolve_train_learning_rate(
    train_cfg: TrainConfig,
) -> ResolvedTrainLearningRateConfig:
    batching = resolve_train_batching(train_cfg)
    scaling_mode = str(train_cfg.lr_scaling)
    if scaling_mode not in {"none", "sqrt"}:
        raise ValueError("train.lr_scaling must be one of: none, sqrt.")

    accumulation_active = batching.effective_batch_size > batching.micro_batch_size
    if accumulation_active and scaling_mode != "sqrt":
        raise ValueError(
            "train.lr_scaling must be 'sqrt' when "
            "train.effective_batch_size > train.micro_batch_size."
        )

    base_learning_rate = float(train_cfg.optimizer.learning_rate)
    scale_factor = 1.0
    scaling_active = False
    if (
        scaling_mode == "sqrt"
        and batching.effective_batch_size > batching.micro_batch_size
        and batching.accumulation_steps > 1
    ):
        scale_factor = math.sqrt(
            float(batching.effective_batch_size) / float(batching.micro_batch_size)
        )
        scaling_active = True

    return ResolvedTrainLearningRateConfig(
        base_learning_rate=base_learning_rate,
        scale_factor=scale_factor,
        applied_learning_rate=base_learning_rate * scale_factor,
        scaling_active=scaling_active,
        scaling_mode=scaling_mode,
    )


def validate_train_lr_scheduler_config(
    scheduler_cfg: LRSchedulerChainConfig | None,
) -> None:
    if scheduler_cfg is None:
        return

    stages = scheduler_cfg.stages
    if len(stages) == 0:
        raise ValueError(
            "train.lr_scheduler.stages must contain at least one stage when set."
        )

    for idx, stage in enumerate(stages):
        if stage.type not in {"linear", "cosine"}:
            raise ValueError(
                "train.lr_scheduler.stages[].type must be one of: linear, cosine."
            )
        if not 0.0 <= float(stage.end_factor) <= 1.0:
            raise ValueError(
                "train.lr_scheduler.stages[].end_factor must be in [0, 1]."
            )
        if stage.start_factor is not None and not 0.0 <= float(stage.start_factor) <= 1.0:
            raise ValueError(
                "train.lr_scheduler.stages[].start_factor must be in [0, 1] when set."
            )
        if stage.steps is None:
            if idx != len(stages) - 1:
                raise ValueError(
                    "train.lr_scheduler.stages[].steps can be None only for the final stage."
                )
            continue
        if int(stage.steps) <= 0:
            raise ValueError("train.lr_scheduler.stages[].steps must be > 0 when set.")


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
        vocab_size=base_vocab_size + num_special_tokens,
        eos_id=eos_id,
        pad_id=pad_id,
        unk_id=unk_id,
        base_vocab_size=base_vocab_size,
        num_special_tokens=num_special_tokens,
    )


def validate_experiment_config(config: ExperimentConfig) -> None:
    if not config.run.project_name.strip():
        raise ValueError("run.project_name must be non-empty.")
    if config.run.run_name is not None and not config.run.run_name.strip():
        raise ValueError("run.run_name must be non-empty when provided.")
    if config.run.group_name is not None and not config.run.group_name.strip():
        raise ValueError("run.group_name must be non-empty when provided.")
    if config.run.seed < 0:
        raise ValueError("run.seed must be >= 0.")
    if not config.run.artifacts_root.strip():
        raise ValueError("run.artifacts_root must be non-empty.")
    if not isinstance(config.run.persist_local_artifacts, bool):
        raise ValueError("run.persist_local_artifacts must be a bool.")
    if config.run.checkpoint_every_n_steps < 0:
        raise ValueError("run.checkpoint_every_n_steps must be >= 0.")
    if config.run.compile_warmup_steps < 0:
        raise ValueError("run.compile_warmup_steps must be >= 0.")
    if config.run.activation_memory_budget is not None and not (
        0.0 <= config.run.activation_memory_budget <= 1.0
    ):
        raise ValueError("run.activation_memory_budget must be in [0, 1] when set.")
    if not isinstance(config.run.torch_compile_trace, bool):
        raise ValueError("run.torch_compile_trace must be a bool.")
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
        if (
            config.dataset.validation_split is not None
            and not config.dataset.validation_split.strip()
        ):
            raise ValueError(
                "dataset.validation_split must be non-empty when provided."
            )
        if config.dataset.shuffle_buffer_size <= 0:
            raise ValueError("dataset.shuffle_buffer_size must be > 0.")
    else:
        raise ValueError(
            f"Unsupported dataset.name '{config.dataset.name}'. "
            "Expected one of: local_text, hf_text."
        )

    if config.tokenizer.name == "bpe":
        if not config.tokenizer.vocab_path.strip():
            raise ValueError("tokenizer.vocab_path must be non-empty.")
        resolve_special_token_ids(config.tokenizer)
    elif config.tokenizer.name == "hf_pretrained":
        if not config.tokenizer.pretrained_name_or_path.strip():
            raise ValueError(
                "tokenizer.pretrained_name_or_path must be non-empty for hf_pretrained."
            )
        if config.tokenizer.bpb_mode not in {"off", "approx", "exact"}:
            raise ValueError("tokenizer.bpb_mode must be one of: off, approx, exact.")
    else:
        raise ValueError(
            "Unsupported tokenizer.name "
            f"'{config.tokenizer.name}'. Expected: bpe, hf_pretrained."
        )

    if config.model.name not in {
        "baseline_decoder",
        "ac_every_n_decoder",
        "sac_decoder",
    }:
        raise ValueError(
            f"Unsupported model.name '{config.model.name}'. "
            "Expected one of: baseline_decoder, ac_every_n_decoder, sac_decoder."
        )
    if config.model.d_model <= 0:
        raise ValueError("model.d_model must be > 0.")
    if config.model.n_heads <= 0:
        raise ValueError("model.n_heads must be > 0.")
    if config.model.layers <= 0:
        raise ValueError("model.layers must be > 0.")
    if not 0.0 <= config.model.dropout <= 1.0:
        raise ValueError("model.dropout must be in [0, 1].")
    if config.model.attention_impl not in {"basic", "sdpa"}:
        raise ValueError(
            "model.attention_impl must be one of: basic, sdpa."
        )
    if (
        config.model.name == "baseline_decoder"
        and config.model.norm_placement not in {"pre", "post"}
    ):
        raise ValueError("model.norm_placement must be one of: pre, post.")
    if config.model.d_model % config.model.n_heads != 0:
        raise ValueError(
            "model.d_model must be divisible by model.n_heads "
            f"(got d_model={config.model.d_model}, n_heads={config.model.n_heads})."
        )
    if (
        config.model.name == "ac_every_n_decoder"
        and config.model.checkpoint_every_n_layers <= 0
    ):
        raise ValueError("model.checkpoint_every_n_layers must be > 0.")

    if config.train.epochs is None and config.train.max_steps is None:
        raise ValueError(
            "train.epochs must be set when train.max_steps is not provided."
        )
    if config.train.epochs is not None and config.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0 when provided.")
    supported_optimizers = {"adam", "adamw", "sgd"}
    if config.train.optimizer.name not in supported_optimizers:
        raise ValueError(
            f"Unsupported train.optimizer.name '{config.train.optimizer.name}'. "
            "Expected one of: adam, adamw, sgd."
        )
    if config.train.optimizer.learning_rate <= 0:
        raise ValueError("train.optimizer.learning_rate must be > 0.")
    if config.train.optimizer.weight_decay < 0:
        raise ValueError("train.optimizer.weight_decay must be >= 0.")
    resolve_train_learning_rate(config.train)
    validate_train_lr_scheduler_config(config.train.lr_scheduler)
    if config.train.seq_len <= 0:
        raise ValueError("train.seq_len must be > 0.")
    if config.train.stride <= 0:
        raise ValueError("train.stride must be > 0.")
    if config.train.data_mode not in {"materialized", "streaming"}:
        raise ValueError("train.data_mode must be one of: materialized, streaming.")
    if not 0 < config.train.data_fraction <= 1:
        raise ValueError("train.data_fraction must be in (0, 1].")
    if config.train.max_steps is not None and config.train.max_steps <= 0:
        raise ValueError("train.max_steps must be > 0 when provided.")
    if config.train.max_steps is not None and config.train.data_mode != "streaming":
        raise ValueError("train.max_steps is only supported in streaming mode.")
    if not isinstance(config.train.run_validation, bool):
        raise ValueError("train.run_validation must be a bool.")

    if config.split.name == "holdout":
        if not 0 < config.split.train_fraction < 1:
            raise ValueError("split.train_fraction must be in (0, 1).")
    elif config.split.name != "pre_split":
        raise ValueError(
            f"Unsupported split.name '{config.split.name}'. "
            "Expected: holdout, pre_split."
        )

    if config.train.data_mode == "streaming":
        if config.dataset.name != "hf_text":
            raise ValueError(
                "train.data_mode='streaming' requires dataset.name='hf_text'."
            )
        if config.tokenizer.name != "hf_pretrained":
            raise ValueError(
                "train.data_mode='streaming' requires tokenizer.name='hf_pretrained'."
            )
        if config.split.name != "pre_split":
            raise ValueError(
                "train.data_mode='streaming' requires split.name='pre_split'."
            )
        if config.train.data_fraction != 1.0:
            raise ValueError(
                "train.data_fraction is not supported in streaming mode; use max_steps."
            )
        if config.train.run_validation and not config.dataset.validation_split:
            raise ValueError(
                "Streaming validation requires dataset.validation_split to be set."
            )
    elif config.split.name != "holdout":
        raise ValueError(
            "split.name='pre_split' is only supported when train.data_mode='streaming'."
        )

    if config.logging.provider not in {"console", "local", "wandb"}:
        raise ValueError(
            f"Unsupported logging.provider '{config.logging.provider}'. "
            "Expected one of: console, local, wandb."
        )
    if not isinstance(config.logging.enable_artifact_io, bool):
        raise ValueError("logging.enable_artifact_io must be a bool.")
    if config.logging.provider == "wandb" and (
        config.run.run_name is None or not config.run.run_name.strip()
    ):
        raise ValueError(
            "logging.provider='wandb' requires run.run_name to be set to a stable value."
        )

    wandb_cfg = config.logging.wandb
    bpb_metrics_enabled = (
        config.logging.provider in {"wandb", "local"} and wandb_cfg.enable_bits_per_byte
    )
    if wandb_cfg.log_every_n_steps <= 0:
        raise ValueError("logging.wandb.log_every_n_steps must be > 0.")
    if wandb_cfg.diagnostics_every_n_steps <= 0:
        raise ValueError("logging.wandb.diagnostics_every_n_steps must be > 0.")
    if wandb_cfg.layer_grad_norm_stride <= 0:
        raise ValueError("logging.wandb.layer_grad_norm_stride must be > 0.")
    if (
        wandb_cfg.layer_grad_norms_every_n_steps is not None
        and wandb_cfg.layer_grad_norms_every_n_steps <= 0
    ):
        raise ValueError(
            "logging.wandb.layer_grad_norms_every_n_steps must be > 0 when set."
        )
    if (
        wandb_cfg.parameter_optimizer_norms_every_n_steps is not None
        and wandb_cfg.parameter_optimizer_norms_every_n_steps <= 0
    ):
        raise ValueError(
            "logging.wandb.parameter_optimizer_norms_every_n_steps must be > 0 when set."
        )
    if wandb_cfg.val_every_n_steps < 0:
        raise ValueError("logging.wandb.val_every_n_steps must be >= 0.")
    if wandb_cfg.attention_entropy_every_n_steps <= 0:
        raise ValueError("logging.wandb.attention_entropy_every_n_steps must be > 0.")
    if wandb_cfg.attention_entropy_head_cap <= 0:
        raise ValueError("logging.wandb.attention_entropy_head_cap must be > 0.")
    if wandb_cfg.attention_entropy_token_cap <= 0:
        raise ValueError("logging.wandb.attention_entropy_token_cap must be > 0.")
    if (
        bpb_metrics_enabled
        and config.tokenizer.name == "hf_pretrained"
        and config.tokenizer.bpb_mode == "exact"
        and not config.tokenizer.use_fast
    ):
        raise ValueError(
            "tokenizer.bpb_mode='exact' requires tokenizer.use_fast=True when "
            "logging.wandb.enable_bits_per_byte is enabled."
        )
