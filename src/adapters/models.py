from __future__ import annotations

import torch

from src.core.config import (
    ACEveryNDecoderConfig,
    BaselineDecoderConfig,
    SACDecoderConfig,
)
from src.core.registry import register_model_adapter
from src.core.types import SpecialTokenIds, VocabInfo
from src.components.models.baseline_model import BaselineModel
from src.components.models.baseline_with_AC_model import ACEveryN_DecoderModel
from src.components.models.baseline_with_SAC_model import SelectiveAC_DecoderModel


class BaselineDecoderModelAdapter:
    def build(
        self,
        cfg: BaselineDecoderConfig,
        vocab: VocabInfo,
        special: SpecialTokenIds,
    ) -> torch.nn.Module:
        return BaselineModel(
            vocab_size=vocab.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            layers=cfg.layers,
            dropout=cfg.dropout,
            pad_id=special.pad_id,
        )


class ACEveryNDecoderModelAdapter:
    def build(
        self,
        cfg: ACEveryNDecoderConfig,
        vocab: VocabInfo,
        special: SpecialTokenIds,
    ) -> torch.nn.Module:
        return ACEveryN_DecoderModel(
            vocab_size=vocab.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            layers=cfg.layers,
            dropout=cfg.dropout,
            pad_id=special.pad_id,
            checkpoint_every_n_layers=cfg.checkpoint_every_n_layers,
            use_activation_checkpointing=True,
        )


class SACDecoderModelAdapter:
    def build(
        self,
        cfg: SACDecoderConfig,
        vocab: VocabInfo,
        special: SpecialTokenIds,
    ) -> torch.nn.Module:
        return SelectiveAC_DecoderModel(
            vocab_size=vocab.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            layers=cfg.layers,
            dropout=cfg.dropout,
            pad_id=special.pad_id,
        )


def register_model_adapters() -> None:
    register_model_adapter("baseline_decoder", BaselineDecoderModelAdapter())
    register_model_adapter("ac_every_n_decoder", ACEveryNDecoderModelAdapter())
    register_model_adapter("sac_decoder", SACDecoderModelAdapter())
