from __future__ import annotations

import torch

from baseline.core.config import BaselineDecoderConfig
from baseline.core.registry import register_model_adapter
from baseline.core.types import SpecialTokenIds, VocabInfo
from baseline.model import BaselineModel


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


register_model_adapter("baseline_decoder", BaselineDecoderModelAdapter())
