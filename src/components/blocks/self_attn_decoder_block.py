from __future__ import annotations

import torch
from torch import nn

from src.core.config import NormPlacement

from ..attention import build_self_attention
from ..primitive.layers import DropoutLayer, LinearLayer, LayerNorm, ReluActivation



class SelfAttnDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        attention_impl: str = "basic",
        norm_placement: NormPlacement = "post",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_placement = norm_placement
        self.attention_impl = attention_impl
        self.multi_head_attention = build_self_attention(
            attention_impl,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.linear1 = LinearLayer(d_model, d_model * 4)
        self.relu = ReluActivation()
        self.linear2 = LinearLayer(d_model * 4, d_model)
        self.linear_dropout = DropoutLayer(p=dropout)

    def forward(self, Q: torch.Tensor, key_padding_mask: torch.Tensor = None):
        if self.norm_placement == "post": # Post-Norm
            attention = self.multi_head_attention(
                Q,
                is_causal=True,
                key_padding_mask=key_padding_mask,
            )

            x = self.ln1(Q + attention)
            linear_out = self.linear2(self.relu(self.linear1(x)))
            linear_out = self.linear_dropout(linear_out)
            return self.ln2(x + linear_out)

        if self.norm_placement == "pre": # Pre-Norm
            Q_norm = self.ln1(Q)

            attention = self.multi_head_attention(
                Q_norm,
                is_causal=True,
                key_padding_mask=key_padding_mask,
            )

            x = Q + attention

            linear_out = self.linear2(self.relu(self.linear1(self.ln2(x))))
            linear_out = self.linear_dropout(linear_out)

            return x + linear_out

        raise ValueError(
            "norm_placement must be one of: pre, post "
            f"(got {self.norm_placement!r})."
        )
