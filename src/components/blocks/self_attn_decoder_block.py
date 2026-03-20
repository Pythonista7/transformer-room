from __future__ import annotations

import torch
from torch import nn

from ..attention import build_self_attention
from ..primitive.layers import DropoutLayer, LinearLayer, LayerNorm, ReluActivation


class SelfAttnDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        attention_impl: str = "basic",
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        attention = self.multi_head_attention(
            Q,
            is_causal=True,
            key_padding_mask=key_padding_mask,
        )

        x = self.ln1(Q + attention)
        linear_out = self.linear2(self.relu(self.linear1(x)))
        linear_out = self.linear_dropout(linear_out)
        return self.ln2(x + linear_out)
