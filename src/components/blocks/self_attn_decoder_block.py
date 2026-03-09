
# ===============================================================
#             DECODER BLOCK DEFINITIONS
# ===============================================================

import torch
from torch import nn

from ..attention.basic_mh_self_attn import BasicMultiHeadSelfAttention
from ..primitive.layers import DropoutLayer, LinearLayer, LayerNorm, ReluActivation


class BasicSelfAttnDecoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        """
        NOTE: I've currently skipped "padding key mask" in the MHA block since i wanted to use the tested/jagged tensors, 
        need to do this when creating embedding/processing training data!
        """
        super().__init__(**kwargs)

        self.multi_head_attention = BasicMultiHeadSelfAttention(
            E_q=d_model, E_out=d_model, n_heads=n_heads, E_bias=True
        )
        self.attn_dropout = DropoutLayer(p=dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.linear1 = LinearLayer(d_model, d_model * 4)
        self.relu = ReluActivation()
        self.linear2 = LinearLayer(d_model * 4, d_model)
        self.linear_dropout = DropoutLayer(p=dropout)

    def forward(self, Q: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        The Decoder consists of 2 blocks:
        1. Attention Block
        2. Linear Block
        """
        # self attn
        attention = self.multi_head_attention(
            Q, is_causal=True, key_padding_mask=key_padding_mask
        )
        # apply dropout
        attention = self.attn_dropout(attention)
        
        # Linear Block
        # Add & Norm
        x = self.ln1(Q + attention)
        # Linear Block
        linear_out = self.linear2(self.relu(self.linear1(x)))
        linear_out = self.linear_dropout(linear_out)
        
        # Add & Norm
        out = self.ln2(x + linear_out)
        return out
