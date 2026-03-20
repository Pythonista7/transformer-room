
from torch import nn
import torch

from src.components.primitive.layers import DropoutLayer, LayerNorm, LinearLayer, ReluActivation
from ..attention import SDPASelfAttn

class SDPASelfAttnDecoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.multi_head_attention = SDPASelfAttn(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
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
            Q, key_padding_mask=key_padding_mask, is_causal= True
        )
        
        # Linear Block
        # Add & Norm
        x = self.ln1(Q + attention)
        # Linear Block
        linear_out = self.linear2(self.relu(self.linear1(x)))
        linear_out = self.linear_dropout(linear_out)
        
        # Add & Norm
        out = self.ln2(x + linear_out)
        return out
