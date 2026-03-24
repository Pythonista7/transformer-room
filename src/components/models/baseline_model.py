from typing import List

import torch
import torch.nn as nn

from ..blocks.self_attn_decoder_block import SelfAttnDecoderBlock
from ..positional.positional_encoder import SinusoidalPositionalEncoder as PositionalEncoder
from ..primitive.layers import EmbeddingLayer, LayerNorm, LinearLayer
from src.core.config import NormPlacement


class BaselineModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        layers,
        d_model,
        n_heads,
        pad_id=None,
        dropout=0.1,
        attention_impl: str = "basic",
        norm_placement: NormPlacement = "post",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_placement = norm_placement
        if self.norm_placement not in {"pre", "post"}:
            raise ValueError(
                "norm_placement must be one of: pre, post "
                f"(got {self.norm_placement!r})."
            )
        self.pad_id = pad_id
        self.embedding_layer = EmbeddingLayer(
            key_size=vocab_size, embedding_size=d_model, pad_idx=pad_id
        )
        self.pos_encoding = PositionalEncoder(d_model=d_model) 
        
        self.layer_count = layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_impl = attention_impl
        self.dec_layers: List[SelfAttnDecoderBlock] = torch.nn.ModuleList(
            [
                SelfAttnDecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    attention_impl=attention_impl,
                    norm_placement=self.norm_placement
                )
                for _ in range(self.layer_count)
            ]
        )
        self.output_proj = LinearLayer(d_model, vocab_size) # Projecting back to vocab size for prediction.
        if self.norm_placement == "pre":
            self.final_ln = LayerNorm(d_model)
        
    def forward(self, inputs, key_padding_mask=None):
        # inputs of shape [batch, tokens]
        # Convert tokens into embeddings [batch,tokens,d_embed]
        embeddings = self.embedding_layer(inputs)
        
        # Add positional encodings to the embeddings [batch,tokens,d_embed]
        x = self.pos_encoding(embeddings) # this does embedding + positional encoding and returns the result.
        
        # Decoder Stack
        for layer in self.dec_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        if self.norm_placement == "pre":
            x = self.final_ln(x)
        # Final output of shape [batch,tokens,d_model]
        # Now project back the d_model output to the vocab size for prediction. This can be done with a linear layer.
        out = self.output_proj(x)
        
        return out
