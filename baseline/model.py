from typing import List

import torch
import torch.nn as nn

from .layers import EmbeddingLayer, LinearLayer, SelfAttnDecoder
from .positional_encoder import PositionalEncoder


class BaselineModel(nn.Module):
    def __init__(self, vocab_size, layers, d_model, n_heads, pad_id=None, **kwargs):
        super().__init__(**kwargs)
        
        self.pad_id = pad_id
        self.embedding_layer = EmbeddingLayer(
            key_size=vocab_size, embedding_size=d_model, pad_idx=pad_id
        )
        self.pos_encoding = PositionalEncoder(d_model=d_model) 
        
        self.layer_count = layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dec_layers: List[SelfAttnDecoder] = torch.nn.ModuleList(
            [
                SelfAttnDecoder(d_model=d_model, n_heads=n_heads)
                for _ in range(self.layer_count)
            ]
        )
        self.output_proj = LinearLayer(d_model, vocab_size) # Projecting back to vocab size for prediction.

    def forward(self, inputs, key_padding_mask=None):
        # inputs of shape [batch, tokens]
        # Convert tokens into embeddings [batch,tokens,d_embed]
        embeddings = self.embedding_layer(inputs)
        
        # Add positional encodings to the embeddings [batch,tokens,d_embed]
        x = self.pos_encoding(embeddings) # this does embedding + positional encoding and returns the result.
        
        # Decoder Stack
        for layer in self.dec_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        # Final output of shape [batch,tokens,d_model]
        # Now project back the d_model output to the vocab size for prediction. This can be done with a linear layer.
        out = self.output_proj(x)
        return out
