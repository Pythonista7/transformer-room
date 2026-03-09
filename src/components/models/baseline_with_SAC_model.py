from functools import partial
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
)

from ..blocks.self_attn_decoder_block import BasicSelfAttnDecoder
from ..positional.positional_encoder import SinusoidalPositionalEncoder as PositionalEncoder
from ..primitive.layers import EmbeddingLayer, LinearLayer

aten = torch.ops.aten
OPS_TO_SAVE = {
    aten.mm.default, # matmul
    aten.bmm.default,# batch matmul
    aten.addmm.default, # linear layer (bias + matmul)
}

def policy_fn(ctx, op, *args, **kwargs):
    if op in OPS_TO_SAVE:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

context_fn = partial(create_selective_checkpoint_contexts, policy_fn)


class SelectiveAC_DecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        layers,
        d_model,
        n_heads,
        pad_id=None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.pad_id = pad_id
        self.embedding_layer = EmbeddingLayer(
            key_size=vocab_size, embedding_size=d_model, pad_idx=pad_id
        )
        self.pos_encoding = PositionalEncoder(d_model=d_model) 
        
        self.layer_count = layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.dec_layers: List[BasicSelfAttnDecoder] = torch.nn.ModuleList(
            [
                BasicSelfAttnDecoder(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(self.layer_count)
            ]
        )
        self.output_proj = LinearLayer(d_model, vocab_size) # Projecting back to vocab size for prediction.

    def _forward_impl(self, inputs, key_padding_mask=None):
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

    def forward(self, inputs, key_padding_mask=None):
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                self._forward_impl,
                inputs,
                key_padding_mask,
                context_fn=context_fn,
                use_reentrant=False,  # Important for correct gradient computation with selective checkpointing
            )
        return self._forward_impl(inputs, key_padding_mask)