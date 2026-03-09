from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..blocks.self_attn_decoder_block import BasicSelfAttnDecoder
from ..positional.positional_encoder import SinusoidalPositionalEncoder as PositionalEncoder
from ..primitive.layers import EmbeddingLayer, LinearLayer


class ACEveryN_DecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        layers,
        d_model,
        n_heads,
        pad_id=None,
        dropout=0.1,
        use_activation_checkpointing=True,
        checkpoint_every_n_layers=1,
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
        self.use_activation_checkpointing = use_activation_checkpointing
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        if self.checkpoint_every_n_layers <= 0:
            raise ValueError(
                "checkpoint_every_n_layers must be > 0, got "
                f"{self.checkpoint_every_n_layers}"
            )
        self.dec_layers: List[BasicSelfAttnDecoder] = torch.nn.ModuleList(
            [
                BasicSelfAttnDecoder(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(self.layer_count)
            ]
        )
        self.output_proj = LinearLayer(d_model, vocab_size) # Projecting back to vocab size for prediction.

    def _forward_decoder_layer(
        self,
        layer: BasicSelfAttnDecoder,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return layer(x, key_padding_mask=key_padding_mask)

    def forward(self, inputs, key_padding_mask=None):
        # inputs of shape [batch, tokens]
        # Convert tokens into embeddings [batch,tokens,d_embed]
        embeddings = self.embedding_layer(inputs)
        
        # Add positional encodings to the embeddings [batch,tokens,d_embed]
        x = self.pos_encoding(embeddings) # this does embedding + positional encoding and returns the result.
        
        # Decoder Stack
        for idx, layer in enumerate(self.dec_layers):
            should_checkpoint = (
                self.use_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and (idx % self.checkpoint_every_n_layers == 0)
            )

            if should_checkpoint:
                x = checkpoint(
                    lambda hidden_states, layer_module=layer: self._forward_decoder_layer(
                        layer_module,
                        hidden_states,
                        key_padding_mask=key_padding_mask,
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = self._forward_decoder_layer(
                    layer,
                    x,
                    key_padding_mask=key_padding_mask,
                )
        
        # Final output of shape [batch,tokens,d_model]
        # Now project back the d_model output to the vocab size for prediction. This can be done with a linear layer.
        out = self.output_proj(x)
        return out