"""
BASELINE MODEL

First I will try to get the original decoder model as used in Vaswani et al. and then modularize it
and improve it along the way.
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

# set random seed for reproducibility
torch.manual_seed(42)


# ===============================================================
#               ACTIVATION FUNCTIONS
# ===============================================================

class SoftmaxActivation(nn.Module):
    def __init__(self, dim=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, x):
        exp_x = torch.exp(
            x - torch.max(x, dim=self.dim, keepdim=True).values
        )  # numerical stability
        sum_exp = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_exp


class ReluActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))


# ===============================================================
#               NORM DEFINITIONS
# ===============================================================

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(size=(dim,)))
        self.beta = torch.nn.Parameter(torch.zeros(size=(dim,)))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        # giving the rescaling ability back to the model with beta and gamma parameters
        return self.gamma * norm + self.beta


# ===============================================================
#               LAYER DEFINITIONS
# ===============================================================

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.W = torch.nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W)
        
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(torch.zeros(size=(out_dim,)))

    def forward(self, x):
        res = x @ self.W
        if self.bias:
            return res + self.b
        else:
            return res

class DropoutLayer(nn.Module):
    def __init__(self, p = 0, **kwargs):
        super().__init__(**kwargs)
        self.p = p
    
    def forward(self, x):
        if not self.training == True or self.p == 0:
            return x
        keep = 1 - self.p
        mask = (torch.rand_like(x,device=x.device) < keep).to(x.dtype)
        return x * mask/keep

class EmbeddingLayer(nn.Module):
    """
    I think of embedding here as essentially a projection from a "VOCAB" sized space to a "d_model" sized space.
    That is nothing but a lookup table of "VOCAB" sized keys with each value being "d_model" sized when the you look up a token index, 
    you get back its corresponding embedding vector of size "d_model".
    But its also worth remembering that gradients flow through the embedding layer.
    i.e: The index's which we selected receive grads during backprop.
    """
    def __init__(self, key_size, embedding_size, pad_idx = None, **kwargs):
        super().__init__(**kwargs)
        self.key_size = key_size
        self.embedding_size = embedding_size
        self.padding_idx = pad_idx
        self.embedding_table = torch.nn.Parameter(
            torch.empty(size=(self.key_size, self.embedding_size))
        )
        
        nn.init.normal_(self.embedding_table)
        
        # We zero out padding index cause we dont want grads updating padding representation.
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding_table[self.padding_idx].zero_()
            
    def forward(self, idx: torch.Tensor):
        # idx is of shape [batch, seq_len] and contains integer token indices.
        idx = idx.long()
        out = self.embedding_table[idx]
        
        if self.padding_idx is not None:
            # Ensure padding outputs are exactly zero (even if something tried to change them)
            mask = (idx == self.padding_idx).unsqueeze(-1)  # (..., 1)
            out = out.masked_fill(mask, 0.0)

        return out
        
    
# ===============================================================
#              ATTENTION DEFINITIONS
# ===============================================================

class BasicMultiHeadSelfAttention(nn.Module):
    """
    This should be easy to modify and support cross attention but thats not in my current scope atm.
    """

    def __init__(self, E_q, E_out, n_heads, E_bias: bool, **kwargs):
        """
        Shadowing the MHA implementation here:
        https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html#introducing-the-building-blocks

        But focused only on the self-attn decoder for LM modelling.
        """
        super().__init__(**kwargs)
        self.E_q = E_q
        self.E_out = E_out
        self.n_heads = n_heads

        assert self.E_q % self.n_heads == 0 , f"Embedding dimension {self.E_q} must be divisible by number of heads {self.n_heads}"

        self.head_dim = self.E_q // self.n_heads

        # self.same_qkv = False if implementing cross attn use a flag like this and branch to implement cross-attn.
        # in self-attn all are equal then we can use one large linear layer instead of 3 projections and then split em up.
        self.same_qkv = True
        self.packed_proj = LinearLayer(in_dim=E_q, out_dim=E_q * 3, bias=E_bias)
        self.out_proj = LinearLayer(in_dim=E_q, out_dim=E_out, bias=E_bias)
        self.softmax = SoftmaxActivation(dim=-1)

    def forward(
        self,
        Q: torch.Tensor,
        mask: torch.Tensor = None,
        scale: torch.Tensor = None,
        is_causal: bool = True,
    ):
        """
        is_causal: A boolean flag to indicate whether to apply a causal mask or not.
            if is_causal is True, we will apply a causal mask to the attention scores to prevent attending to future tokens and `mask` will be ignored.
            If False, we will use the provided `mask` to mask out certain tokens as per the use case.
        mask: [Optional] A tensor of shape (seq_len, seq_len) where mask[i,j] = 0 indicates that the j-th token should not be attended to when processing the i-th token.
        """
        # Q, K, V are of shape (batch_size, seq_len, d_model)

        # Linearly Project the inputs as per the defined embedding dims for attention
        all_projs = self.packed_proj(Q)  # Assuming this is just self attention.
        Q, K, V = torch.chunk(all_projs, 3, dim=-1)

        # Reshape for multi-heads
        batch_size, seq_len, d_model = Q.shape  # E_q and d_model should be the same
        assert d_model == self.E_q

        # [batch, seq_len, d_model] --reshape--> [batch, seq_len, n_heads * head_dim] ---transpose--> [batch, n_heads, seq_len, head_dim]
        Q_headed = Q.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        K_headed = K.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        V_headed = V.reshape(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        if scale is None:
            scale = math.sqrt(self.head_dim)

        scores = (Q_headed @ K_headed.transpose(-2, -1)) / scale

        if is_causal:
            # Create causal mask [seq_len, seq_len] - will broadcast to [batch, n_heads, seq_len, seq_len]
            # Lower triangle = 1 (attend), upper triangle = 0 (mask out future tokens)
            mask = torch.tril(
                torch.ones([seq_len, seq_len],device=scores.device), diagonal=0,
            )
            scores = scores.masked_fill(mask == 0, float("-inf"))
        else:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = self.softmax(scores)

        attn = scores @ V_headed

        joined_heads = attn.transpose(1, 2).reshape(
            batch_size, seq_len, d_model
        )  # Concat heads back together, n_heads * head_dim = d_model

        output_projection = self.out_proj(joined_heads)

        return output_projection


# ===============================================================
#             DECODER BLOCK DEFINITIONS
# ===============================================================

class SelfAttnDecoder(nn.Module):
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

    def forward(self, Q: torch.Tensor):
        """
        The Decoder consists of 2 blocks:
        1. Attention Block
        2. Linear Block
        """
        # self attn
        attention = self.multi_head_attention(Q,is_causal=True)
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