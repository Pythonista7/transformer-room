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
        
        nn.init.normal_(self.embedding_table,std=1/math.sqrt(self.embedding_size))
        
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
        