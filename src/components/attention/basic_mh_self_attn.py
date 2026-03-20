from torch import nn
import torch

from .common import attention_scale, build_attention_mask, reshape_for_multi_head
from ..primitive.layers import DropoutLayer, LinearLayer, SoftmaxActivation


class BasicMultiHeadSelfAttention(nn.Module):
    """
    This should be easy to modify and support cross attention but thats not in my current scope atm.
    """

    def __init__(self, E_q, E_out, n_heads, E_bias: bool, dropout: float = 0.0, **kwargs):
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
        self.attn_dropout = DropoutLayer(p=dropout)

    def forward(
        self,
        Q: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
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
        Q_headed = reshape_for_multi_head(
            Q,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        K_headed = reshape_for_multi_head(
            K,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        V_headed = reshape_for_multi_head(
            V,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )

        if scale is None:
            scale = attention_scale(self.head_dim)

        # scores shape = Qh_shape_[B, H, T, D/H ] @ Kh_shape_[B, H, D/H, T] => [B,H,T,T] 
        scores = (Q_headed @ K_headed.transpose(-2, -1)) / scale

        attention_mask = build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=scores.device,
            is_causal=is_causal,
            mask=mask,
            key_padding_mask=key_padding_mask,
        )

        # mask scores with -inf
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        
        # Avoid NaNs when a query row is fully masked.i.e:
        # if a query has no valid keys:
        #     replace that entire row with zeros
        fully_masked = ~attention_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)

        scores = self.softmax(scores) # shape = [B,H,T,T] each query attends to all keys with some distribution over them.
        scores = self.attn_dropout(scores)

        attn = scores @ V_headed # shape = [B,H,T,T] @ [B,H,T,D/H] => [B,H,T,D/H]

        # Now join the heads back together and project the output back to the desired output embedding dimension.
        # Shape before joining heads: [batch, n_heads, seq_len, head_dim]
        # After joining heads: [batch, seq_len, n_heads * head_dim] = [batch, seq_len, d_model]
        joined_heads = attn.transpose(1, 2).reshape(
            batch_size, seq_len, d_model
        )  # Concat heads back together, n_heads * head_dim = d_model

        output_projection = self.out_proj(joined_heads)

        return output_projection
