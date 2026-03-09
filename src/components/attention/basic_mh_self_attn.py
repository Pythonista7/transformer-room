import math
from torch import nn
import torch

from ..primitive.layers import LinearLayer, SoftmaxActivation


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

        # scores shape = Qh_shape_[B, H, T, D/H ] @ Kh_shape_[B, H, D/H, T] => [B,H,T,T] 
        scores = (Q_headed @ K_headed.transpose(-2, -1)) / scale

        attention_mask = torch.ones(
            (batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=scores.device
        )

        if is_causal:
            # Lower triangle = attend, upper triangle = future and must be masked.
            causal_mask = torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device),
                diagonal=0,
            )
            attention_mask = attention_mask & causal_mask.unsqueeze(0).unsqueeze(0)
        elif mask is not None:
            user_mask = mask.to(device=scores.device, dtype=torch.bool)
            if user_mask.dim() == 2:
                user_mask = user_mask.unsqueeze(0).unsqueeze(0)
            elif user_mask.dim() == 3:
                user_mask = user_mask.unsqueeze(1)
            elif user_mask.dim() != 4:
                raise ValueError(
                    f"Unsupported mask dim {user_mask.dim()} for attention mask."
                )
            attention_mask = attention_mask & user_mask

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"key_padding_mask must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            # Broadcast key validity over attention heads and query positions.
            key_mask = key_padding_mask.to(device=scores.device, dtype=torch.bool) # shape = [batch_size, seq_len]
            attention_mask = attention_mask & key_mask.unsqueeze(1).unsqueeze(1)

        # mask scores with -inf
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        
        # Avoid NaNs when a query row is fully masked.i.e:
        # if a query has no valid keys:
        #     replace that entire row with zeros
        fully_masked = ~attention_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)

        scores = self.softmax(scores) # shape = [B,H,T,T] each query attends to all keys with some distribution over them.

        attn = scores @ V_headed # shape = [B,H,T,T] @ [B,H,T,D/H] => [B,H,T,D/H]

        # Now join the heads back together and project the output back to the desired output embedding dimension.
        # Shape before joining heads: [batch, n_heads, seq_len, head_dim]
        # After joining heads: [batch, seq_len, n_heads * head_dim] = [batch, seq_len, d_model]
        joined_heads = attn.transpose(1, 2).reshape(
            batch_size, seq_len, d_model
        )  # Concat heads back together, n_heads * head_dim = d_model

        output_projection = self.out_proj(joined_heads)

        return output_projection
