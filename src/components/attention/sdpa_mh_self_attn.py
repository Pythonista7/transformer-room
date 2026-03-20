import torch
import torch.nn.functional as F

from ..primitive.layers import LinearLayer


class SDPASelfAttn(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        if d_model % n_heads != 0:
            raise ValueError(
                f"Embedding dimension {d_model} must be divisible by number of heads {n_heads}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.p = dropout

        # Same projection layout as the basic attention variant.
        self.packed_proj = LinearLayer(in_dim=d_model, out_dim=d_model * 3, bias=True)
        self.out_proj = LinearLayer(in_dim=d_model, out_dim=d_model, bias=True)

    def forward(
        self,
        Q: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = True,
    ):
        batch_size, seq_len, d_model = Q.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected input last dim {self.d_model}, got {d_model}"
            )

        all_projs = self.packed_proj(Q)
        query, key, value = torch.chunk(all_projs, 3, dim=-1)

        query = query.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        sdpa_is_causal = is_causal

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"key_padding_mask must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )

            key_mask = key_padding_mask.to(device=query.device, dtype=torch.bool)
            key_mask = key_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
            attn_mask = key_mask

            # SDPA does not allow attn_mask + is_causal=True together, so combine here.
            if is_causal:
                causal_mask = torch.tril(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=query.device),
                    diagonal=0,
                ).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                attn_mask = attn_mask & causal_mask
                sdpa_is_causal = False

        attention = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.p if self.training else 0.0,
            is_causal=sdpa_is_causal,
        )

        joined_heads = attention.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(joined_heads)
        
