import torch
import torch.nn.functional as F

from .common import build_attention_mask, reshape_for_multi_head
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
        self._forward_metric_entropy_capture = None

    def forward(
        self,
        Q: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = True,
    ):
        batch_size, seq_len, d_model = Q.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected input last dim {self.d_model}, got {d_model}"
            )

        all_projs = self.packed_proj(Q)
        capture_entropy = getattr(self, "_forward_metric_entropy_capture", None)
        if callable(capture_entropy):
            capture_entropy(
                all_projs=all_projs,
                mask=mask,
                key_padding_mask=key_padding_mask,
                is_causal=bool(is_causal),
                n_heads=int(self.n_heads),
                head_dim=int(self.head_dim),
            )
        query, key, value = torch.chunk(all_projs, 3, dim=-1)

        query = reshape_for_multi_head(
            query,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        key = reshape_for_multi_head(
            key,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        value = reshape_for_multi_head(
            value,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )

        attn_mask = None
        sdpa_is_causal = bool(is_causal and mask is None and key_padding_mask is None)
        if not sdpa_is_causal or mask is not None or key_padding_mask is not None:
            attn_mask = build_attention_mask(
                batch_size=batch_size,
                seq_len=seq_len,
                device=query.device,
                is_causal=is_causal,
                mask=mask,
                key_padding_mask=key_padding_mask,
            )
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
        
