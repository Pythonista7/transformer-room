from .basic_mh_self_attn import BasicMultiHeadSelfAttention
from .sdpa_mh_self_attn import SDPASelfAttn


def build_self_attention(
    attention_impl: str,
    *,
    d_model: int,
    n_heads: int,
    dropout: float = 0.0,
):
    if attention_impl == "basic":
        return BasicMultiHeadSelfAttention(
            E_q=d_model,
            E_out=d_model,
            n_heads=n_heads,
            E_bias=True,
            dropout=dropout,
        )
    if attention_impl == "sdpa":
        return SDPASelfAttn(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
    raise ValueError(
        f"Unsupported attention_impl '{attention_impl}'. Expected one of: basic, sdpa."
    )


__all__ = ["BasicMultiHeadSelfAttention", "SDPASelfAttn", "build_self_attention"]
