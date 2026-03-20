from __future__ import annotations

import math

import torch


def attention_scale(head_dim: int) -> float:
    return math.sqrt(head_dim)


def reshape_for_multi_head(
    x: torch.Tensor,
    *,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    return x.reshape(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)


def build_attention_mask(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    is_causal: bool,
    mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    attention_mask = torch.ones(
        (batch_size, 1, seq_len, seq_len),
        dtype=torch.bool,
        device=device,
    )

    if is_causal:
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=0,
        )
        attention_mask = attention_mask & causal_mask.unsqueeze(0).unsqueeze(0)
    elif mask is not None:
        user_mask = mask.to(device=device, dtype=torch.bool)
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
        key_mask = key_padding_mask.to(device=device, dtype=torch.bool)
        attention_mask = attention_mask & key_mask.unsqueeze(1).unsqueeze(1)

    return attention_mask
