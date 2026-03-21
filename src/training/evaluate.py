from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .runtime import get_autocast_context


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CrossEntropyLoss,
    device: torch.device,
    use_bf16: bool,
    token_byte_lengths: Sequence[int] | None = None,
) -> dict[str, float]:
    model.eval()
    pad_id = int(loss_fn.ignore_index)
    total_tokens = 0
    total_bytes = 0
    total_loss = 0.0
    non_blocking = device.type == "cuda"
    token_byte_lengths_tensor: torch.Tensor | None = None
    if token_byte_lengths is not None:
        token_byte_lengths_tensor = torch.tensor(
            token_byte_lengths,
            dtype=torch.long,
            device=device,
        )

    with torch.no_grad():
        for input_seq, target_seq, key_padding_mask in loader:
            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)
            key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)

            with get_autocast_context(device=device, use_bf16=use_bf16):
                output = model(input_seq, key_padding_mask=key_padding_mask)
                loss_sum = loss_fn(
                    output.reshape(-1, output.size(-1)),
                    target_seq.reshape(-1),
                )

            valid_target_mask = target_seq != pad_id
            tokens = int(valid_target_mask.sum().item())
            if tokens == 0:
                continue

            if token_byte_lengths_tensor is not None:
                valid_target_ids = target_seq[valid_target_mask]
                batch_bytes = int(
                    token_byte_lengths_tensor[valid_target_ids].sum().item()
                )
                total_bytes += batch_bytes

            total_tokens += tokens
            total_loss += loss_sum.item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    bits_per_byte = float("nan")
    if total_bytes > 0:
        bits_per_byte = float((total_loss / math.log(2.0)) / float(total_bytes))

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_bits_per_byte": bits_per_byte,
    }
