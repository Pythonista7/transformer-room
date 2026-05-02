from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import islice

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .runtime import get_autocast_context


def _unpack_batch_tensors(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 3:
        input_seq, target_seq, key_padding_mask = batch
        return input_seq, target_seq, key_padding_mask, None
    if len(batch) == 4:
        input_seq, target_seq, key_padding_mask, target_byte_lengths = batch
        return input_seq, target_seq, key_padding_mask, target_byte_lengths
    raise ValueError(
        "Expected batch with 3 or 4 tensors: "
        "(input_seq, target_seq, key_padding_mask[, target_byte_lengths])."
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CrossEntropyLoss,
    device: torch.device,
    use_bf16: bool,
    token_byte_lengths: Sequence[int] | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    if max_eval_batches is not None and int(max_eval_batches) <= 0:
        raise ValueError("max_eval_batches must be > 0 when provided.")

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
        batch_iter = (
            iter(loader)
            if max_eval_batches is None
            else islice(loader, int(max_eval_batches))
        )
        for batch in batch_iter:
            (
                input_seq,
                target_seq,
                key_padding_mask,
                target_byte_lengths,
            ) = _unpack_batch_tensors(batch)
            input_seq = input_seq.to(device, non_blocking=non_blocking)
            target_seq = target_seq.to(device, non_blocking=non_blocking)
            key_padding_mask = key_padding_mask.to(device, non_blocking=non_blocking)
            if target_byte_lengths is not None:
                target_byte_lengths = target_byte_lengths.to(
                    device,
                    non_blocking=non_blocking,
                )

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

            if target_byte_lengths is not None:
                batch_bytes = int(
                    target_byte_lengths[valid_target_mask].sum().item()
                )
                total_bytes += batch_bytes
            elif token_byte_lengths_tensor is not None:
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
