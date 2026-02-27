"""
python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --top-k 20 \
  --plot-dir baseline/models/wikitext2_gpt2_v1/memviz \
  --json-out baseline/models/wikitext2_gpt2_v1/memviz/summary.json
"""
from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from baseline.model import BaselineModel


FLOAT_FP32_BYTES = 4
FLOAT_BF16_BYTES = 2
BOOL_BYTES = 1


@dataclass(slots=True)
class ShapeConfig:
    batch_size: int = 64
    seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 8
    layers: int = 12
    base_vocab_size: int = 33280
    num_special_tokens: int = 3

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + self.num_special_tokens


@dataclass(slots=True)
class TensorRecord:
    bucket: str
    name: str
    shape: tuple[int, ...]
    numel: int
    dtype_kind: str  # "float" | "bool"


@dataclass(slots=True)
class StageMemory:
    stage: str
    allocated_gib: float
    reserved_gib: float
    delta_allocated_gib: float
    peak_allocated_gib: float
    peak_reserved_gib: float


def _to_gib(numel: int, bytes_per_elem: int) -> float:
    return (numel * bytes_per_elem) / (1024**3)


def _bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def _shape_str(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def _bar(value: float, max_value: float, width: int = 40) -> str:
    if max_value <= 0:
        return ""
    filled = int(round(width * (value / max_value)))
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _unwrap_value(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    maybe = payload.get("value")
    if isinstance(maybe, dict):
        return maybe
    return payload


def load_shape_config_from_run_config(path: Path) -> ShapeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_cfg = _unwrap_value(payload.get("model"))
    train_cfg = _unwrap_value(payload.get("train"))
    tok_cfg = _unwrap_value(payload.get("tokenizer"))

    return ShapeConfig(
        batch_size=int(train_cfg.get("batch_size", 64)),
        seq_len=int(train_cfg.get("seq_len", 1024)),
        d_model=int(model_cfg.get("d_model", 768)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        layers=int(model_cfg.get("layers", 12)),
        base_vocab_size=int(tok_cfg.get("base_vocab_size", 33280)),
        num_special_tokens=int(tok_cfg.get("num_special_tokens", 3)),
    )


def collect_meta_records(cfg: ShapeConfig) -> list[TensorRecord]:
    records: list[TensorRecord] = []

    def push(bucket: str, name: str, x: torch.Tensor, dtype_kind: str = "float") -> None:
        records.append(
            TensorRecord(
                bucket=bucket,
                name=name,
                shape=tuple(int(s) for s in x.shape),
                numel=int(x.numel()),
                dtype_kind=dtype_kind,
            )
        )

    def mk_fwd(bucket: str, name: str):
        def _hook(_mod, _inp, out):
            if torch.is_tensor(out):
                push(bucket, name, out, dtype_kind="float")

        return _hook

    def mk_pre(bucket: str, name: str):
        def _hook(_mod, inp):
            if inp and torch.is_tensor(inp[0]):
                push(bucket, name, inp[0], dtype_kind="float")

        return _hook

    with torch.device("meta"):
        model = BaselineModel(
            vocab_size=cfg.vocab_size,
            layers=cfg.layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            pad_id=cfg.vocab_size - 2,
        )

    handles = []
    for name, mod in model.named_modules():
        if name.endswith("multi_head_attention.packed_proj"):
            handles.append(mod.register_forward_hook(mk_fwd("input-projects", name)))
        elif name.endswith("multi_head_attention.softmax"):
            handles.append(
                mod.register_forward_pre_hook(
                    mk_pre("per-layer-attn", f"{name}.in(scores)")
                )
            )
            handles.append(
                mod.register_forward_hook(
                    mk_fwd("per-layer-attn", f"{name}.out(probs)")
                )
            )
        elif name.endswith("linear1"):
            handles.append(mod.register_forward_hook(mk_fwd("per-layer-mlp", name)))
        elif name.endswith("linear2"):
            handles.append(mod.register_forward_hook(mk_fwd("per-layer-mlp", name)))
        elif name == "output_proj":
            handles.append(mod.register_forward_hook(mk_fwd("output-project", name)))

    x = torch.zeros((cfg.batch_size, cfg.seq_len), dtype=torch.long, device="meta")
    key_padding_mask = torch.ones(
        (cfg.batch_size, cfg.seq_len), dtype=torch.bool, device="meta"
    )
    with torch.no_grad():
        _ = model(x, key_padding_mask=key_padding_mask)

    for h in handles:
        h.remove()

    # Add bool masks created in attention forward. These are real allocations in your custom MHA.
    attention_mask_shape = (cfg.batch_size, 1, cfg.seq_len, cfg.seq_len)
    causal_mask_shape = (cfg.seq_len, cfg.seq_len)
    mask_numel = int(torch.Size(attention_mask_shape).numel())
    causal_numel = int(torch.Size(causal_mask_shape).numel())
    for layer_idx in range(cfg.layers):
        records.append(
            TensorRecord(
                bucket="others",
                name=f"dec_layers.{layer_idx}.attention_mask(bool)",
                shape=attention_mask_shape,
                numel=mask_numel,
                dtype_kind="bool",
            )
        )
        records.append(
            TensorRecord(
                bucket="others",
                name=f"dec_layers.{layer_idx}.causal_mask(bool)",
                shape=causal_mask_shape,
                numel=causal_numel,
                dtype_kind="bool",
            )
        )

    # Key padding mask per batch on device.
    records.append(
        TensorRecord(
            bucket="others",
            name="key_padding_mask(bool)",
            shape=(cfg.batch_size, cfg.seq_len),
            numel=cfg.batch_size * cfg.seq_len,
            dtype_kind="bool",
        )
    )
    return records


def _count_direct_params(module: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in module.parameters(recurse=False))


def _bucket_for_module(name: str) -> str:
    if name.endswith("multi_head_attention.packed_proj"):
        return "input-projects"
    if ".multi_head_attention." in name:
        return "per-layer-attn"
    if name.endswith("linear1") or name.endswith("linear2"):
        return "per-layer-mlp"
    if name == "output_proj":
        return "output-project"
    return "others"


def collect_param_counts(
    cfg: ShapeConfig,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    bucket_totals: defaultdict[str, int] = defaultdict(int)
    op_totals: defaultdict[tuple[str, str], int] = defaultdict(int)

    with torch.device("meta"):
        model = BaselineModel(
            vocab_size=cfg.vocab_size,
            layers=cfg.layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            pad_id=cfg.vocab_size - 2,
        )

    for name, mod in model.named_modules():
        if not name:
            continue
        param_count = _count_direct_params(mod)
        if param_count <= 0:
            continue
        bucket = _bucket_for_module(name)
        bucket_totals[bucket] += param_count
        op_totals[(bucket, _normalize_name(name))] += param_count

    return dict(bucket_totals), dict(op_totals)


def _record_gib(record: TensorRecord, dtype: str) -> float:
    if record.dtype_kind == "bool":
        return _to_gib(record.numel, BOOL_BYTES)
    if dtype == "fp32":
        return _to_gib(record.numel, FLOAT_FP32_BYTES)
    return _to_gib(record.numel, FLOAT_BF16_BYTES)


def _normalize_name(name: str) -> str:
    return re.sub(r"dec_layers\.\d+\.", "dec_layers.*.", name)


def summarize(records: list[TensorRecord], dtype: str) -> tuple[list[tuple[str, float]], list[tuple[str, str, int, tuple[int, ...], float]]]:
    bucket_totals: defaultdict[str, float] = defaultdict(float)
    for rec in records:
        bucket_totals[rec.bucket] += _record_gib(rec, dtype)

    grouped: defaultdict[tuple[str, str, tuple[int, ...], str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "gib": 0.0}
    )
    for rec in records:
        key = (rec.bucket, _normalize_name(rec.name), rec.shape, rec.dtype_kind)
        grouped[key]["count"] += 1
        grouped[key]["gib"] += _record_gib(rec, dtype)

    rows: list[tuple[str, str, int, tuple[int, ...], float]] = []
    for (bucket, name, shape, _dtype_kind), stats in grouped.items():
        rows.append((bucket, name, int(stats["count"]), shape, float(stats["gib"])))
    rows.sort(key=lambda x: x[4], reverse=True)

    bucket_rows = sorted(bucket_totals.items(), key=lambda x: x[1], reverse=True)
    return bucket_rows, rows


def render_terminal_view(
    cfg: ShapeConfig,
    records: list[TensorRecord],
    top_k: int,
    bucket_param_counts: dict[str, int],
    op_param_counts: dict[tuple[str, str], int],
) -> None:
    fp32_buckets, fp32_rows = summarize(records, dtype="fp32")
    bf16_buckets, bf16_rows = summarize(records, dtype="bf16")

    print("Memory Estimate (shape-only, no CUDA alloc)")
    print(
        "Config: "
        f"B={cfg.batch_size}, T={cfg.seq_len}, D={cfg.d_model}, "
        f"H={cfg.n_heads}, L={cfg.layers}, V={cfg.vocab_size}"
    )
    print()

    bucket_names = sorted(
        {name for name, _ in fp32_buckets}
        .union({name for name, _ in bf16_buckets})
        .union(bucket_param_counts.keys())
    )
    fp32_map = {name: val for name, val in fp32_buckets}
    bf16_map = {name: val for name, val in bf16_buckets}
    max_bucket = max([fp32_map.get(x, 0.0) for x in bucket_names] + [1e-9])

    bucket_table_rows: list[list[str]] = []
    for name in bucket_names:
        fp32_val = fp32_map.get(name, 0.0)
        bf16_val = bf16_map.get(name, 0.0)
        bucket_table_rows.append(
            [
                name,
                f"{bucket_param_counts.get(name, 0):,}",
                f"{fp32_val:7.3f}",
                f"{bf16_val:7.3f}",
                _bar(fp32_val, max_bucket),
                _bar(bf16_val, max_bucket),
            ]
        )

    _print_table(
        headers=[
            "Bucket",
            "Params",
            "FP32 GiB",
            "BF16 GiB",
            "FP32 Bar",
            "BF16 Bar",
        ],
        rows=bucket_table_rows,
    )

    print()
    print(f"Top {top_k} contributors (aggregated by op type):")
    top_rows: list[list[str]] = []
    bf16_map_rows = {(b, n, c, s): g for b, n, c, s, g in bf16_rows}
    for bucket, name, count, shape, fp32_gib in fp32_rows[:top_k]:
        bf16_gib = bf16_map_rows.get((bucket, name, count, shape), fp32_gib / 2.0)
        op_params = op_param_counts.get((bucket, name), 0)
        top_rows.append(
            [
                bucket,
                name,
                str(count),
                _shape_str(shape),
                f"{op_params:,}",
                f"{fp32_gib:7.3f}",
                f"{bf16_gib:7.3f}",
            ]
        )

    _print_table(
        headers=["Bucket", "Op", "Count", "Shape", "Params", "FP32 GiB", "BF16 GiB"],
        rows=top_rows,
    )


def render_plots(
    cfg: ShapeConfig,
    records: list[TensorRecord],
    out_dir: Path,
    top_k: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(
            "Skipping PNG plots: matplotlib is not installed. "
            "Install with `pip install matplotlib`."
        )
        _ = exc
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_buckets, fp32_rows = summarize(records, dtype="fp32")
    bf16_buckets, bf16_rows = summarize(records, dtype="bf16")

    # Bucket grouped bars
    bucket_names = sorted(
        {name for name, _ in fp32_buckets}.union({name for name, _ in bf16_buckets})
    )
    fp32_map = {name: val for name, val in fp32_buckets}
    bf16_map = {name: val for name, val in bf16_buckets}
    fp32_vals = [fp32_map.get(name, 0.0) for name in bucket_names]
    bf16_vals = [bf16_map.get(name, 0.0) for name in bucket_names]

    x = list(range(len(bucket_names)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], fp32_vals, width=width, label="fp32")
    ax.bar([i + width / 2 for i in x], bf16_vals, width=width, label="bf16")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=20, ha="right")
    ax.set_ylabel("GiB")
    ax.set_title(
        "Memory By Bucket\n"
        f"B={cfg.batch_size}, T={cfg.seq_len}, D={cfg.d_model}, H={cfg.n_heads}, L={cfg.layers}, V={cfg.vocab_size}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bucket_memory_fp32_vs_bf16.png", dpi=160)
    plt.close(fig)

    # Top contributors horizontal bars (fp32)
    top = fp32_rows[:top_k]
    labels = [f"{b}:{n}" for b, n, _count, _shape, _g in top]
    vals_fp32 = [g for _b, _n, _c, _s, g in top]
    vals_bf16 = []
    bf16_map_rows = {(b, n, c, s): g for b, n, c, s, g in bf16_rows}
    for b, n, c, s, g in top:
        vals_bf16.append(bf16_map_rows.get((b, n, c, s), g / 2.0))

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(labels) + 1)))
    y = list(range(len(labels)))
    ax.barh(y, vals_fp32, alpha=0.8, label="fp32")
    ax.barh(y, vals_bf16, alpha=0.8, label="bf16")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("GiB")
    ax.set_title(f"Top {top_k} Contributors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top_contributors_fp32_vs_bf16.png", dpi=160)
    plt.close(fig)
    return True


def save_json(
    cfg: ShapeConfig,
    records: list[TensorRecord],
    path: Path,
    bucket_param_counts: dict[str, int],
    op_param_counts: dict[tuple[str, str], int],
) -> None:
    fp32_buckets, fp32_rows = summarize(records, dtype="fp32")
    bf16_buckets, bf16_rows = summarize(records, dtype="bf16")
    payload = {
        "config": {
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "layers": cfg.layers,
            "vocab_size": cfg.vocab_size,
            "base_vocab_size": cfg.base_vocab_size,
            "num_special_tokens": cfg.num_special_tokens,
        },
        "bucket_totals": {
            "fp32": [
                {
                    "bucket": b,
                    "params": int(bucket_param_counts.get(b, 0)),
                    "gib": g,
                }
                for b, g in fp32_buckets
            ],
            "bf16": [
                {
                    "bucket": b,
                    "params": int(bucket_param_counts.get(b, 0)),
                    "gib": g,
                }
                for b, g in bf16_buckets
            ],
        },
        "top_ops": {
            "fp32": [
                {
                    "bucket": b,
                    "name": n,
                    "count": c,
                    "shape": list(s),
                    "params": int(op_param_counts.get((b, n), 0)),
                    "gib": g,
                }
                for b, n, c, s, g in fp32_rows
            ],
            "bf16": [
                {
                    "bucket": b,
                    "name": n,
                    "count": c,
                    "shape": list(s),
                    "params": int(op_param_counts.get((b, n), 0)),
                    "gib": g,
                }
                for b, n, c, s, g in bf16_rows
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_profile_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --profile-device cuda, but CUDA is unavailable.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported profile device: {device_name}")


def _resolve_bf16_mode(device: torch.device, bf16_mode: str) -> bool:
    if device.type != "cuda":
        return False

    supports = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if bf16_mode == "auto":
        return supports
    if bf16_mode == "on":
        if not supports:
            print("bf16 requested but this CUDA device/runtime reports no bf16 support.")
        return supports
    if bf16_mode == "off":
        return False
    raise ValueError(f"Unsupported bf16 mode: {bf16_mode}")


def _get_cuda_stage_memory(stage: str, before_allocated: int, device: torch.device) -> StageMemory:
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return StageMemory(
        stage=stage,
        allocated_gib=_bytes_to_gib(allocated),
        reserved_gib=_bytes_to_gib(reserved),
        delta_allocated_gib=_bytes_to_gib(allocated - before_allocated),
        peak_allocated_gib=_bytes_to_gib(peak_allocated),
        peak_reserved_gib=_bytes_to_gib(peak_reserved),
    )


def _print_train_profile_table(rows: list[StageMemory]) -> None:
    table_rows = [
        [
            row.stage,
            f"{row.allocated_gib:7.3f}",
            f"{row.reserved_gib:7.3f}",
            f"{row.delta_allocated_gib:+7.3f}",
            f"{row.peak_allocated_gib:7.3f}",
            f"{row.peak_reserved_gib:7.3f}",
        ]
        for row in rows
    ]
    _print_table(
        headers=[
            "Stage",
            "Alloc GiB",
            "Reserved GiB",
            "Delta Alloc GiB",
            "Peak Alloc GiB",
            "Peak Reserved GiB",
        ],
        rows=table_rows,
    )


def run_train_step_profile(
    cfg: ShapeConfig,
    profile_device: str,
    bf16_mode: str,
    learning_rate: float,
    seed: int,
) -> None:
    device = _resolve_profile_device(profile_device)
    use_bf16 = _resolve_bf16_mode(device, bf16_mode)

    print()
    print("Train Step Profile (real allocs)")
    print(
        f"Device={device} | bf16_autocast={'on' if use_bf16 else 'off'} | "
        f"B={cfg.batch_size}, T={cfg.seq_len}, D={cfg.d_model}, H={cfg.n_heads}, L={cfg.layers}, V={cfg.vocab_size}"
    )

    if device.type != "cuda":
        print(
            "CUDA memory counters are unavailable on CPU. "
            "Use `--profile-device cuda` on your GPU machine for allocator stats."
        )

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    pad_id = cfg.vocab_size - 2
    model = BaselineModel(
        vocab_size=cfg.vocab_size,
        layers=cfg.layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        pad_id=pad_id,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    input_seq = torch.randint(
        low=0,
        high=max(cfg.vocab_size - 3, 1),
        size=(cfg.batch_size, cfg.seq_len),
        dtype=torch.long,
        device=device,
    )
    target_seq = torch.randint(
        low=0,
        high=max(cfg.vocab_size - 3, 1),
        size=(cfg.batch_size, cfg.seq_len),
        dtype=torch.long,
        device=device,
    )
    key_padding_mask = torch.ones(
        (cfg.batch_size, cfg.seq_len), dtype=torch.bool, device=device
    )

    stage_rows: list[StageMemory] = []
    max_stage_peak_allocated_gib = 0.0
    max_stage_peak_reserved_gib = 0.0

    def record_stage(stage_name: str, fn) -> None:
        nonlocal max_stage_peak_allocated_gib, max_stage_peak_reserved_gib
        if device.type != "cuda":
            fn()
            return

        before_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        torch.cuda.synchronize(device)
        stage_row = _get_cuda_stage_memory(
            stage=stage_name, before_allocated=before_allocated, device=device
        )
        stage_rows.append(stage_row)
        max_stage_peak_allocated_gib = max(
            max_stage_peak_allocated_gib, stage_row.peak_allocated_gib
        )
        max_stage_peak_reserved_gib = max(
            max_stage_peak_reserved_gib, stage_row.peak_reserved_gib
        )

    try:
        optimizer.zero_grad(set_to_none=True)
        output: torch.Tensor | None = None
        loss: torch.Tensor | None = None

        def forward_pass() -> None:
            nonlocal output, loss
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                output = model(input_seq, key_padding_mask=key_padding_mask)
                loss_sum = loss_fn(
                    output.reshape(-1, output.size(-1)),
                    target_seq.reshape(-1),
                )
            valid_tokens = int((target_seq != pad_id).sum().item())
            denom = max(valid_tokens, 1)
            loss = loss_sum / denom

        def backward_pass() -> None:
            assert loss is not None
            loss.backward()

        def step_pass() -> None:
            optimizer.step()

        record_stage("forward+loss", forward_pass)
        record_stage("backward", backward_pass)
        record_stage("optimizer.step", step_pass)

        if device.type == "cuda":
            print()
            _print_train_profile_table(stage_rows)
            print()
            print(
                "Max stage peak allocated: "
                f"{max_stage_peak_allocated_gib:0.3f} GiB"
            )
            print(
                "Max stage peak reserved: "
                f"{max_stage_peak_reserved_gib:0.3f} GiB"
            )
    except torch.OutOfMemoryError as exc:
        print()
        print(f"OOM during train-step profile: {exc}")
        if device.type == "cuda" and stage_rows:
            print("Stages completed before OOM:")
            _print_train_profile_table(stage_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize estimated activation memory for BaselineModel using meta tensors "
            "(shape-only; no CUDA memory allocation)."
        )
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--base-vocab-size", type=int, default=33280)
    parser.add_argument("--num-special-tokens", type=int, default=3)
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Optional run_config.json to auto-load shape values.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of top contributors to print/plot.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional output directory for PNG charts (requires matplotlib).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path for machine-readable summary JSON.",
    )
    parser.add_argument(
        "--train-step-profile",
        action="store_true",
        help=(
            "Run one real train step (forward+backward+optimizer.step) and report "
            "stage-wise memory counters."
        ),
    )
    parser.add_argument(
        "--profile-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for --train-step-profile.",
    )
    parser.add_argument(
        "--profile-bf16",
        choices=["auto", "on", "off"],
        default="auto",
        help="bf16 autocast mode for --train-step-profile.",
    )
    parser.add_argument(
        "--profile-lr",
        type=float,
        default=0.001,
        help="Optimizer learning rate for --train-step-profile.",
    )
    parser.add_argument(
        "--profile-seed",
        type=int,
        default=42,
        help="Random seed for synthetic profile batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_config is not None:
        cfg = load_shape_config_from_run_config(args.run_config)
    else:
        cfg = ShapeConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            layers=args.layers,
            base_vocab_size=args.base_vocab_size,
            num_special_tokens=args.num_special_tokens,
        )

    records = collect_meta_records(cfg)
    bucket_param_counts, op_param_counts = collect_param_counts(cfg)
    render_terminal_view(
        cfg=cfg,
        records=records,
        top_k=args.top_k,
        bucket_param_counts=bucket_param_counts,
        op_param_counts=op_param_counts,
    )

    if args.plot_dir is not None:
        plotted = render_plots(
            cfg=cfg, records=records, out_dir=args.plot_dir, top_k=args.top_k
        )
        if plotted:
            print()
            print(f"Saved plots to: {args.plot_dir}")

    if args.json_out is not None:
        save_json(
            cfg=cfg,
            records=records,
            path=args.json_out,
            bucket_param_counts=bucket_param_counts,
            op_param_counts=op_param_counts,
        )
        print(f"Saved JSON summary to: {args.json_out}")

    if args.train_step_profile:
        run_train_step_profile(
            cfg=cfg,
            profile_device=args.profile_device,
            bf16_mode=args.profile_bf16,
            learning_rate=float(args.profile_lr),
            seed=int(args.profile_seed),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
