# Memory Visualization Utility

This utility estimates and profiles memory for the baseline decoder model.

File: `baseline/utils/util.py`  
Module entrypoint: `python -m baseline.utils.util`

## What It Supports

1. Shape-only memory estimation (no CUDA allocation, no OOM risk).
2. FP32 vs BF16 comparison tables in terminal (including parameter counts by bucket/op).
3. Optional PNG plots (with `matplotlib`).
4. Optional JSON export for downstream analysis.
5. One-step real training memory profiling (`forward+loss`, `backward`, `optimizer.step`).

## Setup

From repo root:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

For PNG charts:

```bash
.venv/bin/python -m pip install matplotlib
```

## Quick Start

Run with defaults:

```bash
.venv/bin/python -m baseline.utils.util
```

## Core Use Cases

### 1) Shape-Only Estimate From Explicit CLI Args

Use this when experimenting with hypothetical shapes.

```bash
.venv/bin/python -m baseline.utils.util \
  --batch-size 48 \
  --seq-len 1024 \
  --d-model 768 \
  --n-heads 8 \
  --layers 12 \
  --base-vocab-size 33280 \
  --num-special-tokens 3 \
  --top-k 20
```

### 2) Shape-Only Estimate From `run_config.json`

Use your saved run artifact config directly.

```bash
.venv/bin/python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --top-k 20
```

### 3) Save Plots (PNG)

Generates:
- `bucket_memory_fp32_vs_bf16.png`
- `top_contributors_fp32_vs_bf16.png`

```bash
.venv/bin/python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --plot-dir baseline/models/wikitext2_gpt2_v1/memviz
```

### 4) Save JSON Summary

```bash
.venv/bin/python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --json-out baseline/models/wikitext2_gpt2_v1/memviz/summary.json
```

### 5) Combined: Table + Plots + JSON

```bash
.venv/bin/python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --top-k 25 \
  --plot-dir baseline/models/wikitext2_gpt2_v1/memviz \
  --json-out baseline/models/wikitext2_gpt2_v1/memviz/summary.json
```

### 6) Real One-Step Train Memory Profile (CUDA)

Runs one synthetic train step and prints stage memory counters.

```bash
.venv/bin/python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --train-step-profile \
  --profile-device cuda \
  --profile-bf16 auto
```

### 7) Train Step Profile With BF16 Forced Off

```bash
.venv/bin/python -m baseline.utils.util \
  --run-config baseline/models/wikitext2_gpt2_v1/run_config.json \
  --train-step-profile \
  --profile-device cuda \
  --profile-bf16 off
```

### 8) CPU Dry-Run For Profiler Wiring

Useful to validate command wiring when GPU is unavailable.

```bash
.venv/bin/python -m baseline.utils.util \
  --batch-size 2 \
  --seq-len 32 \
  --d-model 64 \
  --n-heads 8 \
  --layers 1 \
  --base-vocab-size 256 \
  --num-special-tokens 3 \
  --train-step-profile \
  --profile-device cpu
```

## CLI Arguments

### Model/shape inputs

- `--batch-size` (default: `64`)
- `--seq-len` (default: `1024`)
- `--d-model` (default: `768`)
- `--n-heads` (default: `8`)
- `--layers` (default: `12`)
- `--base-vocab-size` (default: `33280`)
- `--num-special-tokens` (default: `3`)
- `--run-config` (optional path; overrides shape fields above)

### Reporting/outputs

- `--top-k` (default: `12`)
- `--plot-dir` (optional, requires `matplotlib`)
- `--json-out` (optional)

### Real train-step profile

- `--train-step-profile` (flag)
- `--profile-device {auto,cuda,cpu}` (default: `auto`)
- `--profile-bf16 {auto,on,off}` (default: `auto`)
- `--profile-lr` (default: `0.001`)
- `--profile-seed` (default: `42`)

## Notes

1. Shape-only mode is a formula-based estimate and does not include all runtime allocator behavior.
2. Real train-step mode uses synthetic token batches, not your dataset pipeline.
3. If `matplotlib` is missing and `--plot-dir` is provided, the utility skips plot generation and continues.
