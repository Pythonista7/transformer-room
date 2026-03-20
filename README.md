# Transformer Efficiency Lab (TEL)

A hands-on lab for understanding, implementing, and optimizing transformer training from first principles.

TEL is where I build decoder language models, training loops, and systems-level experiments to study how transformers behave in practice — not just how they look on paper.

## What this repo is

This repo is part of my effort to:

- implement transformer components from scratch
- understand training dynamics deeply
- benchmark optimization and memory techniques
- build intuition through code, ablations, and measurement

The focus is not just “train a model,” but to answer questions like:

- What actually happens inside attention, dropout, and backprop?
- How do batch size, gradient accumulation, and LR scaling interact?
- What do activation checkpointing and compilation really buy us?
- How should we measure throughput, memory, and training quality together?

## Current scope

TEL currently focuses on:

- decoder-style language model training
- modular experiment configuration
- dataset / tokenizer / model / logger adapters
- micro-batching and gradient accumulation
- token-aware gradient scaling
- bf16 autocast and `torch.compile`
- checkpointing and artifact tracking
- step, epoch, and validation metrics
- reproducible experimentation

## Project goals

The broader goal of TEL is to create a strong experimental spine for studying transformer efficiency across:

- optimization
- memory usage
- throughput
- architectural tradeoffs
- training stability
- scaling behavior

This repo is designed to be a research-and-engineering sandbox rather than a polished framework.

## Repo philosophy

A few principles guide this project:

- **Build from scratch** to understand the mechanics.
- **Measure everything** — loss alone is not enough.
- **Prefer clear abstractions** over magic.
- **Keep experiments reproducible** and easy to compare.
- **Use the repo as a lab notebook** for real learning.

## Training pipeline at a glance

At a high level:

1. load corpus
2. build tokenizer + vocab
3. create train/val loaders
4. build model
5. resolve batching + learning rate
6. train with metrics, checkpointing, and validation
7. save final artifacts

## Related notes and writeups

### TEL overview
- [TEL Home](https://ashwinms.com/TEL/)
- [Phase_0](https://ashwinms.com/TEL/Baseline/)

### Core Phase_0 tracks
- [Lineage of Transformer](https://ashwinms.com/TEL/Baseline/1-Lineage-of-the-Transformer)
- [Implementing a Baseline Model](https://ashwinms.com/TEL/Baseline/2-Implementing-a-Baseline-Model)
- [Resources](https://ashwinms.com/TEL/Baseline/Resources)

### Baseline experiments
- [LR vs Batch Size Empirical Sweep](https://ashwinms.com/TEL/Baseline/4-Experiments/1-Hyper-Param-Sweep/LR-vs-Batch-Size-Empirical-Sweep)
- [Experiments with Weight Decay](https://ashwinms.com/TEL/Baseline/4-Experiments/1-Hyper-Param-Sweep/Experiments-with-Weight-Decay)
- [Activation Checkpointing](https://ashwinms.com/TEL/Baseline/4-Experiments/Mem--and--GPU-Exp/Activation-Checkpointing)
- [Micro-batching + Gradient Accumulation](https://ashwinms.com/TEL/Baseline/4-Experiments/Mem--and--GPU-Exp/Micro-batching-%2B-Gradient-Accumulation)

## Why this exists

Most learning resources explain transformers at a high level. Fewer force you to confront the practical details:

- loss reduction choice
- padding-aware token accounting
- when gradients should be scaled
- how optimizer state behaves
- what “effective batch size” really means
- which optimizations help, and which only sound good

TEL exists to close that gap.

## Status

Active and evolving.

This is an experimental repo, so expect:

- frequent iteration
- changing APIs
- ablation-heavy code
- implementation notes tied to ongoing experiments