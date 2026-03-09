"""
0. No compile, No AC, Baseline model as it is
1. No compile, Use only AC every N layers [baseline_with_AC_model.py] & [N = 5, 2, 1(all)]
2. No compile, Use SAC [baseline_with_SAC_model.py]
3. Use BaselineModel with .compile()
4. Use BaselineModel .compile with the memory budget api, with pareto frontier of budgets [0.2, 0.4, 0.5, 0.6, 0.8]
> torch._dynamo.config.activation_memory_budget = 0.5
> out = torch.compile(fn)(inp)

Track:
- Peak allocated memory usage
- Peak reserved memory usage
- Step time
- epoch time
- train loss
- val loss
- forward-pass-time
- backward-pass-time (important!)
- optim-step-time

Warnings:
Warm up compiled runs separately.
- torch.compile() has compile overhead and early-iteration instability, so don’t include first-step timings in the comparison. Measure after warmup, otherwise you’re benchmarking compilation tax, not training tax. The memory budget API is specifically tied to torch.compile, so this matters even more there
- Reset CUDA peak stats before measuring.

"""
