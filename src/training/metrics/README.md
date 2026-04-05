# Metrics Plugin Pipeline

This package keeps training metrics modular and logger-agnostic.

## Lifecycle

`MetricsEngine` calls plugins in fixed registry order:

1. `on_train_start()`
2. `on_step_start(step_ctx)`
3. `after_microbatch_backward(microbatch_ctx)` (called once per successful micro-batch backward inside an optimizer step)
4. `after_backward(step_ctx)`
5. `after_optimizer_step(step_ctx)`
6. `collect_step_metrics(step_ctx)`
7. `collect_periodic_val_metrics(val_ctx)`
8. `collect_epoch_metrics(epoch_ctx)`
9. `on_train_end()`

`build_metric_schedule(...)` computes cadence/gating booleans used by plugins.

## Key Rules

- Plugins must emit stable scalar keys.
- Metric keys must be unique across plugins for each collection phase.
- `MetricsEngine` raises an error on key collisions.
- Backend grouping/formatting stays in logger adapters.

## Built-In Diagnostics Metrics

- `global_param_norm`
- `layer_param_norm_first|middle|last`
- `param_update_norm`
- `update_to_weight_ratio`
- `adam_m_norm`, `adam_v_norm`, `adam_elemwise_snr_norm`
- `layer_estimated_variance_norm_first|middle|last`

`param_update_norm` is approximated from optimizer state for `Adam`/`AdamW` using each parameter group's current (`lr`, `betas`, `eps`) at metric time.
It is not emitted for unsupported optimizers.
For VRAM safety, heavy parameter/optimizer diagnostics avoid full pre-step model snapshots.
Use `parameter_optimizer_norms_every_n_steps` to decouple these heavy metrics from the general diagnostics cadence.

## Attention Entropy Capture Flow

`ForwardHookMetricsPlugin` captures attention entropy with a one-shot stash per step:

1. On `on_train_start()`, the plugin installs forward hooks on selected attention modules and their `packed_proj` submodules.
2. The `packed_proj` hook captures the projected `all_projs` tensor, and the outer attention hook captures `mask`, `key_padding_mask`, and `is_causal` kwargs before stashing the projection bundle for the current layer.
3. The collector keeps only the first capture per layer for the current step (first micro-batch semantics). Later micro-batches in the same step are ignored.
4. On `collect_step_metrics(...)`, entropy is computed off-graph from the stashed projections and emitted as `attention_entropy_first|middle|last`.
5. After collection, stash/capture state is cleared so validation or other forwards do not repopulate step entropy.

### Train Loop Sequence

The concrete call path during training is:

1. `train_loop(...)` builds `MetricsEngine`, which includes `ForwardHookMetricsPlugin`.
2. `metrics_engine.on_train_start()` installs the forward hooks once for the whole run.
3. At the start of each optimizer step, `train_loop(...)` builds `StepMetricsContext` and calls `metrics_engine.on_step_start(step_ctx)`.
4. `ForwardHookMetricsPlugin.on_step_start(...)` calls `collector.begin_step(...)`, which clears prior step state and enables capture for the current step if the schedule says entropy should be collected.
5. Each micro-batch forward runs `model(...)`, which runs attention and triggers:
   - the `packed_proj` hook, which caches `all_projs`
   - the outer attention hook, which forwards `all_projs`, `mask`, `key_padding_mask`, and `is_causal` into `stash_attention_projection_once(...)`
6. `stash_attention_projection_once(...)` accepts only the first capture for each sampled layer in the current optimizer step. Later micro-batches hit the same hook path but return early because that layer has already been captured.
7. After gradient accumulation and optimizer step, `train_loop(...)` calls `metrics_engine.collect_step_metrics(step_ctx)`.
8. `ForwardHookMetricsPlugin.collect_step_metrics(...)` computes entropy from the stashed first-micro-batch projections and emits `attention_entropy_first|middle|last`.
9. Step state is cleared after collection, so the next optimizer step starts with an empty stash.

Notes:
- This avoids recomputing projections in metric hooks.
- Stashed tensors stay on-device (no CPU transfer during stash).
- Entropy capture no longer requires attention modules to call back into Python from inside `forward(...)`.
- `torch.compile(fullgraph=True)` is treated as best-effort only for this metric path.

## Add a New Metric

1. Create a plugin in `plugins/` (copy `plugins/template.py`).
2. Implement the needed lifecycle method(s).
3. Register it in `build_default_metric_plugins(...)` in `registry.py`.
4. Add plugin-focused tests and, if needed, integration assertions.

## Debug Timing

Set `metrics_debug_timing=True` when calling `model_pipeline(...)` to print per-plugin timing totals by phase.
