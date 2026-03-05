# Metrics Plugin Pipeline

This package keeps training metrics modular and logger-agnostic.

## Lifecycle

`MetricsEngine` calls plugins in fixed registry order:

1. `on_train_start()`
2. `on_step_start(step_ctx)`
3. `after_backward(step_ctx)`
4. `after_optimizer_step(step_ctx)`
5. `collect_step_metrics(step_ctx)`
6. `collect_periodic_val_metrics(val_ctx)`
7. `collect_epoch_metrics(epoch_ctx)`
8. `on_train_end()`

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

`param_update_norm` is computed from pre-step vs post-step parameter snapshots. It is not directly exposed by optimizer state dicts.
For VRAM safety, heavy parameter/optimizer diagnostics are reduced from CPU snapshots instead of GPU-resident clones.
Use `parameter_optimizer_norms_every_n_steps` to decouple these heavy metrics from the general diagnostics cadence.

## Add a New Metric

1. Create a plugin in `plugins/` (copy `plugins/template.py`).
2. Implement the needed lifecycle method(s).
3. Register it in `build_default_metric_plugins(...)` in `registry.py`.
4. Add plugin-focused tests and, if needed, integration assertions.

## Debug Timing

Set `metrics_debug_timing=True` when calling `model_pipeline(...)` to print per-plugin timing totals by phase.
