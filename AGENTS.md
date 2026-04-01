# Repository Guidelines

## Project Structure & Module Organization
Core training code lives in `src/`.
- `src/train.py`: pipeline entrypoint (`model_pipeline`) and train loop orchestration.
- `src/core/`: typed configs, protocol types, and adapter registry.
- `src/adapters/`: dataset/tokenizer/model/split/logger implementations.
- `src/training/`: runtime, optimizer/scheduler, evaluation, artifacts, metrics plugins.
- `src/components/`: transformer building blocks (attention, blocks, models, tokenizers).

Experiments are under `experiments/` (for example `experiments/baseline/...`).
Tests live in `tests/` as `test_*.py`. Local corpora are in `datasets/`; generated outputs commonly land in `src/models/` and `artifacts/`.

## Build, Test, and Development Commands
Use the project venv from repo root.
- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`: run full test suite.
- `.venv/bin/python -m unittest tests.test_pipeline_smoke`: run one module quickly.
- `.venv/bin/python -m experiments.baseline.hyperparam_sweeps.tiny_shakespeare`: run a baseline experiment via module path.
- `.venv/bin/python experiments/baseline/hyperparam_sweeps/<experiment>.py`: run an experiment script directly.

Install/update dependencies with `.venv/bin/pip install -r requirements.txt` when needed.

## Coding Style & Naming Conventions
Target Python 3.13 with 4-space indentation and type hints on public surfaces.
Follow existing patterns:
- `snake_case` for functions/variables/files.
- `PascalCase` for classes and config dataclasses.
- Keep modules focused and explicit; prefer small adapter-bound units over implicit global behavior.
- Use absolute imports from `src` package roots where possible.

## Testing Guidelines
Framework: `unittest` (with `unittest.mock` where needed).
- Name files `test_*.py` and classes `*Tests`.
- Keep tests near behavior seams (config validation, adapters, metrics engine, pipeline smoke).
- Add regression tests for each bug fix and for new config branches.

## Commit & Pull Request Guidelines
Recent history uses short prefixes like `ft:`, `fx:`, `rf:`, `ch:`.
- Commit format: `<prefix>: <concise description>` (imperative, scoped).
- Keep commits focused; avoid mixing refactors and behavior changes.

PRs should include:
- What changed and why.
- Risk/rollback notes for training/runtime behavior.
- Repro steps (exact command run) and test evidence.
- Linked issue/experiment context; include screenshots only for UI/report artifacts.

## Security & Configuration Tips
Do not commit secrets or local `.env` values. Keep large artifacts/checkpoints out of git; use configured artifact logging (for example W&B) for shareable run outputs.
