from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.config import (
    BPETokenizerConfig,
    BaselineDecoderConfig,
    ExperimentConfig,
    HoldoutSplitConfig,
    LocalTextDatasetConfig,
    LoggingConfig,
    OptimizerConfig,
    RunConfig,
    TrainConfig,
)
from src.training.runtime import (
    finalize_torch_compile_trace,
    maybe_compile_model,
    resolve_torch_compile_trace_dir,
)


class _FakeLogger:
    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id
        self.uploaded: list[dict[str, object]] = []

    def get_run_id(self) -> str | None:
        return self._run_id

    def upload_run_files(
        self,
        paths: list[str],
        *,
        base_path: str | None = None,
    ) -> None:
        self.uploaded.append(
            {
                "paths": list(paths),
                "base_path": base_path,
            }
        )


def make_config(
    *,
    provider: str = "console",
    use_torch_compile: bool = True,
    torch_compile_trace: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(
            project_name="runtime-test",
            run_name="runtime-test-run" if provider == "wandb" else None,
            artifacts_root="/tmp/artifacts",
            use_torch_compile=use_torch_compile,
            torch_compile_trace=torch_compile_trace,
        ),
        dataset=LocalTextDatasetConfig(path="/tmp/dataset.txt"),
        tokenizer=BPETokenizerConfig(vocab_path="/tmp/vocab.txt"),
        model=BaselineDecoderConfig(d_model=32, n_heads=4, layers=1),
        train=TrainConfig(
            epochs=1,
            optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.0),
            effective_batch_size=4,
            seq_len=16,
            stride=16,
        ),
        split=HoldoutSplitConfig(train_fraction=0.9, seed=42, shuffle=True),
        logging=LoggingConfig(provider=provider),
    )


class RuntimeCompileTests(unittest.TestCase):
    def test_resolve_torch_compile_trace_dir_uses_wandb_run_id(self) -> None:
        config = make_config(provider="wandb", torch_compile_trace=True)
        trace_dir = resolve_torch_compile_trace_dir(config, _FakeLogger(run_id="zft2vu6j"))
        self.assertEqual(trace_dir, Path("/tmp/tracedir/zft2vu6j"))

    def test_resolve_torch_compile_trace_dir_uses_default_console_root(self) -> None:
        config = make_config(provider="console", torch_compile_trace=True)
        trace_dir = resolve_torch_compile_trace_dir(config, _FakeLogger())
        self.assertEqual(trace_dir, Path("/tmp/tracedir"))

    def test_maybe_compile_model_sets_torch_trace_before_compile(self) -> None:
        model = torch.nn.Linear(4, 4)
        config = make_config(torch_compile_trace=True)
        trace_dir = Path("/tmp/tracedir/example-run")

        def fake_compile(module: torch.nn.Module, **_: object) -> torch.nn.Module:
            self.assertEqual(os.environ.get("TORCH_TRACE"), str(trace_dir))
            return module

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(torch, "compile", side_effect=fake_compile),
        ):
            compiled_model, compile_enabled, compile_status = maybe_compile_model(
                model,
                torch.device("cuda"),
                config,
                trace_dir=trace_dir,
            )

        self.assertIs(compiled_model, model)
        self.assertTrue(compile_enabled)
        self.assertEqual(compile_status, "enabled")

    def test_maybe_compile_model_leaves_torch_trace_unset_when_disabled(self) -> None:
        model = torch.nn.Linear(4, 4)
        config = make_config(torch_compile_trace=False)

        def fake_compile(module: torch.nn.Module, **_: object) -> torch.nn.Module:
            self.assertNotIn("TORCH_TRACE", os.environ)
            return module

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(torch, "compile", side_effect=fake_compile),
        ):
            maybe_compile_model(
                model,
                torch.device("cuda"),
                config,
                trace_dir=None,
            )

    def test_finalize_torch_compile_trace_uploads_raw_wandb_run_files(self) -> None:
        config = make_config(provider="wandb", torch_compile_trace=True)
        logger = _FakeLogger(run_id="zft2vu6j")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "zft2vu6j"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_file = trace_dir / "dedicated_log_torch_trace_zft2vu6j.log"
            trace_file.write_text("trace", encoding="utf-8")

            finalize_torch_compile_trace(
                config=config,
                logger=logger,
                trace_dir=trace_dir,
            )

        self.assertEqual(len(logger.uploaded), 1)
        self.assertEqual(
            logger.uploaded[0]["paths"],
            [str(trace_file)],
        )
        self.assertEqual(logger.uploaded[0]["base_path"], str(trace_dir.parent))

    def test_finalize_torch_compile_trace_runs_tlparse_for_console(self) -> None:
        config = make_config(provider="console", torch_compile_trace=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)
            trace_log = trace_dir / "dedicated_log_torch_trace_zft2vu6j.log"
            trace_log.write_text("trace", encoding="utf-8")

            with (
                mock.patch("src.training.runtime.shutil.which", return_value="/usr/bin/tlparse"),
                mock.patch("src.training.runtime.subprocess.run") as run_mock,
            ):
                finalize_torch_compile_trace(
                    config=config,
                    logger=_FakeLogger(),
                    trace_dir=trace_dir,
                )

        run_mock.assert_called_once_with(
            [
                "/usr/bin/tlparse",
                str(trace_log),
                "-o",
                "tl_out",
                "--overwrite",
            ],
            check=True,
        )

    def test_finalize_torch_compile_trace_requires_tlparse_for_console(self) -> None:
        config = make_config(provider="console", torch_compile_trace=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)
            (trace_dir / "dedicated_log_torch_trace_zft2vu6j.log").write_text(
                "trace",
                encoding="utf-8",
            )

            with mock.patch("src.training.runtime.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "requires `tlparse`"):
                    finalize_torch_compile_trace(
                        config=config,
                        logger=_FakeLogger(),
                        trace_dir=trace_dir,
                    )


if __name__ == "__main__":
    unittest.main()
