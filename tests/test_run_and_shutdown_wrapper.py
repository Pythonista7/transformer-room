from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "baseline" / "utils" / "run_and_shutdown.py"


def _build_shutdown_marker_cmd(marker_path: Path) -> str:
    code = (
        "from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text('shutdown-ok', encoding='utf-8')\n"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _single_file(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one match for {pattern}, found {len(matches)}")
    return matches[0]


class RunAndShutdownWrapperTests(unittest.TestCase):
    def _run_wrapper(
        self,
        *,
        child_code: str,
        child_exit_code: int,
        shutdown_cmd: str,
        log_dir: Path,
        wrapper_env: dict[str, str] | None = None,
        unset_env_keys: list[str] | None = None,
        inject_dummy_wandb: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        child_script = f"{child_code}\nimport sys\nsys.exit({child_exit_code})\n"
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--shutdown-cmd",
            shutdown_cmd,
            "--log-dir",
            str(log_dir),
            "--run-name",
            "wrapper-test",
            "--",
            sys.executable,
            "-c",
            child_script,
        ]
        env = dict(os.environ)
        if unset_env_keys:
            for key in unset_env_keys:
                env.pop(key, None)
        if inject_dummy_wandb:
            env.setdefault("WANDB_API_KEY", "dummy-test-key")
        if wrapper_env:
            env.update(wrapper_env)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def test_stdout_and_stderr_are_captured_in_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            shutdown_marker = tmp_path / "shutdown_marker.txt"

            result = self._run_wrapper(
                child_code="import sys\nprint('child-stdout')\nprint('child-stderr', file=sys.stderr)",
                child_exit_code=0,
                shutdown_cmd=_build_shutdown_marker_cmd(shutdown_marker),
                log_dir=log_dir,
            )

            self.assertEqual(result.returncode, 0)
            log_path = _single_file(log_dir, "*.log")
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("child-stdout", log_text)
            self.assertIn("child-stderr", log_text)

    def test_metadata_contains_required_fields_and_child_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            shutdown_marker = tmp_path / "shutdown_marker.txt"

            result = self._run_wrapper(
                child_code="print('ok')",
                child_exit_code=0,
                shutdown_cmd=_build_shutdown_marker_cmd(shutdown_marker),
                log_dir=log_dir,
            )

            self.assertEqual(result.returncode, 0)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["child_return_code"], 0)
            self.assertIn("command", metadata)
            self.assertIn("cwd", metadata)
            self.assertIn("hostname", metadata)
            self.assertIn("pid", metadata)
            self.assertIn("start_time_utc", metadata)
            self.assertIn("end_time_utc", metadata)
            self.assertIn("duration_sec", metadata)
            self.assertIn("shutdown", metadata)

    def test_shutdown_runs_after_success_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            shutdown_marker = tmp_path / "shutdown_marker.txt"

            result = self._run_wrapper(
                child_code="print('done')",
                child_exit_code=0,
                shutdown_cmd=_build_shutdown_marker_cmd(shutdown_marker),
                log_dir=log_dir,
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(shutdown_marker.exists())

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["shutdown"]["return_code"], 0)

    def test_shutdown_runs_after_non_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            shutdown_marker = tmp_path / "shutdown_marker.txt"

            result = self._run_wrapper(
                child_code="print('failing-run')",
                child_exit_code=7,
                shutdown_cmd=_build_shutdown_marker_cmd(shutdown_marker),
                log_dir=log_dir,
            )

            self.assertEqual(result.returncode, 7)
            self.assertTrue(shutdown_marker.exists())

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["child_return_code"], 7)

    def test_shutdown_failure_does_not_override_child_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"

            result = self._run_wrapper(
                child_code="print('failing-child')",
                child_exit_code=5,
                shutdown_cmd="false",
                log_dir=log_dir,
            )

            self.assertEqual(result.returncode, 5)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 5)
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["shutdown"]["return_code"], 1)
            self.assertIn("shutdown command exited with code", metadata["shutdown"]["error"])

    def test_missing_command_after_separator_returns_argument_error(self) -> None:
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--shutdown-cmd",
            "echo shutdown",
            "--",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing command after '--'", result.stderr)

    def test_default_pytorch_alloc_conf_is_set_before_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            shutdown_marker = tmp_path / "shutdown_marker.txt"

            result = self._run_wrapper(
                child_code="import os\nprint('alloc_conf=' + str(os.environ.get('PYTORCH_ALLOC_CONF')))",
                child_exit_code=0,
                shutdown_cmd=_build_shutdown_marker_cmd(shutdown_marker),
                log_dir=log_dir,
                unset_env_keys=["PYTORCH_ALLOC_CONF"],
            )

            self.assertEqual(result.returncode, 0)
            log_path = _single_file(log_dir, "*.log")
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("alloc_conf=expandable_segments:True", log_text)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["env"]["pytorch_alloc_conf"], "expandable_segments:True")
            self.assertEqual(metadata["env"]["pytorch_alloc_conf_source"], "defaulted")

    def test_missing_wandb_api_key_fails_without_interactive_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            shutdown_marker = tmp_path / "shutdown_marker.txt"

            result = self._run_wrapper(
                child_code="print('will-not-run')",
                child_exit_code=0,
                shutdown_cmd=_build_shutdown_marker_cmd(shutdown_marker),
                log_dir=log_dir,
                unset_env_keys=["WANDB_API_KEY"],
                inject_dummy_wandb=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(shutdown_marker.exists())

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 2)
            self.assertIn("WANDB_API_KEY is not set", metadata["env"]["error"])


if __name__ == "__main__":
    unittest.main()
