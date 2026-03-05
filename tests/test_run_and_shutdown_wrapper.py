from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "baseline" / "utils" / "run_and_shutdown.py"


def _single_file(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one match for {pattern}, found {len(matches)}")
    return matches[0]


def _write_vast_sdk_stub(stub_dir: Path, *, should_fail: bool = False) -> None:
    stub_dir.mkdir(parents=True, exist_ok=True)
    stop_impl = (
        "        raise RuntimeError('vast stop failed')\n"
        if should_fail
        else (
            "        marker_path = os.environ.get('VAST_TEST_MARKER', '').strip()\n"
            "        payload = {'stopped_id': ID, 'api_key': self.api_key}\n"
            "        if marker_path:\n"
            "            Path(marker_path).write_text(json.dumps(payload), encoding='utf-8')\n"
            "        return payload\n"
        )
    )
    (stub_dir / "vastai_sdk.py").write_text(
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "class VastAI:\n"
        "    def __init__(self, api_key):\n"
        "        self.api_key = api_key\n"
        "\n"
        "    def stop_instance(self, ID):\n"
        + stop_impl,
        encoding="utf-8",
    )


class RunAndShutdownWrapperTests(unittest.TestCase):
    def _run_wrapper(
        self,
        *,
        child_code: str,
        child_exit_code: int,
        log_dir: Path,
        shutdown_provider: str = "command",
        wrapper_env: dict[str, str] | None = None,
        unset_env_keys: list[str] | None = None,
        inject_dummy_wandb: bool = True,
        pythonpath_entries: list[Path],
    ) -> subprocess.CompletedProcess[str]:
        child_script = f"{child_code}\nimport sys\nsys.exit({child_exit_code})\n"
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--log-dir",
            str(log_dir),
            "--run-name",
            "wrapper-test",
            "--",
            sys.executable,
            "-c",
            child_script,
        ]
        if shutdown_cmd is not None:
            cmd[2:2] = ["--shutdown-cmd", shutdown_cmd]
        env = dict(os.environ)
        if unset_env_keys:
            for key in unset_env_keys:
                env.pop(key, None)
        if inject_dummy_wandb:
            env.setdefault("WANDB_API_KEY", "dummy-test-key")
        if wrapper_env:
            env.update(wrapper_env)

        pythonpath_parts = [str(path) for path in pythonpath_entries]
        existing_pythonpath = env.get("PYTHONPATH", "").strip()
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

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
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="import sys\nprint('child-stdout')\nprint('child-stderr', file=sys.stderr)",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                },
                pythonpath_entries=[stub_dir],
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
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('ok')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                },
                pythonpath_entries=[stub_dir],
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
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "vast_stop.json"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('done')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                    "VAST_TEST_MARKER": str(marker_path),
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(marker_path.exists())

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["shutdown"]["return_code"], 0)
            self.assertEqual(metadata["shutdown"]["provider"], "vast")

    def test_shutdown_runs_after_non_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "vast_stop.json"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('failing-run')",
                child_exit_code=7,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                    "VAST_TEST_MARKER": str(marker_path),
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 7)
            self.assertTrue(marker_path.exists())

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["child_return_code"], 7)

    def test_vast_shutdown_failure_does_not_override_child_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir, should_fail=True)

            result = self._run_wrapper(
                child_code="print('failing-child')",
                child_exit_code=5,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 5)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 5)
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertIsNone(metadata["shutdown"]["return_code"])
            self.assertIn("vast stop failed", metadata["shutdown"]["error"])

    def test_missing_command_after_separator_returns_argument_error(self) -> None:
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing command after '--'", result.stderr)

    def test_default_pytorch_alloc_conf_is_set_before_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="import os\nprint('alloc_conf=' + str(os.environ.get('PYTORCH_ALLOC_CONF')))",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                },
                unset_env_keys=["PYTORCH_ALLOC_CONF"],
                pythonpath_entries=[stub_dir],
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
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('will-not-run')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                },
                unset_env_keys=["WANDB_API_KEY"],
                inject_dummy_wandb=False,
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 2)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 2)
            self.assertIn("WANDB_API_KEY is not set", metadata["env"]["error"])

    def test_shutdown_uses_vast_sdk_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "vast_stop.json"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('done')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "VAST_CONTAINERLABEL": "C.12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                    "VAST_TEST_MARKER": str(marker_path),
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(marker_path.exists())
            marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker_payload["stopped_id"], 12345)
            self.assertEqual(marker_payload["api_key"], "vast-test-key")

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["shutdown"]["provider"], "vast")
            self.assertEqual(metadata["shutdown"]["return_code"], 0)
            self.assertEqual(metadata["shutdown"]["vast_instance_id"], 12345)
            self.assertEqual(metadata["env"]["vast_api_key_source"], "CONTAINER_API_KEY")

    def test_missing_vast_api_key_fails_without_interactive_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('will-not-run')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={"CONTAINER_ID": "12345"},
                unset_env_keys=["CONTAINER_API_KEY", "VAST_API_KEY"],
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 2)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 2)
            self.assertIn("VAST_API_KEY is not set", metadata["env"]["error"])

    def test_missing_container_id_fails_without_running_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_vast_sdk_stub(stub_dir)

            result = self._run_wrapper(
                child_code="print('will-not-run')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={"CONTAINER_API_KEY": "vast-test-key"},
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 2)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 2)
            self.assertIn("CONTAINER_ID/VAST_CONTAINERLABEL is not set", metadata["env"]["error"])


if __name__ == "__main__":
    unittest.main()
