from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import src.utils.run_and_shutdown as run_and_shutdown


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "src" / "utils" / "run_and_shutdown.py"


def _single_file(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one match for {pattern}, found {len(matches)}")
    return matches[0]


def _write_requests_stub(stub_dir: Path, *, status_code: int = 200, should_fail: bool = False) -> None:
    stub_dir.mkdir(parents=True, exist_ok=True)
    if should_fail:
        body = """import json as json_module
import os
from pathlib import Path

class RequestException(Exception):
    pass

class Response:
    def __init__(self, *, status_code, payload, text):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

def put(url, *, headers, json, timeout):
    raise RequestException("network down")
"""
    else:
        body = f"""import json as json_module
import os
from pathlib import Path

class RequestException(Exception):
    pass

class Response:
    def __init__(self, *, status_code, payload, text):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

def put(url, *, headers, json, timeout):
    marker_path = os.environ.get("VAST_TEST_MARKER", "").strip()
    payload = {{"success": {str(status_code < 400)}, "stopped_id": int(url.rstrip("/").split("/")[-1])}}
    if marker_path:
        Path(marker_path).write_text(
            json_module.dumps({{"url": url, "headers": headers, "json": json}}),
            encoding="utf-8",
        )
    return Response(status_code={status_code}, payload=payload, text=json_module.dumps(payload))
"""
    (stub_dir / "requests.py").write_text(body, encoding="utf-8")


def _write_thunder_requests_stub(
    stub_dir: Path,
    *,
    control_plane_instance_id: str = "0",
    instance_id: str = "550e8400-e29b-41d4-a716-446655440000",
    instance_name: str = "wrapper-test-instance",
    snapshot_status_code: int = 202,
    delete_status_code: int = 200,
    should_fail: bool = False,
) -> None:
    stub_dir.mkdir(parents=True, exist_ok=True)
    if should_fail:
        body = """class RequestException(Exception):
    pass

class Response:
    def __init__(self, *, status_code, payload, text):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

def get(url, *, headers, timeout):
    raise RequestException("network down")

def post(url, *, headers, json, timeout):
    raise RequestException("network down")
"""
    else:
        body = f"""import json as json_module
import os
from pathlib import Path

class RequestException(Exception):
    pass

class Response:
    def __init__(self, *, status_code, payload, text):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

def _read_marker(marker_path):
    if marker_path and Path(marker_path).exists():
        return json_module.loads(Path(marker_path).read_text(encoding="utf-8"))
    return []

def _write_marker(marker_path, payload):
    if marker_path:
        Path(marker_path).write_text(json_module.dumps(payload), encoding="utf-8")

def get(url, *, headers, timeout):
    marker_path = os.environ.get("THUNDER_TEST_MARKER", "").strip()
    existing = _read_marker(marker_path)
    existing.append({{"method": "GET", "url": url, "headers": headers}})
    _write_marker(marker_path, existing)

    payload = {{
        "{control_plane_instance_id}": {{
            "uuid": "{instance_id}",
            "name": "{instance_name}",
            "status": "RUNNING"
        }}
    }}
    return Response(
        status_code=200,
        payload=payload,
        text=json_module.dumps(payload),
    )

def post(url, *, headers, json, timeout):
    marker_path = os.environ.get("THUNDER_TEST_MARKER", "").strip()
    existing = _read_marker(marker_path)
    existing.append({{"method": "POST", "url": url, "headers": headers, "json": json}})
    _write_marker(marker_path, existing)

    if url.endswith("/snapshots/create"):
        payload = {{"accepted": {str(snapshot_status_code < 400)}}}
        return Response(
            status_code={snapshot_status_code},
            payload=payload,
            text=json_module.dumps(payload),
        )

    payload = {{"deleted": {str(delete_status_code < 400)}}}
    return Response(
        status_code={delete_status_code},
        payload=payload,
        text=json_module.dumps(payload),
    )
"""
    (stub_dir / "requests.py").write_text(body, encoding="utf-8")


class RunAndShutdownWrapperTests(unittest.TestCase):
    def _make_tee(self) -> run_and_shutdown._Tee:
        log_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
        self.addCleanup(lambda: Path(log_file.name).unlink(missing_ok=True))
        self.addCleanup(log_file.close)
        return run_and_shutdown._Tee(log_file)

    def test_confirm_shutdown_or_abort_aborts_on_abort_input(self) -> None:
        tee = self._make_tee()
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True
        fake_stdin.readline.return_value = "  AbOrT  \n"

        with (
            mock.patch.object(run_and_shutdown.sys, "stdin", fake_stdin),
            mock.patch.object(
                run_and_shutdown.select,
                "select",
                return_value=([fake_stdin], [], []),
            ),
        ):
            result = run_and_shutdown._confirm_shutdown_or_abort(
                tee=tee,
                window_seconds=30,
            )

        self.assertFalse(result["proceed"])
        self.assertTrue(result["aborted_by_user"])
        self.assertEqual(result["interrupt_input"], "AbOrT")
        self.assertTrue(result["interrupt_prompt_shown"])
        self.assertEqual(result["skip_reason"], "aborted_by_user")

    def test_confirm_shutdown_or_abort_times_out_and_proceeds(self) -> None:
        tee = self._make_tee()
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True

        with (
            mock.patch.object(run_and_shutdown.sys, "stdin", fake_stdin),
            mock.patch.object(
                run_and_shutdown.select,
                "select",
                return_value=([], [], []),
            ),
        ):
            result = run_and_shutdown._confirm_shutdown_or_abort(
                tee=tee,
                window_seconds=30,
            )

        self.assertTrue(result["proceed"])
        self.assertFalse(result["aborted_by_user"])
        self.assertIsNone(result["interrupt_input"])
        self.assertTrue(result["interrupt_prompt_shown"])
        self.assertEqual(result["skip_reason"], "interrupt_timeout_auto_proceed")

    def test_confirm_shutdown_or_abort_non_tty_auto_proceeds(self) -> None:
        tee = self._make_tee()
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False

        with mock.patch.object(run_and_shutdown.sys, "stdin", fake_stdin):
            result = run_and_shutdown._confirm_shutdown_or_abort(
                tee=tee,
                window_seconds=30,
            )

        self.assertTrue(result["proceed"])
        self.assertFalse(result["aborted_by_user"])
        self.assertFalse(result["interrupt_prompt_shown"])
        self.assertIsNone(result["interrupt_input"])
        self.assertEqual(result["skip_reason"], "no_tty_auto_proceed")

    def _run_wrapper(
        self,
        *,
        provider: str = "vast",
        child_code: str,
        child_exit_code: int,
        log_dir: Path,
        wrapper_env: dict[str, str] | None = None,
        unset_env_keys: list[str] | None = None,
        inject_dummy_wandb: bool = True,
        pythonpath_entries: list[Path],
        wrapper_cwd: Path | None = None,
        child_command: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        child_script = f"{child_code}\nimport sys\nsys.exit({child_exit_code})\n"
        command = (
            child_command
            if child_command is not None
            else [
                sys.executable,
                "-c",
                child_script,
            ]
        )
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--provider",
            provider,
            "--log-dir",
            str(log_dir),
            "--run-name",
            "wrapper-test",
            "--",
            *command,
        ]
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
            cwd=wrapper_cwd,
            env=env,
        )

    def test_stdout_and_stderr_are_captured_in_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_requests_stub(stub_dir)

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
            _write_requests_stub(stub_dir)

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
            _write_requests_stub(stub_dir)

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

            request_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(request_payload["json"], {"state": "stopped"})
            self.assertEqual(
                request_payload["headers"]["Authorization"],
                "Bearer vast-test-key",
            )

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["shutdown"]["return_code"], 0)
            self.assertEqual(metadata["shutdown"]["provider"], "vast_api")
            self.assertEqual(metadata["shutdown"]["interrupt_window_sec"], 30)
            self.assertFalse(metadata["shutdown"]["interrupt_prompt_shown"])
            self.assertFalse(metadata["shutdown"]["aborted_by_user"])
            self.assertEqual(metadata["shutdown"]["skip_reason"], "no_tty_auto_proceed")

    def test_shutdown_abort_input_skips_vast_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "vast_stop.json"
            _write_requests_stub(stub_dir)

            cmd = [
                sys.executable,
                str(WRAPPER_PATH),
                "--provider",
                "vast",
                "--log-dir",
                str(log_dir),
                "--run-name",
                "wrapper-test",
                "--",
                sys.executable,
                "-c",
                "print('done')",
            ]
            env = dict(os.environ)
            env["WANDB_API_KEY"] = "dummy-test-key"
            env["CONTAINER_ID"] = "12345"
            env["CONTAINER_API_KEY"] = "vast-test-key"
            env["VAST_TEST_MARKER"] = str(marker_path)
            env["PYTHONPATH"] = str(stub_dir)

            master_fd, slave_fd = os.openpty()
            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=slave_fd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )
                os.close(slave_fd)
                os.write(master_fd, b"abort\n")
                stdout, stderr = process.communicate(timeout=20)
            finally:
                os.close(master_fd)

            self.assertEqual(process.returncode, 0, msg=f"stdout={stdout}\nstderr={stderr}")
            self.assertFalse(marker_path.exists(), "shutdown call should be skipped")

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["shutdown"]["attempted"])
            self.assertTrue(metadata["shutdown"]["aborted_by_user"])
            self.assertTrue(metadata["shutdown"]["interrupt_prompt_shown"])
            self.assertEqual(metadata["shutdown"]["interrupt_input"], "abort")
            self.assertEqual(metadata["shutdown"]["skip_reason"], "aborted_by_user")
            self.assertIsNone(metadata["shutdown"]["return_code"])

    def test_shutdown_runs_after_non_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "vast_stop.json"
            _write_requests_stub(stub_dir)

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
            _write_requests_stub(stub_dir, status_code=401)

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
            self.assertIn("HTTP 401", metadata["shutdown"]["error"])

    def test_network_error_does_not_override_child_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_requests_stub(stub_dir, should_fail=True)

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
            self.assertIn("Vast API request failed", metadata["shutdown"]["error"])

    def test_missing_command_after_separator_returns_argument_error(self) -> None:
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--provider",
            "vast",
            "--",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing command after '--'", result.stderr)

    def test_missing_provider_returns_argument_error(self) -> None:
        cmd = [
            sys.executable,
            str(WRAPPER_PATH),
            "--log-dir",
            "runs/logs",
            "--run-name",
            "wrapper-test",
            "--",
            sys.executable,
            "-c",
            "print('unused')",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("the following arguments are required: --provider", result.stderr)

    def test_default_pytorch_alloc_conf_is_set_before_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_requests_stub(stub_dir)

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

    def test_project_root_is_injected_into_child_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            child_script_path = tmp_path / "nested" / "child_imports_src.py"
            _write_requests_stub(stub_dir)
            child_script_path.parent.mkdir(parents=True, exist_ok=True)
            child_script_path.write_text(
                "from src.utils.vast import resolve_vast_api_key\n"
                "print('src-import-ok')\n",
                encoding="utf-8",
            )

            result = self._run_wrapper(
                child_code="print('unused')",
                child_exit_code=0,
                child_command=[sys.executable, str(child_script_path)],
                log_dir=log_dir,
                wrapper_cwd=tmp_path,
                wrapper_env={
                    "CONTAINER_ID": "12345",
                    "CONTAINER_API_KEY": "vast-test-key",
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 0)
            log_path = _single_file(log_dir, "*.log")
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("src-import-ok", log_text)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["env"]["pythonpath_injected_project_root"])

    def test_missing_wandb_api_key_fails_without_interactive_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_requests_stub(stub_dir)

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

    def test_shutdown_uses_vast_api_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "vast_stop.json"
            _write_requests_stub(stub_dir)

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
            self.assertTrue(marker_payload["url"].endswith("/instances/12345/"))
            self.assertEqual(marker_payload["headers"]["Authorization"], "Bearer vast-test-key")

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["shutdown"]["provider"], "vast_api")
            self.assertEqual(metadata["shutdown"]["return_code"], 0)
            self.assertEqual(metadata["shutdown"]["vast_instance_id"], 12345)
            self.assertEqual(metadata["env"]["vast_api_key_source"], "CONTAINER_API_KEY")

    def test_missing_vast_api_key_fails_without_interactive_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_requests_stub(stub_dir)

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
            _write_requests_stub(stub_dir)

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

    def test_thunder_shutdown_runs_after_success_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir)

            result = self._run_wrapper(
                provider="thunder",
                child_code="print('done')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={
                    "TNR_INSTANCE_ID": "550e8400-e29b-41d4-a716-446655440000",
                    "TNR_API_TOKEN": "thunder-test-key",
                    "THUNDER_TEST_MARKER": str(marker_path),
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(marker_path.exists())

            requests_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(len(requests_payload), 2)
            self.assertEqual(requests_payload[0]["method"], "GET")
            self.assertTrue(requests_payload[0]["url"].endswith("/instances/list"))
            self.assertEqual(requests_payload[1]["method"], "POST")
            self.assertTrue(requests_payload[1]["url"].endswith("/instances/0/delete"))
            self.assertEqual(
                requests_payload[1]["headers"]["Authorization"],
                "Bearer thunder-test-key",
            )

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["shutdown"]["provider"], "thunder_api")
            self.assertEqual(
                metadata["shutdown"]["thunder_instance_id"],
                "550e8400-e29b-41d4-a716-446655440000",
            )
            self.assertEqual(metadata["shutdown"]["return_code"], 0)
            self.assertFalse(metadata["shutdown"]["create_snapshot"])
            self.assertFalse(metadata["shutdown"]["response"]["create_snapshot"])
            self.assertIsNone(metadata["shutdown"]["response"]["snapshot_name"])
            self.assertEqual(
                metadata["shutdown"]["response"]["resolved_instance_id"],
                "0",
            )

            log_path = _single_file(log_dir, "*.log")
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("[wrapper] running shutdown step | provider=thunder", log_text)
            self.assertIn("[shutdown][thunder]", log_text)

    def test_thunder_shutdown_runs_after_non_zero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir)

            result = self._run_wrapper(
                provider="thunder",
                child_code="print('failing-run')",
                child_exit_code=7,
                log_dir=log_dir,
                wrapper_env={
                    "TNR_INSTANCE_ID": "550e8400-e29b-41d4-a716-446655440000",
                    "TNR_API_TOKEN": "thunder-test-key",
                    "THUNDER_TEST_MARKER": str(marker_path),
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 7)
            self.assertTrue(marker_path.exists())

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["attempted"])
            self.assertEqual(metadata["child_return_code"], 7)

    def test_missing_tnr_instance_id_fails_without_running_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_thunder_requests_stub(stub_dir)

            result = self._run_wrapper(
                provider="thunder",
                child_code="print('will-not-run')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={"TNR_API_TOKEN": "thunder-test-key"},
                unset_env_keys=["TNR_INSTANCE_ID"],
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 2)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 2)
            self.assertIn("TNR_INSTANCE_ID is not set", metadata["env"]["error"])

    def test_missing_tnr_api_token_fails_without_interactive_tty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            _write_thunder_requests_stub(stub_dir)

            result = self._run_wrapper(
                provider="thunder",
                child_code="print('will-not-run')",
                child_exit_code=0,
                log_dir=log_dir,
                wrapper_env={"TNR_INSTANCE_ID": "550e8400-e29b-41d4-a716-446655440000"},
                unset_env_keys=["TNR_API_TOKEN"],
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 2)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["child_return_code"], 2)
            self.assertIn("TNR_API_TOKEN is not set", metadata["env"]["error"])

    def test_thunder_api_token_from_set_env_is_used_for_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir)

            cmd = [
                sys.executable,
                str(WRAPPER_PATH),
                "--provider",
                "thunder",
                "--log-dir",
                str(log_dir),
                "--run-name",
                "wrapper-test",
                "--set-env",
                "TNR_API_TOKEN=thunder-via-cli",
                "--",
                sys.executable,
                "-c",
                "print('done')",
            ]
            env = dict(os.environ)
            env["WANDB_API_KEY"] = "dummy-test-key"
            env["TNR_INSTANCE_ID"] = "550e8400-e29b-41d4-a716-446655440000"
            env["THUNDER_TEST_MARKER"] = str(marker_path)
            env["PYTHONPATH"] = str(stub_dir)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 0)
            requests_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(
                requests_payload[1]["headers"]["Authorization"],
                "Bearer thunder-via-cli",
            )

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["env"]["thunder_api_key_source"], "cli")

    def test_thunder_snapshot_failure_does_not_attempt_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir, snapshot_status_code=500)
            cmd = [
                sys.executable,
                str(WRAPPER_PATH),
                "--provider",
                "thunder",
                "--create-snapshot",
                "--log-dir",
                str(log_dir),
                "--run-name",
                "wrapper-test",
                "--",
                sys.executable,
                "-c",
                "print('failing-child')\nimport sys\nsys.exit(5)\n",
            ]
            env = dict(os.environ)
            env["WANDB_API_KEY"] = "dummy-test-key"
            env["TNR_INSTANCE_ID"] = "550e8400-e29b-41d4-a716-446655440000"
            env["TNR_API_TOKEN"] = "thunder-test-key"
            env["THUNDER_TEST_MARKER"] = str(marker_path)
            env["PYTHONPATH"] = str(stub_dir)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 5)
            requests_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(len(requests_payload), 2)
            self.assertTrue(requests_payload[1]["url"].endswith("/snapshots/create"))

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("HTTP 500", metadata["shutdown"]["error"])
            self.assertIsNone(metadata["shutdown"]["return_code"])
            self.assertTrue(metadata["shutdown"]["create_snapshot"])
            self.assertIn("Traceback", metadata["shutdown"]["error_traceback"])

    def test_thunder_snapshot_failure_after_successful_child_returns_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir, snapshot_status_code=500)

            cmd = [
                sys.executable,
                str(WRAPPER_PATH),
                "--provider",
                "thunder",
                "--create-snapshot",
                "--log-dir",
                str(log_dir),
                "--run-name",
                "wrapper-test",
                "--",
                sys.executable,
                "-c",
                "print('done')",
            ]
            env = dict(os.environ)
            env["WANDB_API_KEY"] = "dummy-test-key"
            env["TNR_INSTANCE_ID"] = "550e8400-e29b-41d4-a716-446655440000"
            env["TNR_API_TOKEN"] = "thunder-test-key"
            env["THUNDER_TEST_MARKER"] = str(marker_path)
            env["PYTHONPATH"] = str(stub_dir)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 1)
            requests_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(len(requests_payload), 2)
            self.assertTrue(requests_payload[1]["url"].endswith("/snapshots/create"))

    def test_thunder_create_snapshot_flag_adds_snapshot_before_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir)

            cmd = [
                sys.executable,
                str(WRAPPER_PATH),
                "--provider",
                "thunder",
                "--create-snapshot",
                "--log-dir",
                str(log_dir),
                "--run-name",
                "wrapper-test",
                "--",
                sys.executable,
                "-c",
                "print('done')",
            ]
            env = dict(os.environ)
            env["WANDB_API_KEY"] = "dummy-test-key"
            env["TNR_INSTANCE_ID"] = "550e8400-e29b-41d4-a716-446655440000"
            env["TNR_API_TOKEN"] = "thunder-test-key"
            env["THUNDER_TEST_MARKER"] = str(marker_path)
            env["PYTHONPATH"] = str(stub_dir)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 0)
            requests_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(len(requests_payload), 3)
            self.assertTrue(requests_payload[1]["url"].endswith("/snapshots/create"))
            self.assertTrue(requests_payload[2]["url"].endswith("/instances/0/delete"))

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["shutdown"]["create_snapshot"])
            self.assertTrue(metadata["shutdown"]["response"]["create_snapshot"])
            self.assertIsNotNone(metadata["shutdown"]["response"]["snapshot_name"])

    def test_thunder_delete_failure_preserves_snapshot_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "logs"
            stub_dir = tmp_path / "stubs"
            marker_path = tmp_path / "thunder_stop.json"
            _write_thunder_requests_stub(stub_dir, delete_status_code=500)

            result = self._run_wrapper(
                provider="thunder",
                child_code="print('failing-child')",
                child_exit_code=5,
                log_dir=log_dir,
                wrapper_env={
                    "TNR_INSTANCE_ID": "550e8400-e29b-41d4-a716-446655440000",
                    "TNR_API_TOKEN": "thunder-test-key",
                    "THUNDER_TEST_MARKER": str(marker_path),
                },
                pythonpath_entries=[stub_dir],
            )

            self.assertEqual(result.returncode, 5)

            metadata_path = _single_file(log_dir, "*.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("HTTP 500", metadata["shutdown"]["error"])
            self.assertIn("snapshot_attempted=False", metadata["shutdown"]["error"])


if __name__ == "__main__":
    unittest.main()
