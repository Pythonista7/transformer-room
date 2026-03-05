"""
Example:
  .venv/bin/python baseline/utils/run_and_shutdown.py \
    --shutdown-cmd "sudo shutdown -h now" \
    --log-dir runs/logs \
    --run-name optim-adam-vs-adamw \
    -- .venv/bin/python baseline/experiments/OptimAdamVsW.py
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PYTORCH_ALLOC_CONF = "expandable_segments:True"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _timestamp_slug(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S-%fZ")


def _sanitize_slug(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return sanitized or "run"


def _render_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


class _Tee:
    def __init__(self, log_file) -> None:
        self._log_file = log_file

    def write_line(self, line: str) -> None:
        sys.stdout.write(line)
        sys.stdout.flush()
        self._log_file.write(line)
        self._log_file.flush()

    def write_message(self, message: str) -> None:
        self.write_line(message.rstrip("\n") + "\n")


def _parse_env_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE format.")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Environment variable key cannot be empty.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise argparse.ArgumentTypeError(
            f"Invalid environment variable key '{key}'. "
            "Expected pattern [A-Za-z_][A-Za-z0-9_]*."
        )
    return key, raw_value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an experiment command, tee stdout/stderr to a log file, "
            "persist metadata, then run a shutdown command."
        )
    )
    parser.add_argument(
        "--shutdown-cmd",
        required=True,
        help="Shell command to execute after the experiment process exits.",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/logs",
        help="Directory where timestamped .log and .json files are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run slug used in log/metadata filenames.",
    )
    parser.add_argument(
        "--shutdown-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout in seconds for the shutdown command.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        type=_parse_env_assignment,
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable assignment for the child process. Repeatable.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'.",
    )

    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after '--'")
    if args.shutdown_timeout_sec <= 0:
        parser.error("--shutdown-timeout-sec must be > 0")

    args.command = command
    return args


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _forward_signal_to_child(signum: int, child: subprocess.Popen[str] | None) -> None:
    if child is None or child.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(child.pid, signum)
        else:
            child.send_signal(signum)
    except ProcessLookupError:
        return


def _prepare_child_env(
    *,
    extra_env: list[tuple[str, str]],
    tee: _Tee,
) -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(os.environ)
    extra_env_keys: set[str] = set()
    for key, value in extra_env:
        env[key] = value
        extra_env_keys.add(key)

    env_summary: dict[str, Any] = {
        "extra_env_keys": sorted(extra_env_keys),
        "pytorch_alloc_conf_source": None,
        "pytorch_alloc_conf": None,
        "wandb_api_key_set": False,
        "wandb_api_key_source": None,
    }

    alloc_conf = env.get("PYTORCH_ALLOC_CONF", "").strip()
    if alloc_conf:
        source = "cli" if "PYTORCH_ALLOC_CONF" in extra_env_keys else "existing"
    else:
        env["PYTORCH_ALLOC_CONF"] = DEFAULT_PYTORCH_ALLOC_CONF
        alloc_conf = DEFAULT_PYTORCH_ALLOC_CONF
        source = "defaulted"
    env_summary["pytorch_alloc_conf_source"] = source
    env_summary["pytorch_alloc_conf"] = alloc_conf

    wandb_value = env.get("WANDB_API_KEY", "").strip()
    if wandb_value:
        env_summary["wandb_api_key_set"] = True
        env_summary["wandb_api_key_source"] = (
            "cli" if "WANDB_API_KEY" in extra_env_keys else "existing"
        )
        return env, env_summary

    tee.write_message("[wrapper] WANDB_API_KEY is not set; prompting for input.")
    if not sys.stdin.isatty():
        raise RuntimeError(
            "WANDB_API_KEY is not set and no interactive TTY is available. "
            "Set WANDB_API_KEY in your environment or pass --set-env WANDB_API_KEY=..."
        )

    while True:
        entered = getpass.getpass("Enter WANDB_API_KEY: ").strip()
        if entered:
            env["WANDB_API_KEY"] = entered
            env_summary["wandb_api_key_set"] = True
            env_summary["wandb_api_key_source"] = "prompted"
            return env, env_summary
        print("WANDB_API_KEY cannot be empty.", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))

    start_time = _utc_now()
    command = list(args.command)
    command_display = _render_command(command)
    run_slug = _sanitize_slug(args.run_name or Path(command[0]).name)
    timestamp = _timestamp_slug(start_time)

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{timestamp}__{run_slug}"
    log_path = log_dir / f"{base_name}.log"
    metadata_path = log_dir / f"{base_name}.json"

    metadata: dict[str, Any] = {
        "command": command,
        "command_display": command_display,
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "wrapper_pid": os.getpid(),
        "pid": None,
        "start_time_utc": _utc_iso(start_time),
        "end_time_utc": None,
        "duration_sec": None,
        "child_return_code": None,
        "log_file": str(log_path.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "received_signals": [],
        "env": {},
        "shutdown": {
            "command": args.shutdown_cmd,
            "attempted": False,
            "timeout_sec": float(args.shutdown_timeout_sec),
            "start_time_utc": None,
            "end_time_utc": None,
            "duration_sec": None,
            "return_code": None,
            "error": None,
        },
    }

    child: subprocess.Popen[str] | None = None
    observed_signals: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame) -> None:
        observed_signals.append(signum)
        _forward_signal_to_child(signum, child)

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    for handled in handled_signals:
        previous_handlers[handled] = signal.getsignal(handled)
        signal.signal(handled, handle_signal)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = _Tee(log_file)
            tee.write_message(f"[wrapper] start_time_utc={metadata['start_time_utc']}")
            tee.write_message(f"[wrapper] log_file={log_path.resolve()}")
            tee.write_message(f"[wrapper] metadata_file={metadata_path.resolve()}")
            tee.write_message(f"[wrapper] command={command_display}")

            try:
                child_env, env_summary = _prepare_child_env(
                    extra_env=list(args.set_env),
                    tee=tee,
                )
                metadata["env"] = env_summary
                tee.write_message(
                    "[wrapper] env prepared | "
                    f"PYTORCH_ALLOC_CONF={env_summary['pytorch_alloc_conf']} "
                    f"({env_summary['pytorch_alloc_conf_source']}) | "
                    f"WANDB_API_KEY source={env_summary['wandb_api_key_source']}"
                )
            except RuntimeError as exc:
                metadata["env"] = {"error": str(exc)}
                end_time = _utc_now()
                metadata["end_time_utc"] = _utc_iso(end_time)
                metadata["duration_sec"] = round(
                    (end_time - start_time).total_seconds(),
                    6,
                )
                metadata["child_return_code"] = 2
                tee.write_message(f"[wrapper] env setup failed: {exc}")
                _write_metadata(metadata_path, metadata)
                return 2

            child_return_code = 127
            try:
                child = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    errors="replace",
                    start_new_session=True,
                    env=child_env,
                )
                metadata["pid"] = child.pid
                tee.write_message(f"[wrapper] child_pid={child.pid}")

                assert child.stdout is not None
                for output_line in child.stdout:
                    tee.write_line(output_line)
                child.stdout.close()
                child_return_code = child.wait()
            except FileNotFoundError as exc:
                metadata["child_start_error"] = str(exc)
                tee.write_message(f"[wrapper] failed to start child command: {exc}")
            except Exception as exc:  # pragma: no cover - defensive path.
                metadata["child_start_error"] = str(exc)
                tee.write_message(f"[wrapper] unexpected child process error: {exc}")
            finally:
                end_time = _utc_now()
                metadata["end_time_utc"] = _utc_iso(end_time)
                metadata["duration_sec"] = round(
                    (end_time - start_time).total_seconds(),
                    6,
                )
                metadata["child_return_code"] = int(child_return_code)
                metadata["received_signals"] = [_signal_name(sig) for sig in observed_signals]
                tee.write_message(f"[wrapper] child_return_code={child_return_code}")

                # Persist run metadata before shutdown so child results survive a failed shutdown.
                _write_metadata(metadata_path, metadata)

                shutdown_meta = metadata["shutdown"]
                shutdown_meta["attempted"] = True
                shutdown_start = _utc_now()
                shutdown_meta["start_time_utc"] = _utc_iso(shutdown_start)
                tee.write_message("[wrapper] running shutdown command")
                tee.write_message(
                    f"[wrapper] shutdown_cmd={args.shutdown_cmd} | timeout={args.shutdown_timeout_sec}s"
                )

                try:
                    shutdown_result = subprocess.run(
                        args.shutdown_cmd,
                        shell=True,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=float(args.shutdown_timeout_sec),
                    )
                    shutdown_meta["return_code"] = int(shutdown_result.returncode)
                    if shutdown_result.stdout:
                        tee.write_message("[wrapper] shutdown stdout:")
                        for line in shutdown_result.stdout.splitlines():
                            tee.write_message(f"[shutdown][stdout] {line}")
                    if shutdown_result.stderr:
                        tee.write_message("[wrapper] shutdown stderr:")
                        for line in shutdown_result.stderr.splitlines():
                            tee.write_message(f"[shutdown][stderr] {line}")
                    if shutdown_result.returncode != 0:
                        shutdown_meta["error"] = (
                            f"shutdown command exited with code {shutdown_result.returncode}"
                        )
                        tee.write_message(
                            f"[wrapper] shutdown command failed with code {shutdown_result.returncode}"
                        )
                except subprocess.TimeoutExpired as exc:
                    shutdown_meta["error"] = (
                        f"shutdown command timed out after {args.shutdown_timeout_sec}s"
                    )
                    tee.write_message(f"[wrapper] {shutdown_meta['error']}")
                    if exc.stdout:
                        for line in str(exc.stdout).splitlines():
                            tee.write_message(f"[shutdown][stdout] {line}")
                    if exc.stderr:
                        for line in str(exc.stderr).splitlines():
                            tee.write_message(f"[shutdown][stderr] {line}")
                except Exception as exc:  # pragma: no cover - defensive path.
                    shutdown_meta["error"] = str(exc)
                    tee.write_message(f"[wrapper] shutdown command raised: {exc}")
                finally:
                    shutdown_end = _utc_now()
                    shutdown_meta["end_time_utc"] = _utc_iso(shutdown_end)
                    shutdown_meta["duration_sec"] = round(
                        (shutdown_end - shutdown_start).total_seconds(),
                        6,
                    )
                    _write_metadata(metadata_path, metadata)

            return child_return_code
    finally:
        for handled in handled_signals:
            signal.signal(handled, previous_handlers[handled])


if __name__ == "__main__":
    raise SystemExit(main())
