from __future__ import annotations

import os
import re
from typing import Any, Mapping


VAST_INSTANCE_ID_ENV_KEYS: tuple[str, ...] = ("CONTAINER_ID", "VAST_CONTAINERLABEL")
VAST_API_KEY_ENV_KEYS: tuple[str, ...] = ("CONTAINER_API_KEY", "VAST_API_KEY")
DEFAULT_VAST_API_BASE_URL = "https://console.vast.ai/api/v0"
DEFAULT_VAST_API_TIMEOUT_SEC = 30.0


def parse_vast_instance_id(value: str) -> int:
    candidate = value.strip()
    if not candidate:
        raise ValueError("Vast instance ID cannot be empty.")
    if candidate.isdigit():
        return int(candidate)

    match = re.fullmatch(r"(?:[A-Za-z]+\.)?(\d+)", candidate)
    if match:
        return int(match.group(1))

    raise ValueError(
        "Unsupported Vast instance identifier "
        f"'{value}'. Expected digits or a label like 'C.123456'."
    )


def resolve_vast_instance_id(env: Mapping[str, str]) -> tuple[int | None, str | None]:
    for key in VAST_INSTANCE_ID_ENV_KEYS:
        raw_value = env.get(key, "").strip()
        if raw_value:
            return parse_vast_instance_id(raw_value), key
    return None, None


def resolve_vast_api_key(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    for key in VAST_API_KEY_ENV_KEYS:
        value = env.get(key, "").strip()
        if value:
            return value, key
    return None, None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def stop_vast_instance(*, instance_id: int, api_key: str) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Vast shutdown requires the `requests` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    base_url = os.environ.get("VAST_API_BASE_URL", DEFAULT_VAST_API_BASE_URL).rstrip("/")
    url = f"{base_url}/instances/{instance_id}/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.put(
            url,
            headers=headers,
            json={"state": "stopped"},
            timeout=DEFAULT_VAST_API_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Vast API request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise RuntimeError(
            f"Vast API stop_instance failed with HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
    except ValueError:
        payload = {"success": True, "text": response.text}
    return _json_safe(payload)
