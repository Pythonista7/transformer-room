from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping


TNR_INSTANCE_ID_ENV_KEYS: tuple[str, ...] = ("TNR_INSTANCE_ID",)
TNR_API_KEY_ENV_KEYS: tuple[str, ...] = ("TNR_API_TOKEN",)
DEFAULT_TNR_API_BASE_URL = "https://api.thundercompute.com:8443/v1"
DEFAULT_TNR_TIMEOUT_SEC = 30.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_slug(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S-%fZ")


def resolve_thunder_instance_id(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    for key in TNR_INSTANCE_ID_ENV_KEYS:
        value = env.get(key, "").strip()
        if value:
            return value, key
    return None, None


def resolve_thunder_api_key(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    for key in TNR_API_KEY_ENV_KEYS:
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


def snapshot_and_delete_thunder_instance(
    *,
    instance_id: str,
    api_key: str,
    snapshot_name: str | None = None,
) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Thunder shutdown requires the `requests` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    base_url = os.environ.get("TNR_API_BASE_URL", DEFAULT_TNR_API_BASE_URL).rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if snapshot_name is None:
        snapshot_name = f"autosnap-{instance_id[:8]}-{_timestamp_slug(_utc_now())}"

    try:
        snapshot_response = requests.post(
            f"{base_url}/snapshots/create",
            headers=headers,
            json={"instanceId": instance_id, "name": snapshot_name},
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Thunder API request failed: {exc}") from exc

    if snapshot_response.status_code >= 400:
        detail = snapshot_response.text.strip()
        raise RuntimeError(
            "Thunder API snapshot create failed with "
            f"HTTP {snapshot_response.status_code}: {detail}"
        )

    try:
        delete_response = requests.post(
            f"{base_url}/instances/{instance_id}/delete",
            headers=headers,
            json={},
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Thunder API request failed: {exc}") from exc

    if delete_response.status_code >= 400:
        detail = delete_response.text.strip()
        raise RuntimeError(
            "Thunder API delete instance failed with "
            f"HTTP {delete_response.status_code}: {detail} | "
            f"snapshot_name={snapshot_name}"
        )

    try:
        delete_payload = delete_response.json()
    except ValueError:
        delete_payload = {"success": True, "text": delete_response.text}

    return _json_safe(
        {
            "snapshot_name": snapshot_name,
            "snapshot_status_code": snapshot_response.status_code,
            "delete_response": delete_payload,
        }
    )
