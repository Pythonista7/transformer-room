from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping


TNR_INSTANCE_ID_ENV_KEYS: tuple[str, ...] = ("TNR_INSTANCE_ID",)
TNR_INSTANCE_NAME_ENV_KEYS: tuple[str, ...] = ("TNR_INSTANCE_NAME",)
TNR_API_KEY_ENV_KEYS: tuple[str, ...] = ("TNR_API_TOKEN",)
DEFAULT_TNR_API_BASE_URL = "https://api.thundercompute.com:8443"
DEFAULT_TNR_TIMEOUT_SEC = 30.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_slug(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S-%fZ")


def resolve_thunder_instance_id(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    try:
        hostname = subprocess.check_output(["hostname"], text=True).strip()
        instance_id = hostname.split("-")[1] if "-" in hostname else ""
        if instance_id:
            return instance_id, "hostname"
    except Exception:
        pass
    
    for key in TNR_INSTANCE_ID_ENV_KEYS:
        value = env.get(key, "").strip()
        if value:
            return value, key
    
    input_instance_id = input("Enter this instance id:")
    
    return input_instance_id, "console"


def resolve_thunder_instance_name(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    for key in TNR_INSTANCE_NAME_ENV_KEYS:
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


def _resolve_v1_base_url() -> str:
    configured = os.environ.get("TNR_API_BASE_URL", DEFAULT_TNR_API_BASE_URL).strip()
    base_url = configured.rstrip("/") or DEFAULT_TNR_API_BASE_URL
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _response_detail(response: Any) -> str:
    """Return a compact status + body string for error messages."""
    return f"HTTP {response.status_code}: {response.text.strip()}"


def snapshot_and_delete_thunder_instance(
    *,
    instance_id: str,
    api_key: str,
    instance_name: str | None = None,  # unused — kept for call-site compatibility
    snapshot_name: str | None = None,
) -> Any:
    """Snapshot then delete a Thunder Compute instance.

    API reference:
      POST /v1/snapshots/create  {"instanceId": str, "name": str}  -> 202 {"message": str}
      POST /v1/instances/{id}/delete  {}                            -> 200 {"message": "Success"}
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Thunder shutdown requires the `requests` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    v1_base_url = _resolve_v1_base_url()
    headers = _headers(api_key)

    if snapshot_name is None:
        snapshot_name = f"autosnap-{instance_id[:8]}-{_timestamp_slug(_utc_now())}"

    # --- snapshot ---
    snapshot_url = f"{v1_base_url}/snapshots/create"
    snapshot_body = {"instanceId": instance_id, "name": snapshot_name}
    try:
        snapshot_response = requests.post(
            snapshot_url,
            headers=headers,
            json=snapshot_body,
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Thunder API snapshot request failed (POST {snapshot_url}): {exc}"
        ) from exc

    try:
        snapshot_payload: Any = snapshot_response.json()
    except ValueError:
        snapshot_payload = {"text": snapshot_response.text}

    if snapshot_response.status_code >= 400:
        raise RuntimeError(
            f"Thunder API snapshot create failed | "
            f"POST {snapshot_url} | "
            f"request={snapshot_body} | "
            f"{_response_detail(snapshot_response)} | "
            f"response_body={snapshot_payload}"
        )

    # --- delete ---
    delete_url = f"{v1_base_url}/instances/{instance_id}/delete"
    try:
        delete_response = requests.post(
            delete_url,
            headers=headers,
            json={},
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Thunder API delete request failed (POST {delete_url}): {exc} | "
            f"snapshot_name={snapshot_name} (snapshot already created)"
        ) from exc

    try:
        delete_payload: Any = delete_response.json()
    except ValueError:
        delete_payload = {"text": delete_response.text}

    if delete_response.status_code >= 400:
        raise RuntimeError(
            f"Thunder API delete instance failed | "
            f"POST {delete_url} | "
            f"{_response_detail(delete_response)} | "
            f"response_body={delete_payload} | "
            f"snapshot_name={snapshot_name} (snapshot already created)"
        )

    return _json_safe(
        {
            "snapshot_name": snapshot_name,
            "snapshot_url": snapshot_url,
            "snapshot_status_code": snapshot_response.status_code,
            "snapshot_response": snapshot_payload,
            "delete_url": delete_url,
            "delete_status_code": delete_response.status_code,
            "delete_response": delete_payload,
        }
    )
