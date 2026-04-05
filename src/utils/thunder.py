from __future__ import annotations

import os
import re
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


def resolve_thunder_instance_id(
    env: Mapping[str, str],
) -> tuple[str | None, str | None]:
    for key in TNR_INSTANCE_ID_ENV_KEYS:
        value = env.get(key, "").strip()
        if value:
            return value, key

    try:
        hostname = subprocess.check_output(["hostname"], text=True).strip()
        host_label = hostname.split(".", 1)[0].strip().lower()
        hostname_matches = re.findall(
            r"(?<![a-z0-9])([a-z0-9]{8})(?![a-z0-9])", host_label
        )
        for candidate in reversed(hostname_matches):
            if any(char.isdigit() for char in candidate):
                return candidate, "hostname"
    except Exception:
        pass

    input_instance_id = input(
        "Enter this instance id:"
    )  # Ideally it should not come to this.

    return input_instance_id, "console"


def resolve_thunder_instance_name(
    env: Mapping[str, str],
) -> tuple[str | None, str | None]:
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


def _list_thunder_instances(*, requests: Any, api_key: str) -> dict[str, Any]:
    v1_base_url = _resolve_v1_base_url()
    list_url = f"{v1_base_url}/instances/list"
    try:
        list_response = requests.get(
            list_url,
            headers=_headers(api_key),
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Thunder API list instances request failed (GET {list_url}): {exc}"
        ) from exc

    try:
        list_payload: Any = list_response.json()
    except ValueError:
        list_payload = {"text": list_response.text}

    if list_response.status_code >= 400:
        raise RuntimeError(
            f"Thunder API list instances failed | "
            f"GET {list_url} | "
            f"{_response_detail(list_response)} | "
            f"response_body={list_payload}"
        )

    if not isinstance(list_payload, dict):
        raise RuntimeError(
            f"Thunder API list instances returned unsupported payload type: "
            f"{type(list_payload).__name__}"
        )

    return list_payload


def _resolve_control_plane_instance_id(
    *,
    instances_payload: Mapping[str, Any],
    instance_ref: str,
    instance_name: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    instance_ref = instance_ref.strip()
    if not instance_ref and instance_name:
        instance_ref = instance_name.strip()

    if not instance_ref:
        raise RuntimeError("Thunder instance reference cannot be empty.")

    direct_match = instances_payload.get(instance_ref)
    if isinstance(direct_match, dict):
        return instance_ref, direct_match

    candidate_refs = {instance_ref}
    if instance_name:
        candidate_refs.add(instance_name.strip())

    for control_plane_id, raw_instance in instances_payload.items():
        if not isinstance(raw_instance, dict):
            continue

        candidate_values = {
            str(control_plane_id).strip(),
            str(raw_instance.get("uuid", "")).strip(),
            str(raw_instance.get("name", "")).strip(),
            str(raw_instance.get("instance_name", "")).strip(),
            str(raw_instance.get("identifier", "")).strip(),
        }
        candidate_values.discard("")
        if candidate_refs & candidate_values:
            return str(control_plane_id), raw_instance

    available_refs: list[str] = []
    for control_plane_id, raw_instance in instances_payload.items():
        if not isinstance(raw_instance, dict):
            continue
        uuid_value = str(raw_instance.get("uuid", "")).strip()
        name_value = str(raw_instance.get("name", "")).strip()
        display = uuid_value or name_value or str(control_plane_id)
        available_refs.append(f"{control_plane_id}:{display}")

    raise RuntimeError(
        "Thunder instance could not be resolved from /instances/list | "
        f"provided_ref={instance_ref!r} | "
        f"provided_name={instance_name!r} | "
        f"available={available_refs}"
    )


def _is_delete_instance_not_found(
    *,
    status_code: int,
    payload: Any,
) -> bool:
    """Treat Thunder delete 404s as idempotent when the instance is already gone."""
    if status_code != 404 or not isinstance(payload, dict):
        return False

    error_code = str(payload.get("error", "")).strip().lower()
    message = str(payload.get("message", "")).strip().lower()
    return error_code == "instance_not_found" or "instance not found" in message


def snapshot_and_delete_thunder_instance(
    *,
    instance_id: str,
    api_key: str,
    instance_name: str | None = None,  # unused — kept for call-site compatibility
    create_snapshot: bool = False,
    snapshot_name: str | None = None,
) -> Any:
    """Optionally snapshot, then delete a Thunder Compute instance.

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
    instances_payload = _list_thunder_instances(requests=requests, api_key=api_key)
    control_plane_instance_id, resolved_instance = _resolve_control_plane_instance_id(
        instances_payload=instances_payload,
        instance_ref=instance_id,
        instance_name=instance_name,
    )
    instance_display_ref = (
        (
            str(resolved_instance.get("uuid", "")).strip()
            if isinstance(resolved_instance, dict)
            else ""
        )
        or (
            str(resolved_instance.get("name", "")).strip()
            if isinstance(resolved_instance, dict)
            else ""
        )
        or instance_id
    )
    snapshot_response: Any = None
    snapshot_payload: Any = None
    snapshot_url: str | None = None

    if create_snapshot and snapshot_name is None:
        snapshot_name = (
            f"autosnap-{instance_display_ref[:8]}-{_timestamp_slug(_utc_now())}"
        )

    if create_snapshot:
        snapshot_url = f"{v1_base_url}/snapshots/create"
        snapshot_body = {"instanceId": control_plane_instance_id, "name": snapshot_name}
        try:
            snapshot_response = requests.post(
                snapshot_url,
                headers=headers,
                json=snapshot_body,
                timeout=DEFAULT_TNR_TIMEOUT_SEC,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Thunder API snapshot request failed | "
                f"POST {snapshot_url} | "
                f"request={snapshot_body} | "
                f"exception={exc}"
            ) from exc

        try:
            snapshot_payload = snapshot_response.json()
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
    delete_url = f"{v1_base_url}/instances/{control_plane_instance_id}/delete"
    try:
        delete_response = requests.post(
            delete_url,
            headers=headers,
            json={},
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Thunder API delete request failed | "
            f"POST {delete_url} | "
            f"request={{}} | "
            f"exception={exc} | "
            f"snapshot_attempted={create_snapshot} | "
            f"snapshot_name={snapshot_name}"
        ) from exc

    try:
        delete_payload: Any = delete_response.json()
    except ValueError:
        delete_payload = {"text": delete_response.text}

    if delete_response.status_code >= 400:
        if _is_delete_instance_not_found(
            status_code=delete_response.status_code,
            payload=delete_payload,
        ):
            return _json_safe(
                {
                    "create_snapshot": create_snapshot,
                    "snapshot_name": snapshot_name,
                    "instance_ref": instance_id,
                    "resolved_instance_id": control_plane_instance_id,
                    "resolved_instance": resolved_instance,
                    "snapshot_url": snapshot_url,
                    "snapshot_status_code": (
                        snapshot_response.status_code
                        if snapshot_response is not None
                        else None
                    ),
                    "snapshot_response": snapshot_payload,
                    "delete_url": delete_url,
                    "delete_status_code": delete_response.status_code,
                    "delete_response": delete_payload,
                    "delete_already_absent": True,
                }
            )
        raise RuntimeError(
            f"Thunder API delete instance failed | "
            f"POST {delete_url} | "
            f"request={{}} | "
            f"{_response_detail(delete_response)} | "
            f"response_body={delete_payload} | "
            f"snapshot_attempted={create_snapshot} | "
            f"snapshot_name={snapshot_name}"
        )

    return _json_safe(
        {
            "create_snapshot": create_snapshot,
            "snapshot_name": snapshot_name,
            "instance_ref": instance_id,
            "resolved_instance_id": control_plane_instance_id,
            "resolved_instance": resolved_instance,
            "snapshot_url": snapshot_url,
            "snapshot_status_code": (
                snapshot_response.status_code if snapshot_response is not None else None
            ),
            "snapshot_response": snapshot_payload,
            "delete_url": delete_url,
            "delete_status_code": delete_response.status_code,
            "delete_response": delete_payload,
        }
    )
