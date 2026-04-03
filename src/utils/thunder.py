from __future__ import annotations

import os
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
    for key in TNR_INSTANCE_ID_ENV_KEYS:
        value = env.get(key, "").strip()
        if value:
            return value, key
    return None, None


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


def _resolve_base_urls() -> tuple[str, str]:
    configured = os.environ.get("TNR_API_BASE_URL", DEFAULT_TNR_API_BASE_URL).strip()
    base_url = configured.rstrip("/") or DEFAULT_TNR_API_BASE_URL
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")], base_url
    return base_url, f"{base_url}/v1"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_instance_name(payload: Any, instance_id: str) -> str | None:
    name_keys = ("instance_name", "instanceName", "name")
    id_keys = ("identifier", "instance_id", "instanceId", "id", "uuid")

    if isinstance(payload, dict):
        direct_match = payload.get(instance_id)
        if isinstance(direct_match, dict):
            for key in name_keys:
                name = _non_empty_str(direct_match.get(key))
                if name:
                    return name

        for key in id_keys:
            candidate_id = _non_empty_str(payload.get(key))
            if candidate_id == instance_id:
                for name_key in name_keys:
                    name = _non_empty_str(payload.get(name_key))
                    if name:
                        return name

        for value in payload.values():
            name = _extract_instance_name(value, instance_id)
            if name:
                return name

    if isinstance(payload, list):
        for item in payload:
            name = _extract_instance_name(item, instance_id)
            if name:
                return name

    return None


def _lookup_instance_name(
    *,
    requests_module: Any,
    api_key: str,
    instance_id: str,
    v1_base_url: str,
) -> str:
    try:
        response = requests_module.get(
            f"{v1_base_url}/instances/list",
            headers=_headers(api_key),
            timeout=DEFAULT_TNR_TIMEOUT_SEC,
        )
    except requests_module.RequestException as exc:
        raise RuntimeError(f"Thunder API list instances request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text.strip()
        raise RuntimeError(
            "Thunder API list instances failed with "
            f"HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Thunder API list instances returned non-JSON data while resolving "
            f"instance_name for instance_id={instance_id}"
        ) from exc

    instance_name = _extract_instance_name(payload, instance_id)
    if instance_name is None:
        raise RuntimeError(
            "Thunder API list instances did not include an instance_name for "
            f"instance_id={instance_id}"
        )
    return instance_name


def _format_http_error(*, label: str, response: Any) -> str:
    detail = response.text.strip()
    return f"{label} failed with HTTP {response.status_code}: {detail}"


def snapshot_and_delete_thunder_instance(
    *,
    instance_id: str,
    api_key: str,
    instance_name: str | None = None,
    snapshot_name: str | None = None,
) -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Thunder shutdown requires the `requests` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    base_url, v1_base_url = _resolve_base_urls()
    headers = _headers(api_key)

    if snapshot_name is None:
        snapshot_name = f"autosnap-{instance_id[:8]}-{_timestamp_slug(_utc_now())}"

    snapshot_response = None
    snapshot_strategy = None
    snapshot_errors: list[str] = []
    resolved_instance_name = _non_empty_str(instance_name)
    should_try_legacy_snapshot = False

    if resolved_instance_name is None:
        try:
            resolved_instance_name = _lookup_instance_name(
                requests_module=requests,
                api_key=api_key,
                instance_id=instance_id,
                v1_base_url=v1_base_url,
            )
        except RuntimeError as exc:
            snapshot_errors.append(str(exc))
            should_try_legacy_snapshot = True

    if resolved_instance_name is not None:
        try:
            candidate_response = requests.post(
                f"{base_url}/instances/snapshot",
                headers=headers,
                json={"name": snapshot_name, "instance_name": resolved_instance_name},
                timeout=DEFAULT_TNR_TIMEOUT_SEC,
            )
        except requests.RequestException as exc:
            snapshot_errors.append(f"Thunder API request failed: {exc}")
        else:
            if candidate_response.status_code < 400:
                snapshot_response = candidate_response
                snapshot_strategy = "instance_name"
            else:
                snapshot_errors.append(
                    _format_http_error(
                        label="Thunder API snapshot create",
                        response=candidate_response,
                    )
                )
                should_try_legacy_snapshot = candidate_response.status_code in {
                    404,
                    405,
                    422,
                }

    if snapshot_response is None and should_try_legacy_snapshot:
        try:
            candidate_response = requests.post(
                f"{v1_base_url}/snapshots/create",
                headers=headers,
                json={"instanceId": instance_id, "name": snapshot_name},
                timeout=DEFAULT_TNR_TIMEOUT_SEC,
            )
        except requests.RequestException as exc:
            snapshot_errors.append(f"Thunder API request failed: {exc}")
        else:
            if candidate_response.status_code < 400:
                snapshot_response = candidate_response
                snapshot_strategy = "instance_id"
            else:
                snapshot_errors.append(
                    _format_http_error(
                        label="Thunder API snapshot create",
                        response=candidate_response,
                    )
                )

    if snapshot_response is None:
        raise RuntimeError(" | ".join(snapshot_errors))

    try:
        delete_response = requests.post(
            f"{v1_base_url}/instances/{instance_id}/delete",
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
            "instance_name": resolved_instance_name,
            "snapshot_name": snapshot_name,
            "snapshot_status_code": snapshot_response.status_code,
            "snapshot_strategy": snapshot_strategy,
            "delete_response": delete_payload,
        }
    )
