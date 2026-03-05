from __future__ import annotations

import re
from typing import Any, Mapping


VAST_INSTANCE_ID_ENV_KEYS: tuple[str, ...] = ("CONTAINER_ID", "VAST_CONTAINERLABEL")
VAST_API_KEY_ENV_KEYS: tuple[str, ...] = ("CONTAINER_API_KEY", "VAST_API_KEY")


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
        from vastai_sdk import VastAI
    except ImportError as exc:  # pragma: no cover - exercised via wrapper integration.
        raise RuntimeError(
            "Vast shutdown requires the `vastai-sdk` package. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    client = VastAI(api_key=api_key)
    response = client.stop_instance(ID=instance_id)
    return _json_safe(response)
