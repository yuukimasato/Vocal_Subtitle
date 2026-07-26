"""Deterministic cache keys and JSON-safe IR cache boundaries."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar


T = TypeVar("T")
IR_CACHE_VERSION = "ir-cache-v1"


def _json_safe(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise ValueError(f"unsupported mapping key at {path}")
            key_text = str(key)
            if key_text in result:
                raise ValueError(f"duplicate canonical mapping key at {path}: {key_text}")
            result[key_text] = _json_safe(item, f"{path}.{key_text}")
        return result
    raise ValueError(f"value at {path} is not JSON-safe")


def _canonical_json(payload: Any) -> bytes:
    safe = _json_safe(payload)
    try:
        return json.dumps(
            safe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("payload cannot be encoded as canonical JSON") from exc


def fingerprint_ir(payload: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-safe IR payload."""
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _duration(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("audio_duration must be a finite non-negative number or None")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("audio_duration must be a finite non-negative number or None")
    return result


def _validate_additional_params(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("additional_params must be a mapping")
    forbidden = ("path", "token", "credential", "secret", "password")
    for key in value:
        if not isinstance(key, str):
            raise ValueError("additional_params keys must be strings")
        lowered = key.lower()
        if any(term in lowered for term in forbidden):
            raise ValueError(f"additional_params cannot contain sensitive field: {key}")
    return dict(value)


def make_ir_cache_key(
    *,
    artifact_type: str,
    schema_version: str,
    producer_version: str,
    input_sha256: str,
    audio_duration: Optional[float],
    timeline_fingerprint: str,
    coordinate_policy: str,
    context_policy: str,
    additional_params: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a versioned key without exposing paths or credentials."""
    payload = {
        "ir_cache_version": IR_CACHE_VERSION,
        "artifact_type": _required_text(artifact_type, "artifact_type"),
        "schema_version": _required_text(schema_version, "schema_version"),
        "producer_version": _required_text(producer_version, "producer_version"),
        "input_sha256": _required_text(input_sha256, "input_sha256"),
        "audio_duration": _duration(audio_duration),
        "timeline_fingerprint": _required_text(timeline_fingerprint, "timeline_fingerprint"),
        "coordinate_policy": _required_text(coordinate_policy, "coordinate_policy"),
        "context_policy": _required_text(context_policy, "context_policy"),
        "additional_params": _validate_additional_params(additional_params),
    }
    return fingerprint_ir(payload)


def encode_ir_value(value: Any) -> Dict[str, Any]:
    """Convert an IR ``to_dict()`` result to a JSON-safe dictionary."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("IR cache values must be mappings")
    safe = _json_safe(value)
    if not isinstance(safe, dict):  # defensive; Mapping inputs become dicts
        raise ValueError("IR cache values must encode as dictionaries")
    return safe


def decode_ir_value(
    payload: Any,
    loader: Callable[[Mapping[str, Any]], T],
    *,
    expected_schema_version: Optional[str] = None,
) -> Optional[T]:
    """Load a cache value; all validation/loader failures become a miss."""
    if not isinstance(payload, Mapping):
        return None
    try:
        safe = _json_safe(payload)
        if not isinstance(safe, dict):
            return None
        if expected_schema_version is not None and safe.get("schema_version") != expected_schema_version:
            return None
        return loader(safe)
    except Exception:
        return None


def persist_ir_value(
    cache: Any,
    artifact_type: str,
    key: str,
    value: Any,
    *,
    ttl: Optional[int] = None,
) -> bool:
    """Persist a JSON-safe IR value through a CacheManager-like object."""
    try:
        encoded = encode_ir_value(value)
        cache.set(artifact_type, key, encoded, ttl=ttl)
        return True
    except Exception:
        return False


def load_ir_value(
    cache: Any,
    artifact_type: str,
    key: str,
    loader: Callable[[Mapping[str, Any]], T],
    *,
    expected_schema_version: Optional[str] = None,
) -> Optional[T]:
    """Load a persisted IR value; malformed entries are cache misses."""
    try:
        payload = cache.get(artifact_type, key)
    except Exception:
        return None
    return decode_ir_value(
        payload,
        loader,
        expected_schema_version=expected_schema_version,
    )
