"""Shared, serializable backend contract primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

CONTRACT_VERSION = "backend-contract-v1"
ROUTE_VERSION = "asr-route-v1"
DECISION_POLICY_VERSION = "decision-policy-v1"
REPORT_SCHEMA_VERSION = "run-report-v1"


def jsonable(value: Any) -> Any:
    """Convert common project values without exposing implementation objects."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    if hasattr(value, "__fspath__"):
        return str(value)
    return value


@dataclass(frozen=True)
class ErrorInfo:
    """Stable error envelope used across application and infrastructure ports."""

    category: str
    code: str = ""
    message: str = ""
    retryable: bool = False
    recoverable: bool = False
    stage: str | None = None
    engine: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "stage": self.stage,
            "engine": self.engine,
            "details": jsonable(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> ErrorInfo | None:
        if not payload:
            return None
        return cls(
            category=str(payload.get("category", "unrecoverable_failure")),
            code=str(payload.get("code", "")),
            message=str(payload.get("message", "")),
            retryable=bool(payload.get("retryable", False)),
            recoverable=bool(payload.get("recoverable", False)),
            stage=payload.get("stage"),
            engine=payload.get("engine"),
            details=dict(payload.get("details") or {}),
        )

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        *,
        category: str = "unrecoverable_failure",
        stage: str | None = None,
        engine: str | None = None,
        retryable: bool = False,
        recoverable: bool = False,
    ) -> ErrorInfo:
        """Normalize an exception while deliberately excluding its traceback."""
        return cls(
            category=category,
            code=error.__class__.__name__,
            message=str(error),
            retryable=retryable,
            recoverable=recoverable,
            stage=stage,
            engine=engine,
        )
