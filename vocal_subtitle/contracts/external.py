"""Framework-neutral conversion at CLI and WebUI boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import ErrorInfo
from .run import RunResult
from .task import TaskRequest


class ExternalAdapter:
    """Convert plain request/response mappings without importing a web framework."""

    @staticmethod
    def task_request(payload: Mapping[str, Any]) -> TaskRequest:
        return TaskRequest.from_dict(payload)

    @staticmethod
    def task_response(result: RunResult) -> dict[str, Any]:
        return result.to_dict()

    @staticmethod
    def error_response(error: ErrorInfo, *, status_code: int = 500) -> dict[str, Any]:
        return {"status_code": status_code, "error": error.to_dict()}


class CLIAdapter(ExternalAdapter):
    """CLI-facing named adapter kept independent from Click."""

    @staticmethod
    def request(
        input_path: str,
        *,
        output_path: str | None = None,
        profile: str = "default",
        mode: str = "offline",
        task_id: str | None = None,
        **overrides: Any,
    ) -> TaskRequest:
        return TaskRequest(
            input_path=input_path,
            output_path=output_path,
            profile=profile,
            mode=mode,
            task_id=task_id,
            overrides=overrides,
        )


class WebUIAdapter(ExternalAdapter):
    """WebUI-facing named adapter for Pydantic/FastAPI boundary payloads."""

    @staticmethod
    def request(payload: Mapping[str, Any] | Any) -> TaskRequest:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif hasattr(payload, "dict"):
            payload = payload.dict()
        return TaskRequest.from_dict(payload)
