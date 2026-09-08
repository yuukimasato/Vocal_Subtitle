"""Task request and task-history contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional

from .common import CONTRACT_VERSION, ErrorInfo, jsonable


class TaskState(str, Enum):
    PENDING = "pending"
    PREFLIGHT = "preflight"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED_COMPLETED = "degraded_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskRequest:
    input_path: str
    output_path: Optional[str] = None
    profile: str = "default"
    mode: str = "offline"
    overrides: Mapping[str, Any] = field(default_factory=dict)
    requested_by: str = ""
    task_id: Optional[str] = None
    contract_version: str = CONTRACT_VERSION

    def with_task_id(self, task_id: str) -> "TaskRequest":
        return replace(self, task_id=task_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "profile": self.profile,
            "mode": self.mode,
            "overrides": jsonable(self.overrides),
            "requested_by": self.requested_by,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskRequest":
        return cls(
            task_id=payload.get("task_id"),
            input_path=str(payload.get("input_path", "")),
            output_path=payload.get("output_path"),
            profile=str(payload.get("profile", "default")),
            mode=str(payload.get("mode", "offline")),
            overrides=dict(payload.get("overrides") or {}),
            requested_by=str(payload.get("requested_by", "")),
            contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
        )


@dataclass
class TaskSnapshot:
    task_id: str
    run_id: Optional[str] = None
    state: TaskState = TaskState.PENDING
    progress: float = 0.0
    stage: str = ""
    status: str = ""
    error: Optional[ErrorInfo] = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, TaskState):
            self.state = TaskState(str(self.state))
        if not self.status:
            self.status = self.state.value
        self.progress = max(0.0, min(1.0, float(self.progress)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "state": self.state.value,
            "progress": self.progress,
            "stage": self.stage,
            "status": self.status or self.state.value,
            "error": self.error.to_dict() if self.error else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": jsonable(self.metadata),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSnapshot":
        state = payload.get("state", payload.get("status", TaskState.PENDING.value))
        return cls(
            task_id=str(payload.get("task_id", payload.get("id", ""))),
            run_id=payload.get("run_id") or None,
            state=TaskState(str(state)),
            progress=float(payload.get("progress", 0.0)),
            stage=str(payload.get("stage", "")),
            status=str(payload.get("status", "")),
            error=ErrorInfo.from_dict(payload.get("error")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", payload.get("completed_at", ""))),
            metadata=dict(payload.get("metadata") or {}),
            contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
        )

    @classmethod
    def from_legacy_row(cls, row: Mapping[str, Any]) -> "TaskSnapshot":
        progress_payload = row.get("progress_json") or "{}"
        if isinstance(progress_payload, str):
            try:
                progress_payload = json.loads(progress_payload)
            except (TypeError, ValueError):
                progress_payload = {}
        progress_payload = progress_payload if isinstance(progress_payload, Mapping) else {}
        raw_progress = progress_payload.get("progress", progress_payload.get("percent", 0.0))
        progress = float(raw_progress or 0.0)
        if progress > 1:
            progress /= 100.0
        raw_error = row.get("error")
        error = None
        if raw_error:
            error = ErrorInfo(
                category=str(row.get("error_category") or "unrecoverable_failure"),
                message=str(raw_error),
                recoverable=row.get("status") == TaskState.DEGRADED_COMPLETED.value,
            )
        return cls(
            task_id=str(row.get("id", row.get("task_id", ""))),
            run_id=row.get("run_id") or None,
            state=TaskState(str(row.get("status", TaskState.PENDING.value))),
            progress=progress,
            stage=str(progress_payload.get("stage", "")),
            error=error,
            created_at=str(row.get("created_at", "")),
            updated_at=str(row.get("completed_at", row.get("created_at", ""))),
            metadata={
                "input_file_name": row.get("input_file_name", ""),
                "input_file_hash": row.get("input_file_hash", ""),
                "profile": row.get("profile", "default"),
                "result_json": row.get("result_json"),
                "total_duration_seconds": row.get("total_duration_seconds", 0),
            },
        )
