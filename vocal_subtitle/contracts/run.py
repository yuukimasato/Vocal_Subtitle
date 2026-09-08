"""Application run contracts and legacy Pipeline result conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .common import CONTRACT_VERSION, ErrorInfo, jsonable
from .task import TaskRequest, TaskSnapshot, TaskState


@dataclass(frozen=True)
class RunRequest:
    task: TaskRequest
    config: Any = None
    input_fingerprint: str = ""
    services: Any = None
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        config = jsonable(self.config) if self.config is not None else None
        return {
            "task": self.task.to_dict(),
            "config": config,
            "input_fingerprint": self.input_fingerprint,
            "contract_version": self.contract_version,
        }


@dataclass
class RunResult:
    task: TaskSnapshot
    subtitle_path: Optional[str] = None
    events: tuple[Any, ...] = ()
    stats: Any = None
    report: Any = None
    artifacts: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: Optional[ErrorInfo] = None
    from_cache: bool = False
    contract_version: str = CONTRACT_VERSION

    @property
    def status(self) -> str:
        return self.task.state.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "subtitle_path": self.subtitle_path,
            "events": [jsonable(event) for event in self.events],
            "stats": jsonable(self.stats),
            "report": jsonable(self.report),
            "artifacts": dict(self.artifacts),
            "diagnostics": jsonable(self.diagnostics),
            "error": self.error.to_dict() if self.error else None,
            "from_cache": self.from_cache,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_legacy(
        cls,
        payload: Mapping[str, Any],
        request: RunRequest,
    ) -> "RunResult":
        stats = payload.get("stats")
        stat_value = (
            stats.get("status")
            if isinstance(stats, Mapping)
            else getattr(stats, "status", "completed")
        )
        status = str(payload.get("status") or stat_value or "completed")
        try:
            state = TaskState(status)
        except ValueError:
            state = TaskState.FAILED
        stat_task_id = stats.get("task_id", "") if isinstance(stats, Mapping) else getattr(stats, "task_id", "")
        stat_run_id = stats.get("run_id", "") if isinstance(stats, Mapping) else getattr(stats, "run_id", "")
        task_id = request.task.task_id or stat_task_id or ""
        run_id = stat_run_id or None
        error = payload.get("error")
        error_info = error if isinstance(error, ErrorInfo) else None
        if error and error_info is None:
            error_info = ErrorInfo(
                category=getattr(stats, "error_category", "") or "unrecoverable_failure",
                message=str(error),
                recoverable=state == TaskState.DEGRADED_COMPLETED,
            )
        events = payload.get("events") or ()
        subtitle_path = payload.get("subtitle_path")
        return cls(
            task=TaskSnapshot(
                task_id=task_id,
                run_id=run_id,
                state=state,
                error=error_info,
                metadata={"legacy_status": status},
            ),
            subtitle_path=str(subtitle_path) if subtitle_path else None,
            events=tuple(events),
            stats=stats,
            artifacts={"subtitle": str(subtitle_path)} if subtitle_path else {},
            diagnostics=dict(payload.get("diagnostics") or {}),
            error=error_info,
            from_cache=bool(payload.get("from_cache", False)),
        )
