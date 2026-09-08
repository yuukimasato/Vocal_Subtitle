"""Protocol-only application coordinator for the backend contract phase."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Optional

from ..contracts.common import ErrorInfo
from ..contracts.ports import ArtifactPort, PipelineRunPort, ReportPort, TaskPort
from ..contracts.run import RunRequest, RunResult
from ..contracts.task import TaskRequest, TaskSnapshot, TaskState


class BackendRunCoordinator:
    """Coordinate task state, execution, artifacts and reporting.

    This class is intentionally independent of Pipeline, FastAPI and Click.
    Existing entry points can adopt it incrementally through adapters.
    """

    def __init__(
        self,
        *,
        tasks: TaskPort,
        runner: PipelineRunPort,
        reports: Optional[ReportPort] = None,
        artifacts: Optional[ArtifactPort] = None,
    ):
        self.tasks = tasks
        self.runner = runner
        self.reports = reports
        self.artifacts = artifacts

    def run(self, request: RunRequest) -> RunResult:
        task = self._ensure_task(request)
        resolved_request = replace(request, task=request.task.with_task_id(task.task_id))
        self._transition(task, TaskState.PREFLIGHT)
        self._transition(task, TaskState.RUNNING)
        try:
            result = self.runner.run(resolved_request)
        except Exception as exc:
            error = ErrorInfo.from_exception(exc)
            self._transition(task, TaskState.FAILED, error=error)
            failed = TaskSnapshot(
                task_id=task.task_id,
                state=TaskState.FAILED,
                error=error,
            )
            return RunResult(task=failed, error=error, diagnostics={"coordinator": "runner_failed"})

        self._register_artifacts(result)
        self._finalize_task(result, task)
        self._attach_report(result, resolved_request)
        return result

    def _ensure_task(self, request: RunRequest) -> TaskSnapshot:
        if request.task.task_id:
            existing = self.tasks.get(request.task.task_id)
            if existing is not None:
                return existing
        return self.tasks.create(
            request.task,
            input_fingerprint=request.input_fingerprint,
            config=request.config,
        )

    def _transition(
        self,
        task: TaskSnapshot,
        state: TaskState,
        *,
        error: Optional[ErrorInfo] = None,
    ) -> TaskSnapshot:
        if task.state == state:
            return task
        updated = self.tasks.transition(task.task_id, state, run_id=task.run_id or "", error=error)
        task.state = updated.state
        task.run_id = updated.run_id
        return updated

    def _finalize_task(self, result: RunResult, original: TaskSnapshot) -> None:
        state = result.task.state
        if state not in {
            TaskState.COMPLETED,
            TaskState.DEGRADED_COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        }:
            state = TaskState.DEGRADED_COMPLETED if result.diagnostics.get("degraded") else TaskState.COMPLETED
        error = result.error or result.task.error
        try:
            self.tasks.transition(original.task_id, state, run_id=result.task.run_id or "", error=error)
        except Exception:
            # A legacy runner may already have finalized the same task.
            current = self.tasks.get(original.task_id)
            if current is None or current.state != state:
                raise

    def _register_artifacts(self, result: RunResult) -> None:
        if self.artifacts is None:
            return
        for name, path in result.artifacts.items():
            self.artifacts.register(name, path)

    def _attach_report(self, result: RunResult, request: RunRequest) -> None:
        if self.reports is None:
            return
        try:
            report = self.reports.build(result, config_snapshot=_config_snapshot(request.config))
            result.report = report
            self.reports.persist(report, config=request.config)
        except Exception as exc:
            # Reporting is secondary; preserve the completed subtitle result.
            result.diagnostics["report_error"] = ErrorInfo.from_exception(
                exc, category="output_failed", recoverable=True
            ).to_dict()


def _config_snapshot(config: Any) -> Mapping[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return config.to_dict()
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(config)
    if isinstance(config, Mapping):
        return config
    return {}


# Short alias for consumers that do not need the implementation detail name.
RunCoordinator = BackendRunCoordinator
