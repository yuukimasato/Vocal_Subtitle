"""Compatibility adapters for the existing backend implementations."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from .common import ErrorInfo
from .engine import (
    EngineAvailability,
    EngineIdentity,
    EngineRequest,
    EngineResult,
    PrepareResult,
)
from .ports import (
    ArtifactPort,
    EnginePort,
    EngineRegistryPort,
    PipelineRunPort,
    ReportPort,
    TaskPort,
)
from .report import RunReport
from .run import RunRequest, RunResult
from .task import TaskRequest, TaskSnapshot, TaskState


@dataclass
class _FallbackConfig:
    """Minimal dataclass for legacy history rows when no config is supplied."""

    mode: str = "offline"
    profile: str = "default"


class TaskHistoryAdapter(TaskPort):
    """Expose ``TaskHistoryManager`` through the versioned task port."""

    def __init__(self, manager: Any):
        self.manager = manager

    def get(self, task_id: str) -> Optional[TaskSnapshot]:
        row = self.manager.get(task_id)
        return TaskSnapshot.from_legacy_row(row) if row else None

    def create(
        self,
        request: TaskRequest,
        *,
        input_fingerprint: str = "",
        config: Any = None,
    ) -> TaskSnapshot:
        task_id = request.task_id or uuid.uuid4().hex
        path = Path(request.input_path)
        try:
            file_size = path.stat().st_size
        except OSError:
            file_size = 0
        self.manager.create(
            task_id,
            path.name,
            input_fingerprint,
            file_size,
            request.profile,
            config if config is not None else _FallbackConfig(profile=request.profile),
        )
        snapshot = self.get(task_id)
        if snapshot is None:
            raise RuntimeError(f"Task history did not create task {task_id!r}")
        return snapshot

    def transition(
        self,
        task_id: str,
        state: TaskState,
        *,
        run_id: str = "",
        error: Any = None,
    ) -> TaskSnapshot:
        error_info = error if isinstance(error, ErrorInfo) else None
        message = error_info.message if error_info else str(error or "")
        category = error_info.category if error_info else ""
        if state == TaskState.PREFLIGHT:
            self.manager.set_preflight(task_id)
        elif state == TaskState.RUNNING:
            self.manager.set_running(task_id, run_id=run_id)
        elif state == TaskState.COMPLETED:
            self.manager.set_completed(task_id)
        elif state == TaskState.DEGRADED_COMPLETED:
            self.manager.set_degraded_completed(task_id, reason=message, category=category)
        elif state == TaskState.FAILED:
            self.manager.set_failed(
                task_id,
                reason=message or "task failed",
                category=category,
            )
        elif state == TaskState.CANCELLED:
            self.manager.set_cancelled(task_id)
        else:
            raise ValueError(f"Unsupported task transition: {state!r}")
        snapshot = self.get(task_id)
        if snapshot is None:
            raise RuntimeError(f"Unknown task after transition: {task_id!r}")
        return snapshot


class PipelineRunAdapter(PipelineRunPort):
    """Translate the legacy ``Pipeline.run`` dictionary into ``RunResult``."""

    def __init__(self, pipeline_or_runner: Any):
        self._runner = (
            pipeline_or_runner.run
            if hasattr(pipeline_or_runner, "run")
            else pipeline_or_runner
        )
        if not callable(self._runner):
            raise TypeError("pipeline_or_runner must expose run() or be callable")

    def run(self, request: RunRequest) -> RunResult:
        task = request.task
        overrides = dict(task.overrides or {})
        output_format = str(overrides.pop("output_format", "srt"))
        kwargs: dict[str, Any] = {
            "input_path": Path(task.input_path),
            "output_format": output_format,
            "task_id": task.task_id,
        }
        if task.output_path:
            kwargs["output_path"] = Path(task.output_path)
        for reserved in ("input_path", "output_path", "task_id", "output_format"):
            overrides.pop(reserved, None)
        kwargs.update(overrides)
        legacy_result = self._runner(**kwargs)
        if isinstance(legacy_result, RunResult):
            return legacy_result
        if not isinstance(legacy_result, Mapping):
            raise TypeError("Pipeline.run() must return a mapping or RunResult")
        return RunResult.from_legacy(legacy_result, request)


class EngineRegistryAdapter(EngineRegistryPort):
    """Expose governance registry records as engine availability contracts."""

    _READY_STATES = {"ready_shadow", "ready_review", "ready_default"}

    def __init__(self, registry: Any):
        self.registry = registry

    @staticmethod
    def _convert(status: Any) -> EngineAvailability:
        status_value = getattr(status.status, "value", status.status)
        identity = EngineIdentity(
            name=str(status.engine),
            model=str(getattr(status, "model", "")),
            version=str(getattr(status, "model_hash", "")),
        )
        return EngineAvailability(
            identity=identity,
            available=status_value in EngineRegistryAdapter._READY_STATES,
            status=str(status_value),
            reason=str(getattr(status, "degradation_strategy", "") or ""),
            device=str(getattr(status, "device", "") or ""),
        )

    def get(self, engine: str) -> Optional[EngineAvailability]:
        status = self.registry.get(engine)
        return self._convert(status) if status is not None else None

    def list_all(self) -> Iterable[EngineAvailability]:
        return [self._convert(status) for status in self.registry.list_all()]


class ASREngineAdapter(EnginePort):
    """Wrap an existing ASR engine without changing its transcription API."""

    def __init__(self, engine: Any, *, status: str = "ready_shadow", device: str = ""):
        self.engine = engine
        self._status = status
        self._device = device

    def identity(self) -> EngineIdentity:
        return EngineIdentity(
            name=str(getattr(self.engine, "name", self.engine.__class__.__name__)),
            model=str(getattr(self.engine, "model_name", "")),
            category="asr",
        )

    def availability(self) -> EngineAvailability:
        return EngineAvailability(
            identity=self.identity(),
            available=self._status in EngineRegistryAdapter._READY_STATES,
            status=self._status,
            device=self._device,
        )

    def prepare(self, request: Any = None) -> PrepareResult:
        try:
            self.engine.load_model()
            return PrepareResult(ready=True, identity=self.identity())
        except Exception as exc:
            return PrepareResult(
                ready=False,
                identity=self.identity(),
                error=_engine_error(exc, engine=self.identity().name, stage="prepare"),
            )

    def execute(self, request: EngineRequest) -> EngineResult:
        payload = request.payload
        if isinstance(payload, Mapping):
            audio = payload.get("audio")
            sample_rate = int(payload.get("sample_rate", 16000))
            options = dict(payload.get("options") or {})
        else:
            audio = payload
            sample_rate = 16000
            options = {}
        options.update(dict(request.options or {}))
        if request.language is not None:
            options["language"] = request.language
        try:
            output = self.engine.transcribe(audio, sample_rate=sample_rate, **options)
            return EngineResult(
                success=True,
                output=output,
                events=tuple(output or ()),
                identity=self.identity(),
            )
        except Exception as exc:
            return EngineResult(
                success=False,
                identity=self.identity(),
                error=_engine_error(exc, engine=self.identity().name, stage="execute"),
            )

    def release(self) -> None:
        unload = getattr(self.engine, "unload_model", None)
        if callable(unload):
            unload()


def _engine_error(error: Exception, *, engine: str, stage: str) -> ErrorInfo:
    raw_category = str(getattr(error, "category", "engine_failed"))
    category_map = {
        "dependency_unavailable": "dependency_missing",
        "execution_failed": "engine_failed",
        "invalid_result": "engine_failed",
        "model_missing": "model_unavailable",
    }
    category = category_map.get(raw_category, raw_category)
    recoverable = category in {"dependency_missing", "model_unavailable"}
    return ErrorInfo.from_exception(
        error,
        category=category,
        engine=engine,
        stage=stage,
        recoverable=recoverable,
        retryable=category in {"timeout", "resource_exhausted"},
    )


class RunReportAdapter(ReportPort):
    """Adapt ``RunReportBuilder`` while retaining its schema and persistence."""

    def __init__(
        self,
        builder_or_factory: Any,
        *,
        reports_root: Path | str | None = None,
    ):
        self._builder_or_factory = builder_or_factory
        self._reports_root = Path(reports_root) if reports_root else None
        self._builders: dict[str, Any] = {}

    def _builder(self, result: RunResult) -> Any:
        if not isinstance(self._builder_or_factory, type) and hasattr(
            self._builder_or_factory, "build"
        ):
            builder = self._builder_or_factory
        else:
            builder = self._builder_or_factory(
                run_id=result.task.run_id or "",
                task_id=result.task.task_id,
            )
        self._builders[result.task.run_id or ""] = builder
        return builder

    def build(self, result: RunResult, *, config_snapshot: Optional[Mapping[str, Any]] = None) -> RunReport:
        if isinstance(result.report, RunReport):
            return result.report
        builder = self._builder(result)
        legacy_report = builder.build(config_snapshot=dict(config_snapshot or {}))
        return RunReport.from_legacy(legacy_report)

    def persist(self, report: RunReport, *, config: Any = None) -> Path:
        builder = self._builders.get(report.run_id)
        if builder is not None and report._legacy_report is not None:
            return builder.persist(report._legacy_report, config=config)
        root = self._reports_root or Path("cache") / "reports"
        report_dir = root / report.run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "run_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report_path


class ArtifactRegistryAdapter(ArtifactPort):
    """Small in-memory artifact registry for coordinators and tests."""

    def __init__(self):
        self._artifacts: dict[str, Path] = {}

    def register(self, name: str, path: Path | str) -> None:
        self._artifacts[name] = Path(path)

    def exists(self, name: str) -> bool:
        path = self.get(name)
        return path is not None and path.exists()

    def get(self, name: str) -> Optional[Path]:
        return self._artifacts.get(name)
