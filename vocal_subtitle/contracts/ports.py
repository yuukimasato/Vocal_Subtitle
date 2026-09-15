"""Ports consumed by backend application orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .engine import (
    EngineAvailability,
    EngineIdentity,
    EngineRequest,
    EngineResult,
    PrepareResult,
)
from .report import RunReport
from .run import RunRequest, RunResult
from .task import TaskRequest, TaskSnapshot, TaskState


@runtime_checkable
class TaskPort(Protocol):
    def get(self, task_id: str) -> TaskSnapshot | None: ...

    def create(
        self,
        request: TaskRequest,
        *,
        input_fingerprint: str = "",
        config: Any = None,
    ) -> TaskSnapshot: ...

    def transition(
        self,
        task_id: str,
        state: TaskState,
        *,
        run_id: str = "",
        error: Any = None,
    ) -> TaskSnapshot: ...


@runtime_checkable
class PipelineRunPort(Protocol):
    def run(self, request: RunRequest) -> RunResult: ...


@runtime_checkable
class EngineRegistryPort(Protocol):
    def get(self, engine: str) -> EngineAvailability | None: ...

    def list_all(self) -> Iterable[EngineAvailability]: ...


@runtime_checkable
class EnginePort(Protocol):
    def identity(self) -> EngineIdentity: ...

    def availability(self) -> EngineAvailability: ...

    def prepare(self, request: EngineRequest) -> PrepareResult: ...

    def execute(self, request: EngineRequest) -> EngineResult: ...

    def release(self) -> None: ...


@runtime_checkable
class ReportPort(Protocol):
    def build(
        self, result: RunResult, *, config_snapshot: Mapping[str, Any] | None = None
    ) -> RunReport: ...

    def persist(self, report: RunReport, *, config: Any = None) -> Path: ...


@runtime_checkable
class ArtifactPort(Protocol):
    def register(self, name: str, path: Path | str) -> None: ...

    def exists(self, name: str) -> bool: ...

    def get(self, name: str) -> Path | None: ...


@runtime_checkable
class EvidenceReviewPort(Protocol):
    def review(self, request: Any) -> Any: ...


@runtime_checkable
class DecisionProjectionPort(Protocol):
    def project(self, request: Any) -> Any: ...
