"""Stable backend component contracts and ports."""

from .adapters import (
    ArtifactRegistryAdapter,
    ASREngineAdapter,
    EngineRegistryAdapter,
    PipelineRunAdapter,
    RunReportAdapter,
    TaskHistoryAdapter,
)
from .common import (
    CONTRACT_VERSION,
    DECISION_POLICY_VERSION,
    REPORT_SCHEMA_VERSION,
    ROUTE_VERSION,
    ErrorInfo,
)
from .engine import (
    EngineAvailability,
    EngineIdentity,
    EngineRequest,
    EngineResult,
    PrepareResult,
)
from .external import CLIAdapter, ExternalAdapter, WebUIAdapter
from .ports import (
    ArtifactPort,
    DecisionProjectionPort,
    EnginePort,
    EngineRegistryPort,
    EvidenceReviewPort,
    PipelineRunPort,
    ReportPort,
    TaskPort,
)
from .report import RunReport
from .run import RunRequest, RunResult
from .task import TaskRequest, TaskSnapshot, TaskState

__all__ = [
    "CONTRACT_VERSION",
    "ROUTE_VERSION",
    "DECISION_POLICY_VERSION",
    "REPORT_SCHEMA_VERSION",
    "ErrorInfo",
    "TaskState",
    "TaskRequest",
    "TaskSnapshot",
    "RunRequest",
    "RunResult",
    "RunReport",
    "EngineIdentity",
    "EngineAvailability",
    "EngineRequest",
    "EngineResult",
    "PrepareResult",
    "TaskPort",
    "PipelineRunPort",
    "EnginePort",
    "EngineRegistryPort",
    "ReportPort",
    "ArtifactPort",
    "EvidenceReviewPort",
    "DecisionProjectionPort",
    "TaskHistoryAdapter",
    "PipelineRunAdapter",
    "EngineRegistryAdapter",
    "RunReportAdapter",
    "ArtifactRegistryAdapter",
    "ASREngineAdapter",
    "ExternalAdapter",
    "CLIAdapter",
    "WebUIAdapter",
]
