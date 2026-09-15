"""治理模块 — 引擎生命周期管理、实验注册表、配置治理、发布管理。

对应文档:
  - ENGINE_LIFECYCLE.md (engine-lifecycle-v1)
  - EXPERIMENT_REGISTRY.md (experiment-registry-v1)
  - RELEASE_GOVERNANCE.md (release-governance-v1)
"""

from .engine_lifecycle import (
    EngineLifecycle,
    EngineRegistry,
    EngineStatus,
    LifecycleManager,
)
from .experiment_registry import ExperimentRecord, ExperimentRegistry
from .release import (
    AlertReport,
    KnownLimitation,
    ObservabilityMetrics,
    PreReleaseChecklist,
    ReleaseManager,
    ReleaseRecord,
    ReleaseStatus,
    RollbackStrategy,
)

__all__ = [
    # engine_lifecycle
    "EngineLifecycle",
    "EngineStatus",
    "EngineRegistry",
    "LifecycleManager",
    # experiment_registry
    "ExperimentRecord",
    "ExperimentRegistry",
    # release
    "ReleaseStatus",
    "RollbackStrategy",
    "KnownLimitation",
    "ObservabilityMetrics",
    "AlertReport",
    "PreReleaseChecklist",
    "ReleaseRecord",
    "ReleaseManager",
]
