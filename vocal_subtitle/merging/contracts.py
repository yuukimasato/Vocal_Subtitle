"""Dependency contracts for subtitle merge decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Fragment = dict[str, Any]
DecisionGroup = dict[str, Any]


@dataclass(frozen=True)
class MergeDecisionPorts:
    """External capabilities used by local and cloud merge deciders."""

    load_local_model: Callable[[], Any]
    compute_similarity: Callable[[str, str], float]
    request_cloud_decision: Callable[[Sequence[Fragment]], list[DecisionGroup]]
    fallback_rule_decisions: Callable[[Sequence[Fragment]], list[DecisionGroup]]
    semantic_boundary: Callable[[str, str], bool]
    physical_owner_compatible: Callable[[Fragment, Fragment], bool]


@dataclass(frozen=True)
class MergeRequest:
    """Input contract for the merge service."""

    fragments: Sequence[Fragment]
    audio: Any = None
    sample_rate: int = 16000


@dataclass
class MergeResult:
    """Output contract for the merge service."""

    fragments: list[Fragment] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
