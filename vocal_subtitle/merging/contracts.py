"""Dependency contracts for subtitle merge decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


Fragment = Dict[str, Any]
DecisionGroup = Dict[str, Any]


@dataclass(frozen=True)
class MergeDecisionPorts:
    """External capabilities used by local and cloud merge deciders."""

    load_local_model: Callable[[], Any]
    compute_similarity: Callable[[str, str], float]
    request_cloud_decision: Callable[[Sequence[Fragment]], List[DecisionGroup]]
    fallback_rule_decisions: Callable[[Sequence[Fragment]], List[DecisionGroup]]
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

    fragments: List[Fragment] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
