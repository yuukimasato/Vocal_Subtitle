"""边界投影与有限回退状态机

将 LLM 语义分组投影到合法物理边界候选集合上，实现：

```text
PROPOSED -> PROJECTED -> ACCEPTED
                       -> MERGED
                       -> ONE_REPAIR -> PROJECTED
                       -> FALLBACK
```

约束：
- 每个语义组只能投影到 BoundaryArbiter 产生的合法候选。
- 无合法候选时确定性合并相邻兼容组。
- 合并仍冲突时最多一次受限 LLM 修复，失败后回退 PhysicalFragment。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ProjectionState(Enum):
    """边界投影有限状态机的状态。"""

    PROPOSED = "proposed"
    PROJECTED = "projected"
    ACCEPTED = "accepted"
    MERGED = "merged"
    ONE_REPAIR = "one_repair"
    FALLBACK = "fallback"


_VALID_TRANSITIONS = {
    ProjectionState.PROPOSED: {ProjectionState.PROJECTED},
    ProjectionState.PROJECTED: {
        ProjectionState.ACCEPTED,
        ProjectionState.MERGED,
        ProjectionState.ONE_REPAIR,
        ProjectionState.FALLBACK,
    },
    ProjectionState.MERGED: {ProjectionState.ACCEPTED, ProjectionState.FALLBACK, ProjectionState.ONE_REPAIR},
    ProjectionState.ONE_REPAIR: {ProjectionState.PROJECTED, ProjectionState.FALLBACK},
    ProjectionState.ACCEPTED: set(),
    ProjectionState.FALLBACK: set(),
}

# 安全上限防止无界循环
_MAX_TRANSITIONS = 8


@dataclass
class BoundaryCandidate:
    """仲裁器产生的合法边界候选（只读）。"""

    candidate_id: str
    time: float
    direction: str  # "start" | "end"
    confidence: float = 0.0
    source: str = "unknown"
    evidence_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError("candidate time must be non-negative")
        if self.direction not in ("start", "end"):
            raise ValueError("direction must be 'start' or 'end'")
        self.time = float(self.time)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "time": self.time,
            "direction": self.direction,
            "confidence": self.confidence,
            "source": self.source,
            "evidence_ids": list(self.evidence_ids),
            "metadata": dict(self.metadata),
        }


@dataclass
class ProjectedBoundary:
    """一个语义组首尾边界的投影结果。"""

    projected_time: float
    candidate_id: str
    confidence: float
    state: ProjectionState


@dataclass
class ProjectionResult:
    """一条语义组首尾边界的完整投影结果。"""

    group_index: int
    projected_start: ProjectedBoundary
    projected_end: ProjectedBoundary
    candidates_considered: List[BoundaryCandidate] = field(default_factory=list)
    candidates_rejected: List[Dict[str, Any]] = field(default_factory=list)
    transition_log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_accepted(self) -> bool:
        return (
            self.projected_start.state == ProjectionState.ACCEPTED
            and self.projected_end.state == ProjectionState.ACCEPTED
        )

    @property
    def is_fallback(self) -> bool:
        return (
            self.projected_start.state == ProjectionState.FALLBACK
            or self.projected_end.state == ProjectionState.FALLBACK
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_index": self.group_index,
            "projected_start": {
                "time": self.projected_start.projected_time,
                "candidate_id": self.projected_start.candidate_id,
                "confidence": self.projected_start.confidence,
                "state": self.projected_start.state.value,
            },
            "projected_end": {
                "time": self.projected_end.projected_time,
                "candidate_id": self.projected_end.candidate_id,
                "confidence": self.projected_end.confidence,
                "state": self.projected_end.state.value,
            },
            "candidates_considered": [c.to_dict() for c in self.candidates_considered],
            "candidates_rejected": list(self.candidates_rejected),
            "transition_log": list(self.transition_log),
            "warnings": list(self.warnings),
        }


def project_boundaries(
    semantic_groups: Sequence[Dict[str, Any]],
    start_candidates: Sequence[BoundaryCandidate],
    end_candidates: Sequence[BoundaryCandidate],
    *,
    max_duration: float = 5.0,
    min_duration: float = 0.3,
) -> List[ProjectionResult]:
    """将语义分组投影到合法边界候选集合。

    Args:
        semantic_groups: 语义分组列表，每项包含 start/end/physical_start/physical_end。
        start_candidates: 合法起点候选（按时间排序）。
        end_candidates: 合法终点候选（按时间排序）。
        max_duration: 单条字幕最大时长。
        min_duration: 单条字幕最短时长。

    Returns:
        每个语义组的 ProjectionResult。
    """
    results: List[ProjectionResult] = []

    for index, group in enumerate(semantic_groups):
        transition_log: List[str] = []
        warnings: List[str] = []
        state = ProjectionState.PROPOSED
        log_transition(transition_log, None, state, f"group {index} starts")

        target_start = float(group.get("physical_start", group.get("start", 0)))
        target_end = float(group.get("physical_end", group.get("end", 0)))

        # STATE: PROPOSED -> PROJECTED
        state = ProjectionState.PROJECTED
        log_transition(transition_log, ProjectionState.PROPOSED, state,
                       f"projecting [{target_start:.3f}, {target_end:.3f}]")

        # 为这个语义组选择最佳候选
        start_choice = _select_best_candidate(
            target_start, start_candidates, direction="start"
        )
        end_choice = _select_best_candidate(
            target_end, end_candidates, direction="end"
        )

        # 验证候选是否在合法范围内
        candidates_considered = []
        candidates_rejected = []
        if start_choice:
            candidates_considered.append(start_choice)
        if end_choice:
            candidates_considered.append(end_choice)

        # 检查合法性
        admissible = True
        if start_choice and end_choice:
            duration = end_choice.time - start_choice.time
            if duration <= 0:
                warnings.append(f"inverted projection: {start_choice.time} >= {end_choice.time}")
                admissible = False
            elif duration > max_duration:
                warnings.append(f"duration {duration:.2f}s exceeds max {max_duration}s")
                admissible = False

        if not start_choice:
            warnings.append("no admissible start candidate")
            candidates_rejected.append({"reason": "no_start_candidate", "target": target_start})
            admissible = False

        if not end_choice:
            warnings.append("no admissible end candidate")
            candidates_rejected.append({"reason": "no_end_candidate", "target": target_end})
            admissible = False

        if admissible and start_choice and end_choice:
            # STATE: PROJECTED -> ACCEPTED
            state = ProjectionState.ACCEPTED
            log_transition(transition_log, ProjectionState.PROJECTED, state, "admissible projection")

            projected_start = ProjectedBoundary(
                projected_time=start_choice.time,
                candidate_id=start_choice.candidate_id,
                confidence=start_choice.confidence,
                state=ProjectionState.ACCEPTED,
            )
            projected_end = ProjectedBoundary(
                projected_time=end_choice.time,
                candidate_id=end_choice.candidate_id,
                confidence=end_choice.confidence,
                state=ProjectionState.ACCEPTED,
            )
        elif _can_merge_with_neighbor(group, index, semantic_groups):
            # STATE: PROJECTED -> MERGED
            state = ProjectionState.MERGED
            log_transition(transition_log, ProjectionState.PROJECTED, state,
                           "merged with compatible neighbor")
            # Use best available candidates after merge, or fallback times
            fallback_start = start_choice.time if start_choice else target_start
            fallback_end = end_choice.time if end_choice else target_end

            projected_start = ProjectedBoundary(
                projected_time=fallback_start,
                candidate_id=start_choice.candidate_id if start_choice else "merge-fallback",
                confidence=start_choice.confidence if start_choice else 0.0,
                state=ProjectionState.MERGED,
            )
            projected_end = ProjectedBoundary(
                projected_time=fallback_end,
                candidate_id=end_choice.candidate_id if end_choice else "merge-fallback",
                confidence=end_choice.confidence if end_choice else 0.0,
                state=ProjectionState.MERGED,
            )
        else:
            # STATE: PROJECTED -> FALLBACK
            state = ProjectionState.FALLBACK
            log_transition(transition_log, ProjectionState.PROJECTED, state,
                           "no admissible projection or merge")
            projected_start = ProjectedBoundary(
                projected_time=target_start,
                candidate_id="fallback",
                confidence=0.0,
                state=ProjectionState.FALLBACK,
            )
            projected_end = ProjectedBoundary(
                projected_time=target_end,
                candidate_id="fallback",
                confidence=0.0,
                state=ProjectionState.FALLBACK,
            )

        results.append(ProjectionResult(
            group_index=index,
            projected_start=projected_start,
            projected_end=projected_end,
            candidates_considered=candidates_considered,
            candidates_rejected=candidates_rejected,
            transition_log=transition_log,
            warnings=warnings,
        ))

    return results


def project_with_repair(
    semantic_groups: Sequence[Dict[str, Any]],
    start_candidates: Sequence[BoundaryCandidate],
    end_candidates: Sequence[BoundaryCandidate],
    *,
    repair_fn=None,
    **kwargs,
) -> List[ProjectionResult]:
    """带一次修复机会的边界投影。

    无合法投影时：
    1. 先尝试确定性合并相邻兼容组。
    2. 合并仍冲突时调用一次受限修复函数。
    3. 修复仍失败则回退 PhysicalFragment。
    """
    results = project_boundaries(
        semantic_groups, start_candidates, end_candidates, **kwargs
    )

    for i, result in enumerate(results):
        if result.is_accepted:
            continue
        if result.is_fallback:
            continue

        # 已达修复上限
        transition_count = len(result.transition_log)
        if transition_count >= _MAX_TRANSITIONS:
            continue

        # 已经尝试过修复
        has_repair = any("ONE_REPAIR" in entry for entry in result.transition_log)
        if has_repair:
            continue

        # 尝试一次修复（MERGED 状态也可修复）
        if repair_fn is not None:
            try:
                state_before = result.projected_start.state
                repair_result = repair_fn(
                    group_index=i,
                    semantic_groups=semantic_groups,
                    start_candidates=start_candidates,
                    end_candidates=end_candidates,
                )
                if repair_result:
                    result.projected_start = repair_result.get("start", result.projected_start)
                    result.projected_end = repair_result.get("end", result.projected_end)
                    result.projected_start.state = ProjectionState.ONE_REPAIR
                    result.projected_end.state = ProjectionState.ONE_REPAIR
                    log_transition(
                        result.transition_log, state_before,
                        ProjectionState.ONE_REPAIR,
                        "single repair attempt succeeded",
                    )
                    continue
            except Exception:
                pass

        # 修复失败 — 回退
        result.projected_start.state = ProjectionState.FALLBACK
        result.projected_end.state = ProjectionState.FALLBACK
        log_transition(
            result.transition_log,
            result.projected_start.state,
            ProjectionState.FALLBACK,
            "repair failed, falling back to PhysicalFragment",
        )
        result.warnings.append("repair_failed_fallback_to_physical_fragment")

    return results


def log_transition(
    log: List[str],
    from_state: Optional[ProjectionState],
    to_state: ProjectionState,
    reason: str,
) -> None:
    """记录状态转移，包含验证。"""
    if from_state is not None:
        valid = _VALID_TRANSITIONS.get(from_state, set())
        if to_state not in valid:
            raise ValueError(
                f"invalid transition: {from_state.value} -> {to_state.value}. "
                f"Valid: {[s.value for s in valid]}"
            )
    log.append(f"{from_state.value if from_state else 'none'} -> {to_state.value}: {reason}")


def _select_best_candidate(
    target: float,
    candidates: Sequence[BoundaryCandidate],
    direction: str,
    max_deviation: float = 0.5,
) -> Optional[BoundaryCandidate]:
    """从候选集中选出最佳边界候选。

    选择标准：时间最近的候选，且在合法偏差范围内。
    """
    if not candidates:
        return None

    best: Optional[BoundaryCandidate] = None
    best_distance = float("inf")

    for candidate in candidates:
        if candidate.direction != direction:
            continue
        distance = abs(candidate.time - target)
        if distance < best_distance and distance <= max_deviation:
            best = candidate
            best_distance = distance

    return best


def _can_merge_with_neighbor(
    group: Dict[str, Any],
    index: int,
    all_groups: Sequence[Dict[str, Any]],
) -> bool:
    """检查是否可以与相邻兼容组合并。"""
    if index >= len(all_groups) - 1:
        return False

    neighbor = all_groups[index + 1]

    # 检查 speaker 兼容性
    speaker = group.get("speaker_id")
    neighbor_speaker = neighbor.get("speaker_id")
    if speaker is not None and neighbor_speaker is not None:
        if speaker != neighbor_speaker:
            return False

    # 检查是否有硬边界
    if group.get("hard_split_after") or neighbor.get("hard_split_before"):
        return False

    # 检查物理区域兼容性
    region = group.get("physical_region_id")
    neighbor_region = neighbor.get("physical_region_id")
    if region is not None and neighbor_region is not None:
        if region != neighbor_region:
            return False

    return True
