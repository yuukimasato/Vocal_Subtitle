"""统一词级边界裁决策略(高精度时间轴方案 Task 4)。

单一裁决入口,落实两件事:

1. 词级时间来源优先级(方案 4.1):
   ``whisperx_alignment`` > ``faster_whisper_word`` > ``segment_boundary``。
2. 分置信度自动吸附限幅(方案 4.2):
   高置信 ≤120ms;中置信 ≤200ms;低置信不自动移动仅标记;超过 200ms 必须
   通过局部 RMS、骨架连续、相邻事件与全局文本一致性确认,并且任何吸附都
   不得跨越硬静音。

本模块只产生带 ``revision_trace`` 的 BoundaryDecision 修订,不修改识别文本;
boundary_refiner 与 AcousticValidator 提供候选或校验,不再各自覆盖同一端点。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from .boundary_arbiter import BoundaryDecision


WORD_TIME_SOURCE_PRECEDENCE = (
    "whisperx_alignment",
    "faster_whisper_word",
    "segment_boundary",
)

# 各置信度档位的最大自动吸附距离(秒)。unknown(无置信度)不等于低置信,
# 按中置信处理;低置信任何自动移动都被禁止,仅保留诊断。
TIER_SNAP_LIMITS: dict[str, float] = {
    "high": 0.12,
    "medium": 0.20,
    "low": 0.0,
    "unknown": 0.20,
}

# 超过该距离的吸附必须通过 escape hatch 全部确认后才允许。
LARGE_SNAP_LIMIT = 0.20


def time_source_rank(time_source: str) -> int:
    """Rank of a time source; lower is more authoritative."""
    try:
        return WORD_TIME_SOURCE_PRECEDENCE.index(time_source)
    except ValueError:
        return len(WORD_TIME_SOURCE_PRECEDENCE)


def resolve_word_time_source(word: Any) -> str:
    """Read the authoritative time source of one ASR word."""
    metadata = getattr(word, "metadata", None)
    value = ""
    if isinstance(metadata, Mapping):
        value = str(metadata.get("time_source", "") or "")
    return value if value in WORD_TIME_SOURCE_PRECEDENCE else "segment_boundary"


def has_authoritative_word_time(word: Any) -> bool:
    """Whether a word carries a real (non segment-boundary) timestamp.

    Words without any provenance metadata are legacy raw ASR words and are
    treated as authoritative for backward compatibility; only words explicitly
    marked ``segment_boundary`` are rejected as fabricated times.
    """
    metadata = getattr(word, "metadata", None)
    if not isinstance(metadata, Mapping) or "time_source" not in metadata:
        return True
    return resolve_word_time_source(word) != "segment_boundary"


def tier_snap_limit(tier: str) -> float:
    return TIER_SNAP_LIMITS.get(tier, TIER_SNAP_LIMITS["unknown"])


def _crosses_hard_silence(
    raw_time: float,
    candidate: float,
    hard_silences: Sequence[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    for silence in hard_silences:
        start, end = float(silence[0]), float(silence[1])
        if end <= start:
            continue
        left, right = sorted((raw_time, candidate))
        if left < start and right > end:
            return start, end
    return None


def large_snap_allowed(evidence: Optional[Mapping[str, bool]]) -> bool:
    """Escape hatch for moves beyond ``LARGE_SNAP_LIMIT`` (方案 4.2)."""
    if not evidence:
        return False
    required = (
        "rms_confirmed",
        "skeleton_continuous",
        "adjacent_clear",
        "global_text_consistent",
    )
    return all(bool(evidence.get(key)) for key in required)


def enforce_snap_policy(
    decision: BoundaryDecision,
    *,
    raw_time: float,
    tier: str,
    hard_silences: Sequence[tuple[float, float]] = (),
    large_snap_evidence: Optional[Mapping[str, bool]] = None,
) -> BoundaryDecision:
    """Apply the tier snap limits to one boundary decision.

    Returns a revised decision (never mutates the input). Every outcome is
    appended to ``revision_trace`` so downstream consumers can audit why a
    boundary was accepted, clamped or rejected.
    """
    candidate_time = decision.boundary_time
    move = abs(candidate_time - raw_time)
    limit = tier_snap_limit(tier)

    def traced(
        base: BoundaryDecision,
        applied_time: float,
        reason: str,
        extra: Optional[dict[str, Any]] = None,
        **overrides: Any,
    ) -> BoundaryDecision:
        entry = {
            "stage": "snap_policy",
            "tier": tier,
            "raw_time": round(raw_time, 6),
            "candidate_time": round(candidate_time, 6),
            "applied_time": round(applied_time, 6),
            "reason": reason,
        }
        if extra:
            entry.update(extra)
        return replace(
            base,
            revision_trace=base.revision_trace + (entry,),
            **overrides,
        )

    if move <= limit + 1e-9:
        if candidate_time != raw_time:
            return traced(decision, candidate_time, "snap_policy_pass")
        return decision

    crossed = _crosses_hard_silence(raw_time, candidate_time, hard_silences)
    if crossed is not None:
        return traced(
            replace(decision, boundary_time=raw_time, accepted=False),
            raw_time,
            "cross_hard_silence",
            extra={"hard_silence": list(crossed)},
            reason_codes=decision.reason_codes + ("cross_hard_silence",),
        )

    if move > LARGE_SNAP_LIMIT + 1e-9:
        if large_snap_allowed(large_snap_evidence):
            return traced(decision, candidate_time, "large_snap_allowed")
        return traced(
            replace(decision, boundary_time=raw_time, accepted=False),
            raw_time,
            "snap_limit_exceeded",
            extra={"tier_limit": limit},
            reason_codes=decision.reason_codes + ("snap_limit_exceeded",),
        )

    return traced(
        replace(decision, boundary_time=raw_time, accepted=False),
        raw_time,
        "snap_limit_exceeded",
        extra={"tier_limit": limit},
        reason_codes=decision.reason_codes + ("snap_limit_exceeded",),
    )


def apply_word_time_provenance(
    decision: BoundaryDecision,
    word: Any,
) -> BoundaryDecision:
    """Stamp the authoritative word time source onto a decision."""
    return replace(decision, time_source=resolve_word_time_source(word))
