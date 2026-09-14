"""Risk-window scheduling for bounded Context Re-ASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .evidence import CandidateEvidence
from .review_engines import ReviewEnginePort, ReviewUnavailable
from .risk_scoring import RiskAssessment


@dataclass(frozen=True)
class ReviewSchedulerConfig:
    left_context: float = 0.8
    right_context: float = 0.8
    max_group_duration: float = 12.0
    max_window_duration: float = 15.0
    merge_gap: float = 0.8
    # 风险门控下限(高精度方案 Task 6):只对达到该档位或带明确冲突码的
    # 候选调度 re-ASR。None 表示沿用 assessment.review_required 原语义。
    min_level: Optional[str] = None


RISK_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
# 明确冲突窗口:即便档位低于 min_level 也必须复核(优化方案 3.3)。
CONFLICT_CODES = {"global_text_conflict", "context_reasr_conflict"}


@dataclass(frozen=True)
class ReviewWindow:
    id: str
    start: float
    end: float
    candidate_ids: tuple[str, ...]
    physical_clip_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


class ReviewScheduler:
    """Turn medium/high risk candidates into bounded review windows."""

    def __init__(self, config: Optional[ReviewSchedulerConfig] = None):
        self.config = config or ReviewSchedulerConfig()

    def schedule(
        self,
        candidates: Sequence[CandidateEvidence],
        assessments: Sequence[RiskAssessment],
        *,
        timeline: Any = None,
        policy: str = "risk_only",
    ) -> list[ReviewWindow]:
        if policy not in {"risk_only", "full_quality"}:
            raise ValueError(f"unsupported review policy: {policy}")
        assessment_by_id = {item.candidate_id: item for item in assessments}

        def passes_level_gate(assessment: RiskAssessment) -> bool:
            if policy != "full_quality" and not assessment.review_required:
                return False
            min_level = self.config.min_level
            if min_level is None:
                return True
            if RISK_LEVEL_ORDER.get(assessment.level, 0) >= RISK_LEVEL_ORDER.get(
                min_level, 0
            ):
                return True
            return bool(CONFLICT_CODES & set(assessment.evidence_codes))

        targets = [
            candidate for candidate in candidates
            if assessment_by_id.get(candidate.id) is not None
            and (
                policy == "full_quality"
                or passes_level_gate(assessment_by_id[candidate.id])
            )
        ]
        targets.sort(key=lambda item: (item.start, item.end, item.id))
        windows: list[ReviewWindow] = []
        current: list[CandidateEvidence] = []
        for candidate in targets:
            if not current or self._can_merge(current[-1], candidate, timeline=timeline):
                current.append(candidate)
                continue
            windows.append(self._make_window(current, timeline=timeline, index=len(windows)))
            current = [candidate]
        if current:
            windows.append(self._make_window(current, timeline=timeline, index=len(windows)))
        return windows

    def review(
        self,
        audio: Any,
        sample_rate: int,
        windows: Sequence[ReviewWindow],
        engine: Optional[ReviewEnginePort] = None,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[CandidateEvidence], dict[str, Any]]:
        selected_engine = engine or ReviewUnavailable("context-reasr")
        all_candidates: list[CandidateEvidence] = []
        diagnostics: dict[str, Any] = {
            "engine": getattr(selected_engine, "name", "unknown"),
            "windows": [],
        }
        for window in windows:
            candidates, item_diag = self.review_window(
                audio,
                sample_rate,
                window,
                selected_engine,
                language=language,
            )
            if item_diag["status"] == "ok":
                all_candidates.extend(candidates)
            diagnostics["windows"].append(item_diag)
        diagnostics["reviewed_window_count"] = len(windows)
        diagnostics["candidate_count"] = len(all_candidates)
        diagnostics["degraded"] = any(item.get("degraded") for item in diagnostics["windows"])
        return all_candidates, diagnostics

    def review_window(
        self,
        audio: Any,
        sample_rate: int,
        window: ReviewWindow,
        engine: Optional[ReviewEnginePort] = None,
        *,
        language: Optional[str] = None,
    ) -> tuple[list[CandidateEvidence], dict[str, Any]]:
        """Review one window so callers can add independent cache policy."""
        selected_engine = engine or ReviewUnavailable("context-reasr")
        item_diag = {
            "window_id": window.id,
            "start": window.start,
            "end": window.end,
            "candidate_ids": list(window.candidate_ids),
        }
        try:
            result = selected_engine.review(
                audio,
                sample_rate,
                window,
                language=language,
            )
            candidates = list(result or ())
            item_diag.update({"status": "ok", "candidate_count": len(candidates)})
            return candidates, item_diag
        except Exception as exc:
            item_diag.update({"status": "failed", "error": str(exc), "degraded": True})
            return [], item_diag

    def _can_merge(
        self,
        previous: CandidateEvidence,
        current: CandidateEvidence,
        *,
        timeline: Any = None,
    ) -> bool:
        if current.start - previous.end > self.config.merge_gap:
            return False
        if current.end - previous.start + self.config.left_context + self.config.right_context > self.config.max_group_duration:
            return False
        previous_clip = previous.physical_clip_id
        current_clip = current.physical_clip_id
        if previous_clip and current_clip and previous_clip != current_clip:
            return False
        if timeline is not None and self._crosses_hard_gap(previous.end, current.start, timeline):
            return False
        return True

    @staticmethod
    def _crosses_hard_gap(start: float, end: float, timeline: Any) -> bool:
        if end <= start:
            return False
        spans = sorted(
            getattr(timeline, "speech_evidence_spans", ()) or (),
            key=lambda span: (span.start, span.end),
        )
        cursor = start
        for span in spans:
            if span.end <= cursor:
                continue
            if span.start - cursor > 0.8:
                return True
            cursor = max(cursor, span.end)
            if cursor >= end:
                return False
        return end - cursor > 0.8

    def _make_window(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        timeline: Any = None,
        index: int,
    ) -> ReviewWindow:
        start = min(item.start for item in candidates) - self.config.left_context
        end = max(item.end for item in candidates) + self.config.right_context
        duration = getattr(timeline, "duration", None)
        if duration is not None:
            start = max(0.0, start)
            end = min(float(duration), end)
        if end - start > self.config.max_window_duration:
            end = start + self.config.max_window_duration
        clip_ids = {item.physical_clip_id for item in candidates if item.physical_clip_id}
        return ReviewWindow(
            id=f"review:{index:04d}:{round(start * 1000):08d}-{round(end * 1000):08d}",
            start=start,
            end=max(start + 0.01, end),
            candidate_ids=tuple(item.id for item in candidates),
            physical_clip_id=next(iter(clip_ids)) if len(clip_ids) == 1 else None,
            metadata={"left_context": self.config.left_context, "right_context": self.config.right_context},
        )


__all__ = ["ReviewScheduler", "ReviewSchedulerConfig", "ReviewWindow"]
