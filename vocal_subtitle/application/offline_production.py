"""Componentized orchestration for the offline evidence production path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..asr.contracts import (
    EvidenceReviewRequest,
    EvidenceReviewResult,
)
from ..asr.evidence import CandidateEvidence, EvidenceDecision, EvidenceWord
from ..asr.evidence_review import EvidenceReviewRuntimePorts, EvidenceReviewService
from ..mapping.time_mapper import SubtitleEvent
from ..physical.decision_projection import DecisionEventProjector


@dataclass(frozen=True)
class OfflineProductionRequest:
    """Inputs prepared by an offline pipeline route."""

    events: Sequence[SubtitleEvent]
    audio: Any = None
    sample_rate: int = 16000
    physical_timeline: Any = None
    global_evidence: Sequence[CandidateEvidence] = ()
    recovery_engine: Any = None
    recovery_language: str | None = None
    input_hash: str = ""
    audio_hash: str = ""
    physical_timeline_version: str = "physical-timeline-v1"
    route_version: str = ""
    engine: str = ""
    model: str = ""
    pair_decision: Any = None
    secondary_engine: str = ""
    pair_route_version: str = ""


@dataclass
class OfflineProductionResult:
    """Events, decisions and diagnostics returned by the coordinator."""

    events: list[SubtitleEvent] = field(default_factory=list)
    decisions: list[EvidenceDecision] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class OfflineProductionCoordinator:
    """Coordinate review, projection and fallback without owning model loading."""

    def __init__(
        self,
        review_service: EvidenceReviewService | None = None,
        projector: DecisionEventProjector | None = None,
    ) -> None:
        self.review_service = review_service or EvidenceReviewService()
        self.projector = projector or DecisionEventProjector()

    def run(
        self,
        request: OfflineProductionRequest,
        ports: EvidenceReviewRuntimePorts,
    ) -> OfflineProductionResult:
        baseline_events = list(request.events)
        config = ports.config
        if config is None or not getattr(config, "enabled", False):
            return OfflineProductionResult(
                events=baseline_events,
                diagnostics={
                    "status": "disabled",
                    "production_path": "review_disabled",
                },
            )
        authoritative = bool(
            getattr(
                config, "authoritative_mode", not getattr(config, "shadow_mode", True)
            )
        )
        shadow = bool(getattr(config, "shadow_mode", False)) or not authoritative
        if not baseline_events:
            return OfflineProductionResult(
                events=[],
                diagnostics={
                    "status": "empty",
                    "production_path": "empty_input",
                    # An empty ASR result is a valid authoritative outcome
                    # only when the runtime can account for the projection.
                    "review_status": "ok",
                    "decision_count": 0,
                    "physical_projection": {
                        "mode": "empty_input",
                        "decision_count": 0,
                        "event_count": 0,
                        "physical_violation_count": 0,
                        "cross_silence_count": 0,
                        "decision_trace_missing_count": 0,
                        "raw_event_bypass_count": 0,
                    },
                },
            )

        review_request = EvidenceReviewRequest(
            events=baseline_events,
            audio=request.audio,
            sample_rate=request.sample_rate,
            physical_timeline=request.physical_timeline,
            global_evidence=tuple(request.global_evidence or ()),
            input_hash=request.input_hash,
            audio_hash=request.audio_hash,
            physical_timeline_version=request.physical_timeline_version,
            route_version=request.route_version,
            engine=request.engine,
            model=request.model,
            review_policy=getattr(request.pair_decision, "policy", "risk_only"),
            secondary_engine=request.secondary_engine,
            pair_route_version=request.pair_route_version,
            review_policy_version=getattr(
                config, "review_policy_version", "review-policy-v1"
            ),
            risk_policy_version=getattr(
                config, "risk_policy_version", "risk-policy-v1"
            ),
            decision_policy_version=getattr(
                config, "decision_policy_version", "decision-policy-v1"
            ),
            evidence_schema_version=getattr(
                config, "evidence_schema_version", "evidence-v1"
            ),
        )
        try:
            review_result = self.review_service.run(review_request, ports)
            events, projection_diagnostics = self._events_from_review(
                review_result,
                config,
                baseline_events,
                request.physical_timeline,
                audio=request.audio,
                sample_rate=request.sample_rate,
            )
            recovery_candidates, recovery_diagnostics = self._recover_uncovered_ranges(
                request,
                config,
                projection_diagnostics,
            )
            if recovery_candidates:
                review_result = self.review_service.run(
                    replace(
                        review_request,
                        recovery_evidence=tuple(recovery_candidates),
                    ),
                    ports,
                )
                events, projection_diagnostics = self._events_from_review(
                    review_result,
                    config,
                    baseline_events,
                    request.physical_timeline,
                    audio=request.audio,
                    sample_rate=request.sample_rate,
                )
            diagnostics = {
                **dict(review_result.diagnostics or {}),
                "status": "completed",
                "production_path": ("shadow" if shadow else "authoritative"),
                "review_status": "ok",
                "decision_count": len(review_result.decisions),
                "actions": self._decision_action_counts(review_result.decisions),
                "physical_projection": projection_diagnostics,
                "recovery": recovery_diagnostics,
                "engine_pair": (
                    request.pair_decision.to_dict()
                    if request.pair_decision is not None
                    else None
                ),
            }
            return OfflineProductionResult(
                events=events,
                decisions=list(review_result.decisions),
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if not getattr(config, "fallback_to_segmented", True):
                raise
            fallback_decisions = self._baseline_decisions(baseline_events)
            try:
                fallback_projection = self.projector.project_with_diagnostics(
                    fallback_decisions,
                    physical_timeline=request.physical_timeline,
                    audio=request.audio,
                    sample_rate=request.sample_rate,
                )
                fallback_events = fallback_projection.events
                projection_diagnostics = dict(fallback_projection.diagnostics)
                projection_diagnostics["mode"] = "segmented_fallback_projected"
            except Exception as projection_exc:
                raise RuntimeError(
                    f"review failed and fallback projection failed: {projection_exc}"
                ) from projection_exc
            return OfflineProductionResult(
                events=fallback_events,
                decisions=fallback_decisions,
                diagnostics={
                    "status": "degraded",
                    "review_status": "failed",
                    "production_path": "segmented_fallback",
                    "fallback_reason": str(exc),
                    "degraded": True,
                    "error_category": self._error_category(exc),
                    "decision_count": len(fallback_decisions),
                    "actions": self._decision_action_counts(fallback_decisions),
                    "physical_projection": projection_diagnostics,
                },
            )

    @staticmethod
    def _recovery_config(config: Any):
        from ..asr.local_recovery import LocalRecoveryConfig

        return LocalRecoveryConfig(
            max_attempts_per_range=int(
                getattr(config, "local_recovery_max_attempts", 3)
            ),
            min_confidence=float(getattr(config, "local_recovery_min_confidence", 0.5)),
            context_window=float(
                getattr(config, "local_recovery_context_seconds", 0.5)
            ),
            request_tolerance=float(
                getattr(config, "local_recovery_request_tolerance", 0.15)
            ),
        )

    def _recover_uncovered_ranges(
        self,
        request: OfflineProductionRequest,
        config: Any,
        projection_diagnostics: dict[str, Any],
    ) -> tuple[list[CandidateEvidence], dict[str, Any]]:
        """Convert bounded physical coverage gaps into reviewable evidence."""
        if not getattr(config, "local_recovery_enabled", True):
            return [], {"status": "disabled", "candidate_count": 0}
        coverage = dict(projection_diagnostics.get("coverage") or {})
        ranges = list(coverage.get("recovery_ranges") or ())
        if not ranges:
            return [], {
                "status": "not_needed" if coverage.get("complete") else "unavailable",
                "candidate_count": 0,
            }
        if request.audio is None or request.recovery_engine is None:
            return [], {
                "status": "unavailable",
                "reason": "audio_or_primary_engine_missing",
                "range_count": len(ranges),
                "candidate_count": 0,
            }

        from ..asr.local_recovery import (
            LocalRecoveryEngine,
            LocalRecoveryRequest,
        )

        try:
            recovery_requests = []
            for item in ranges:
                if float(item.get("end", 0.0)) <= float(item.get("start", 0.0)):
                    continue
                metadata = {
                    "bin_ids": list(item.get("bin_ids") or ()),
                    "physical_clip_id": item.get("physical_clip_id"),
                }
                # 时间轴仲裁层 R3:审计侧空洞 × turns 换人边界联动的诊断
                # 标记(turns=None 时审计不产出该标记,行为不变)
                if item.get("possible_speaker_hole"):
                    metadata["possible_speaker_hole"] = True
                recovery_requests.append(
                    LocalRecoveryRequest(
                        start=float(item["start"]),
                        end=float(item["end"]),
                        reasons=("uncovered",),
                        metadata=metadata,
                    )
                )
            results = LocalRecoveryEngine(
                request.recovery_engine,
                config=self._recovery_config(config),
                language=request.recovery_language,
            ).process_requests(
                recovery_requests,
                request.audio,
                sample_rate=request.sample_rate,
            )
        except Exception as exc:
            return [], {
                "status": "failed",
                "reason": str(exc),
                "range_count": len(ranges),
                "candidate_count": 0,
            }

        candidates: list[CandidateEvidence] = []
        outcomes: list[dict[str, Any]] = []
        for index, result in enumerate(results, start=1):
            request_metadata = dict(result.request.metadata or {})
            outcomes.append(
                {
                    "start": result.request.start,
                    "end": result.request.end,
                    "outcome": result.outcome,
                    "attempt_count": result.attempt_count,
                    "candidate_count": len(result.candidates),
                }
            )
            if not result.success:
                continue
            words = tuple(
                EvidenceWord(
                    id=item.word_id,
                    text=item.text,
                    start=item.start,
                    end=item.end,
                    confidence=item.confidence,
                    diagnostics={
                        "source": "local_recovery",
                        "recovery_range": {
                            "start": result.request.start,
                            "end": result.request.end,
                        },
                    },
                )
                for item in result.candidates
            )
            if not words:
                continue
            candidates.append(
                CandidateEvidence(
                    id=f"local_recovery:{index:04d}",
                    source="local_recovery",
                    text=" ".join(item.text for item in words),
                    start=min(float(item.start) for item in words),
                    end=max(float(item.end) for item in words),
                    engine=getattr(request.recovery_engine, "name", None),
                    words=words,
                    confidence=sum(float(item.confidence or 0.0) for item in words)
                    / len(words),
                    physical_clip_id=request_metadata.get("physical_clip_id"),
                    language=request.recovery_language,
                    diagnostics={
                        "evidence_role": "local_recovery",
                        "recovery_range": {
                            "start": result.request.start,
                            "end": result.request.end,
                            "bin_ids": request_metadata.get("bin_ids", []),
                        },
                    },
                )
            )
        return candidates, {
            "status": "recovered" if candidates else "incomplete",
            "range_count": len(ranges),
            "candidate_count": len(candidates),
            "outcomes": outcomes,
        }

    @staticmethod
    def _decision_action_counts(decisions) -> dict[str, int]:
        """Count decision actions for run report."""
        from ..asr.evidence import DECISIONS

        counts: dict[str, int] = {a: 0 for a in DECISIONS}
        for d in decisions:
            action = getattr(d, "decision", "keep")
            counts[action] = counts.get(action, 0) + 1
        return counts

    @staticmethod
    def _error_category(exc: Exception) -> str:
        message = str(exc).casefold()
        if "model" in message and ("not found" in message or "unavailable" in message):
            return "model_unavailable"
        if "not installed" in message or "dependency" in message:
            return "dependency_unavailable"
        if "timeout" in message:
            return "engine_timeout"
        return "engine_failed"

    @staticmethod
    def _baseline_decisions(events: Sequence[SubtitleEvent]) -> list[EvidenceDecision]:
        """Promote primary events into explicit keep decisions for fallback."""
        decisions: list[EvidenceDecision] = []
        for event in events:
            candidate_id = f"segmented:event:{int(event.index):06d}"
            words: list[EvidenceWord] = []
            source_ids = list(getattr(event, "source_word_ids", ()) or ())
            for word_index, word in enumerate(getattr(event, "words", ()) or ()):
                if isinstance(word, dict):
                    text = word.get("word", word.get("text", ""))
                    start = word.get("start")
                    end = word.get("end")
                    confidence = word.get("confidence")
                    speaker_id = word.get("speaker_id")
                else:
                    text = getattr(word, "word", getattr(word, "text", ""))
                    start = getattr(word, "start", None)
                    end = getattr(word, "end", None)
                    confidence = getattr(word, "confidence", None)
                    speaker_id = getattr(word, "speaker_id", None)
                if (
                    not text
                    or start is None
                    or end is None
                    or float(end) <= float(start)
                ):
                    continue
                word_id = (
                    source_ids[word_index]
                    if word_index < len(source_ids)
                    else (f"{candidate_id}:word:{word_index:04d}")
                )
                words.append(
                    EvidenceWord(
                        id=word_id,
                        text=str(text),
                        start=float(event.start) + float(start),
                        end=float(event.start) + float(end),
                        confidence=confidence,
                        speaker_id=speaker_id,
                    )
                )
            decisions.append(
                EvidenceDecision(
                    candidate_ids=(candidate_id,),
                    decision="keep",
                    final_text=str(event.text or "").strip(),
                    final_words=tuple(words),
                    start=float(event.start),
                    end=float(event.end),
                    time_source=(
                        getattr(event, "time_source", "")
                        if getattr(event, "time_source", "")
                        in {
                            "native_word_timestamp",
                            "segment_boundary",
                            "physical_acoustic_boundary",
                        }
                        else "segment_boundary"
                    ),
                    confidence=0.0,
                    risk_score=1.0,
                    risk_level="critical",
                    evidence_codes=("primary_candidate_fallback",),
                    physical_validation={"valid": True, "status": "review_fallback"},
                    revision_trace=(
                        {
                            "stage": "segmented_fallback_decision",
                            "candidate_id": candidate_id,
                        },
                    ),
                )
            )
        return decisions

    def _events_from_review(
        self,
        review_result: EvidenceReviewResult,
        config: Any,
        baseline_events: Sequence[SubtitleEvent],
        physical_timeline: Any = None,
        *,
        audio: Any = None,
        sample_rate: int = 16000,
    ) -> tuple[list[SubtitleEvent], dict[str, Any]]:
        authoritative = bool(
            getattr(
                config, "authoritative_mode", not getattr(config, "shadow_mode", True)
            )
        )
        if getattr(config, "shadow_mode", False) or not authoritative:
            # Shadow is diagnostic-only: project the decisions to validate
            # physical invariants, but keep the segmented baseline as output.
            projection = self.projector.project_with_diagnostics(
                review_result.decisions,
                physical_timeline=physical_timeline,
                audio=audio,
                sample_rate=sample_rate,
            )
            return list(baseline_events), {
                **projection.diagnostics,
                "mode": "shadow_observed",
                "returned_event_count": len(baseline_events),
                "projected_event_count": len(projection.events),
            }
        projection = self.projector.project_with_diagnostics(
            review_result.decisions,
            physical_timeline=physical_timeline,
            audio=audio,
            sample_rate=sample_rate,
        )
        return projection.events, projection.diagnostics


__all__ = [
    "OfflineProductionCoordinator",
    "OfflineProductionRequest",
    "OfflineProductionResult",
]
