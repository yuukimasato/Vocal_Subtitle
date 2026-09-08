"""Project evidence decisions through the physical subtitle pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..asr.base import WordTimestamp
from ..asr.evidence import EvidenceDecision
from ..mapping.time_mapper import SubtitleEvent
from .allocator import allocate_words, repair_late_words
from .coverage import audit_physical_coverage
from .decision_ir import decisions_to_global_transcript
from .events import build_events
from .subtitle_bins import build_physical_subtitle_bins


class DecisionProjectionError(ValueError):
    """Raised when a decision cannot safely become a subtitle event."""


@dataclass
class DecisionProjectionResult:
    events: list[SubtitleEvent] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DecisionEventProjector:
    """Build final events from decisions, with one physical projection path."""

    def project(
        self,
        decisions: Sequence[EvidenceDecision],
        physical_timeline: Any = None,
        *,
        audio: Any = None,
        sample_rate: int = 16000,
    ) -> list[SubtitleEvent]:
        return self.project_with_diagnostics(
            decisions,
            physical_timeline=physical_timeline,
            audio=audio,
            sample_rate=sample_rate,
        ).events

    def project_with_diagnostics(
        self,
        decisions: Sequence[EvidenceDecision],
        physical_timeline: Any = None,
        *,
        audio: Any = None,
        sample_rate: int = 16000,
    ) -> DecisionProjectionResult:
        if physical_timeline is None:
            events = []
            for decision in decisions:
                event = self._project_one(decision, len(events) + 1)
                if event is not None:
                    events.append(event)
            trace_missing = sum(
                1
                for event in events
                if not any(
                    item.get("stage") == "decision_event_projection"
                    for item in event.revision_trace
                )
            )
            return DecisionProjectionResult(
                events=events,
                diagnostics={
                    "mode": "direct",
                    "decision_count": len(decisions),
                    "event_count": len(events),
                    "physical_violation_count": sum(
                        1
                        for decision in decisions
                        if (decision.physical_validation or {}).get("valid") is False
                    ),
                    "cross_silence_count": 0,
                    "decision_trace_missing_count": trace_missing,
                    "raw_event_bypass_count": trace_missing,
                },
            )

        return self._project_physical(
            decisions,
            physical_timeline,
            audio=audio,
            sample_rate=sample_rate,
        )

    def _project_physical(
        self,
        decisions: Sequence[EvidenceDecision],
        physical_timeline: Any,
        *,
        audio: Any,
        sample_rate: int,
    ) -> DecisionProjectionResult:
        from .timeline import PhysicalTimeline

        if not isinstance(physical_timeline, PhysicalTimeline):
            raise DecisionProjectionError("physical_timeline must be a PhysicalTimeline")
        if sample_rate <= 0:
            raise DecisionProjectionError("sample_rate must be positive")

        transcript = decisions_to_global_transcript(
            decisions,
            audio_duration=physical_timeline.duration,
        )
        bins = build_physical_subtitle_bins(
            physical_timeline,
            audio=audio,
            sample_rate=sample_rate,
        )
        repaired = repair_late_words(transcript, bins) if bins else transcript
        allocation = allocate_words(
            repaired,
            physical_timeline,
            subtitle_bins=bins or None,
        )
        coverage = (
            audit_physical_coverage(bins, allocation.allocations)
            if bins
            else None
        )
        physical_events = build_events(
            allocation,
            subtitle_bins=bins or None,
        )
        events = [item.to_subtitle_event() for item in physical_events]
        trace_missing = sum(
            1
            for event in physical_events
            if not any(
                item.get("stage") == "evidence_decision"
                and item.get("decision_id")
                for item in event.revision_trace
            )
        )
        cross_silence_count = 0
        for event in physical_events:
            spans = sorted(
                (
                    span.start,
                    span.end,
                )
                for span in event.physical_spans
            )
            if any(start - previous_end > 0.35 for (_, previous_end), (start, _) in zip(spans, spans[1:])):
                cross_silence_count += 1
        decision_by_id = {
            f"decision:{index:06d}": decision
            for index, decision in enumerate(decisions, start=1)
        }
        for event in events:
            evidence_by_id = {
                span.id: span.source
                for span in physical_timeline.speech_evidence_spans
            }
            for span in event.physical_spans:
                sources = [
                    evidence_by_id[item]
                    for item in span.get("evidence_ids", ())
                    if item in evidence_by_id
                ]
                if sources:
                    span["source"] = "+".join(dict.fromkeys(sources))
            decision_ids = [
                str(item.get("decision_id"))
                for item in event.revision_trace
                if item.get("stage") == "evidence_decision"
                and item.get("decision_id")
            ]
            source_decisions = [
                decision_by_id[item]
                for item in decision_ids
                if item in decision_by_id
            ]
            if not source_decisions:
                continue
            overlaps = [
                span for span in event.physical_spans
                if span.get("physical_clip_id")
            ]
            if overlaps:
                clip_ids = {
                    str(span["physical_clip_id"]) for span in overlaps
                }
                if len(clip_ids) == 1:
                    event.physical_region_id = next(iter(clip_ids))
                timeline_overlaps = [
                    span for decision in source_decisions
                    for span in physical_timeline.speech_evidence_spans
                    if min(float(decision.end or event.end), span.end)
                    > max(float(decision.start or event.start), span.start)
                ]
                evidence_start = min(
                    (span.start for span in timeline_overlaps),
                    default=min(float(span["start"]) for span in overlaps),
                )
                evidence_end = max(
                    (span.end for span in timeline_overlaps),
                    default=max(float(span["end"]) for span in overlaps),
                )
                event.physical_start = max(
                    float(source_decisions[0].start or event.start),
                    evidence_start,
                )
                event.physical_end = min(
                    float(source_decisions[-1].end or event.end),
                    evidence_end,
                )
                if event.physical_bin_start is not None:
                    event.physical_start = max(
                        event.physical_start,
                        float(event.physical_bin_start),
                    )
                if event.physical_bin_end is not None:
                    event.physical_end = min(
                        event.physical_end,
                        float(event.physical_bin_end),
                    )
        diagnostics: dict[str, Any] = {
            "mode": "physical",
            "decision_count": len(decisions),
            "event_count": len(events),
            "bin_count": len(bins),
            "bins": [item.to_dict() for item in bins],
            "transcript": dict(transcript.diagnostics),
            "allocation": dict(allocation.diagnostics),
            "rejected_word_ids": [item.word.id for item in allocation.rejected],
            "physical_violation_count": sum(
                1
                for decision in decisions
                if (decision.physical_validation or {}).get("valid") is False
            ),
            "cross_silence_count": cross_silence_count,
            "decision_trace_missing_count": trace_missing,
            "raw_event_bypass_count": trace_missing,
            "coverage": coverage.to_dict() if coverage is not None else {
                "complete": False,
                "status": "no_physical_speech_bins",
            },
        }
        return DecisionProjectionResult(events=events, diagnostics=diagnostics)

    def _project_one(
        self,
        decision: EvidenceDecision,
        index: int,
        *,
        physical_timeline: Any = None,
    ) -> SubtitleEvent | None:
        if decision.decision == "drop":
            return None
        if not decision.final_text or decision.start is None or decision.end is None:
            return None
        if decision.end <= decision.start:
            raise DecisionProjectionError(
                f"decision {decision.candidate_ids} has invalid range"
            )

        words = [
            WordTimestamp(
                word=word.text,
                start=word.start - decision.start,
                end=word.end - decision.start,
                confidence=word.confidence,
                speaker_id=word.speaker_id,
            )
            for word in decision.final_words
            if word.start is not None
            and word.end is not None
            and word.end > word.start
        ]
        source_word_ids = [word.id for word in decision.final_words]
        if not source_word_ids:
            source_word_ids = list(decision.candidate_ids)

        speaker_ids = {
            word.speaker_id for word in decision.final_words
            if word.speaker_id is not None
        }
        physical = decision.physical_validation or {}
        physical_region_id = physical.get("physical_clip_id")
        physical_start = decision.start
        physical_end = decision.end
        physical_spans: list[dict[str, Any]] = []
        warnings = []
        if decision.decision == "unresolved":
            warnings.append("unresolved evidence conflict")
        if not words and decision.final_words:
            warnings.append("missing word timestamps")
        if physical.get("valid") is False:
            warnings.append(str(physical.get("status") or "physical validation failed"))
        if physical_timeline is not None:
            overlaps = [
                span
                for span in (getattr(physical_timeline, "speech_evidence_spans", ()) or ())
                if min(decision.end, span.end) > max(decision.start, span.start)
            ]
            if not overlaps:
                warnings.append("no physical evidence overlap")
            else:
                physical_start = max(
                    decision.start, min(span.start for span in overlaps)
                )
                physical_end = min(
                    decision.end, max(span.end for span in overlaps)
                )
                clip_ids = {
                    span.physical_clip_id
                    for span in overlaps
                    if span.physical_clip_id
                }
                if physical_region_id is None and len(clip_ids) == 1:
                    physical_region_id = next(iter(clip_ids))
                physical_spans = [
                    {
                        "id": span.id,
                        "start": max(decision.start, span.start),
                        "end": min(decision.end, span.end),
                        "source": span.source,
                        "physical_clip_id": span.physical_clip_id,
                    }
                    for span in overlaps
                ]

        revision_trace = [
            *decision.revision_trace,
            {
                "stage": "decision_event_projection",
                "decision": decision.decision,
                "decision_id": decision.decision_id,
                "candidate_ids": list(decision.candidate_ids),
                "final_event_id": f"final:event:{index:06d}",
            },
        ]
        return SubtitleEvent(
            index=index,
            start=decision.start,
            end=decision.end,
            text=decision.final_text,
            words=words,
            asr_text=decision.final_text,
            speaker_id=next(iter(speaker_ids)) if len(speaker_ids) == 1 else None,
            source_word_ids=source_word_ids,
            physical_start=physical_start,
            physical_end=physical_end,
            physical_spans=physical_spans,
            physical_region_id=physical_region_id,
            time_source=decision.time_source,
            alignment_warning=";".join(warnings) or None,
            revision_trace=revision_trace,
            trace_context={
                **dict(getattr(decision, "trace_context", {}) or {}),
                "decision_id": decision.decision_id,
                "final_event_ids": [f"final:event:{index:06d}"],
                "physical_span_ids": [
                    item.get("id") for item in physical_spans
                    if isinstance(item, dict) and item.get("id")
                ],
            },
        )


__all__ = [
    "DecisionEventProjector",
    "DecisionProjectionError",
    "DecisionProjectionResult",
]
