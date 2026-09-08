"""Single decision point for converting evidence into subtitle events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .evidence import CandidateEvidence, EvidenceDecision, EvidenceWord
from .risk_scoring import RiskAssessment, normalize_text


@dataclass(frozen=True)
class DecisionConfig:
    unresolved_keeps_candidate: bool = True
    physical_tolerance: float = 0.30
    replace_min_similarity: float = 0.35
    require_multi_source_drop: bool = True


class EvidenceDecisionEngine:
    """Make conservative, traceable decisions from candidate evidence."""

    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()

    def decide_bundle(
        self,
        segmented: Sequence[CandidateEvidence],
        *,
        review: Sequence[CandidateEvidence] = (),
        assessments: Sequence[RiskAssessment] = (),
        physical_timeline: Any = None,
        secondary_bundles: Sequence[Any] = (),
    ) -> list[EvidenceDecision]:
        by_id = {item.candidate_id: item for item in assessments}
        bundles_by_candidate: dict[str, list[Any]] = defaultdict(list)
        for bundle in secondary_bundles:
            if getattr(bundle, "status", "") != "ok":
                continue
            for candidate_id in getattr(bundle, "candidate_ids", ()):
                bundles_by_candidate[candidate_id].append(bundle)
        decisions = []
        for candidate in segmented:
            assessment = by_id.get(candidate.id) or RiskAssessment(
                candidate_id=candidate.id,
                score=0.0,
                level="low",
                evidence_codes=(),
                factors={},
                review_required=False,
            )
            alternatives = self._alternatives(candidate, review)
            candidate_bundles = bundles_by_candidate.get(candidate.id, [])
            split_parts = self._split_parts(candidate, alternatives)
            if split_parts:
                decisions.extend(self.split(
                    candidate,
                    split_parts,
                    assessment=assessment,
                    physical_timeline=physical_timeline,
                    evidence_codes=self._bundle_codes(candidate_bundles),
                ))
            else:
                decisions.append(self.decide(
                    candidate,
                    assessment=assessment,
                    alternatives=alternatives,
                    physical_timeline=physical_timeline,
                    secondary_bundles=candidate_bundles,
                ))
        return decisions

    def decide(
        self,
        candidate: CandidateEvidence,
        *,
        assessment: RiskAssessment,
        alternatives: Sequence[CandidateEvidence] = (),
        physical_timeline: Any = None,
        secondary_bundles: Sequence[Any] = (),
    ) -> EvidenceDecision:
        trace: list[dict[str, Any]] = [{
            "stage": "risk_scoring",
            "score": assessment.score,
            "level": assessment.level,
            "evidence_codes": list(assessment.evidence_codes),
        }]
        selected = candidate
        decision = "keep"
        codes = list(assessment.evidence_codes)
        codes.extend(self._bundle_codes(secondary_bundles))
        if self._should_drop(
            candidate,
            assessment=assessment,
            alternatives=alternatives,
            bundles=secondary_bundles,
            physical_timeline=physical_timeline,
        ):
            return self._drop_from_evidence(
                candidate,
                assessment=assessment,
                evidence_codes=tuple(dict.fromkeys((*codes, "multi_source_non_speech"))),
                physical_timeline=physical_timeline,
                sources=self._drop_sources(secondary_bundles),
            )
        if alternatives:
            selected = self._select_alternative(candidate, alternatives)
            if selected.id != candidate.id:
                decision = "replace"
                codes.append(f"{selected.source}_agreement")
                trace.append({
                    "stage": "alternative_selection",
                    "source": selected.source,
                    "selected_candidate_id": selected.id,
                    "alternative_count": len(alternatives),
                })
        elif assessment.level in {"high", "critical"}:
            decision = "unresolved"
            codes.append("review_unavailable_or_conflicting")
            trace.append({"stage": "review", "status": "unresolved"})

        physical = self._validate_physical(selected, physical_timeline)
        if not physical["valid"]:
            if selected.id != candidate.id:
                trace.append({
                    "stage": "replacement_rejected",
                    "reason": "physical_validation_failed",
                    "selected_candidate_id": selected.id,
                })
                selected = candidate
                physical = self._validate_physical(candidate, physical_timeline)
            decision = "unresolved"
            codes.append("physical_boundary_conflict")
            trace.append({"stage": "physical_validation", **physical})
        else:
            trace.append({"stage": "physical_validation", **physical})

        keep_unresolved = not (
            decision == "unresolved" and not self.config.unresolved_keeps_candidate
        )
        final_text = selected.text if decision != "drop" and keep_unresolved else ""
        final_words = selected.words if decision != "drop" and keep_unresolved else ()
        return EvidenceDecision(
            candidate_ids=tuple(dict.fromkeys([candidate.id] + [item.id for item in alternatives])),
            decision=decision,
            final_text=final_text,
            final_words=tuple(final_words),
            start=selected.start if decision != "drop" and keep_unresolved else None,
            end=selected.end if decision != "drop" and keep_unresolved else None,
            time_source=self._time_source(selected),
            confidence=selected.confidence if decision != "drop" and keep_unresolved else None,
            risk_score=assessment.score,
            risk_level=assessment.level,
            evidence_codes=tuple(dict.fromkeys(codes)),
            physical_validation=physical,
            revision_trace=tuple(trace),
        )

    def drop(
        self,
        candidate: CandidateEvidence,
        *,
        assessment: RiskAssessment,
        evidence_codes: Iterable[str],
        physical_timeline: Any = None,
        independent_sources: Sequence[str] = (),
    ) -> EvidenceDecision:
        """Explicit drop helper; callers must provide a high-risk assessment."""
        if assessment.level not in {"high", "critical"}:
            raise ValueError("drop requires a high or critical risk assessment")
        if self.config.require_multi_source_drop and len(set(independent_sources)) < 2:
            raise ValueError("drop requires at least two independent evidence sources")
        physical = self._validate_physical(candidate, physical_timeline)
        codes = tuple(dict.fromkeys((*assessment.evidence_codes, *evidence_codes)))
        return EvidenceDecision(
            candidate_ids=(candidate.id,),
            decision="drop",
            final_text="",
            final_words=(),
            start=None,
            end=None,
            time_source=self._time_source(candidate),
            confidence=None,
            risk_score=assessment.score,
            risk_level=assessment.level,
            evidence_codes=codes,
            physical_validation=physical,
            revision_trace=({"stage": "explicit_drop", "codes": list(codes)},),
        )

    @staticmethod
    def _alternatives(candidate: CandidateEvidence, review: Sequence[CandidateEvidence]) -> list[CandidateEvidence]:
        return [
            item for item in review
            if min(candidate.end, item.end) > max(candidate.start, item.start)
        ]

    def _select_alternative(
        self,
        candidate: CandidateEvidence,
        alternatives: Sequence[CandidateEvidence],
    ) -> CandidateEvidence:
        # Prefer candidates with word timing and confidence, then the longest
        # text. This is deliberately not a majority vote.
        ranked = sorted(
            alternatives,
            key=lambda item: (
                item.has_word_times,
                item.confidence is not None,
                item.confidence if item.confidence is not None else -1.0,
                len(item.text),
            ),
            reverse=True,
        )
        best = ranked[0]
        if normalize_text(best.text) != normalize_text(candidate.text):
            from difflib import SequenceMatcher

            similarity = SequenceMatcher(
                None, normalize_text(best.text), normalize_text(candidate.text)
            ).ratio()
            if similarity < self.config.replace_min_similarity:
                return candidate
        if not best.has_word_times and not candidate.has_word_times:
            return candidate
        return best

    def split(
        self,
        candidate: CandidateEvidence,
        parts: Sequence[CandidateEvidence],
        *,
        assessment: RiskAssessment,
        physical_timeline: Any = None,
        evidence_codes: Sequence[str] = (),
    ) -> list[EvidenceDecision]:
        """Return one traceable decision per accepted timed part."""
        if len(parts) < 2:
            raise ValueError("split requires at least two parts")
        result = []
        for part in parts:
            physical = self._validate_physical(part, physical_timeline)
            accepted = physical["valid"]
            result.append(EvidenceDecision(
                candidate_ids=(candidate.id, part.id),
                decision="split" if accepted else "unresolved",
                final_text=part.text if accepted or self.config.unresolved_keeps_candidate else "",
                final_words=part.words if accepted or self.config.unresolved_keeps_candidate else (),
                start=part.start if accepted or self.config.unresolved_keeps_candidate else None,
                end=part.end if accepted or self.config.unresolved_keeps_candidate else None,
                time_source=self._time_source(part),
                confidence=part.confidence if accepted or self.config.unresolved_keeps_candidate else None,
                risk_score=assessment.score,
                risk_level=assessment.level,
                evidence_codes=tuple(dict.fromkeys((*assessment.evidence_codes, *evidence_codes, "context_reasr_split"))),
                physical_validation=physical,
                revision_trace=({
                    "stage": "split",
                    "source_candidate_id": candidate.id,
                    "part_candidate_id": part.id,
                    "physical_valid": accepted,
                },),
            ))
        return result

    @staticmethod
    def _split_parts(
        candidate: CandidateEvidence,
        alternatives: Sequence[CandidateEvidence],
    ) -> list[CandidateEvidence]:
        parts = [
            item for item in alternatives
            if item.source != "segmented"
            and item.has_word_times
            and item.start >= candidate.start - 0.10
            and item.end <= candidate.end + 0.10
        ]
        parts.sort(key=lambda item: (item.start, item.end, item.id))
        accepted: list[CandidateEvidence] = []
        cursor = candidate.start - 0.10
        for part in parts:
            if part.start < cursor:
                continue
            accepted.append(part)
            cursor = part.end
        return accepted if len(accepted) >= 2 else []

    @staticmethod
    def _bundle_codes(bundles: Sequence[Any]) -> tuple[str, ...]:
        codes: list[str] = []
        for bundle in bundles:
            source = str(getattr(bundle, "source", "secondary"))
            evidence = getattr(bundle, "evidence", {}) or {}
            if source == "forced_aligner" and evidence.get("word_count", 0):
                codes.append("forced_alignment_observed")
            if source == "sed":
                label = str(evidence.get("label", evidence.get("class", ""))).casefold()
                if any(token in label for token in ("breath", "music", "noise", "non_speech", "non-speech")):
                    codes.append("sed_non_speech")
            if source == "semantic_review":
                risk = str(evidence.get("risk", evidence.get("class", evidence.get("label", "")))).casefold()
                if any(token in risk for token in ("non_speech", "non-speech", "hallucination")):
                    codes.append("semantic_non_speech")
        return tuple(dict.fromkeys(codes))

    @classmethod
    def _drop_sources(cls, bundles: Sequence[Any]) -> tuple[str, ...]:
        sources = []
        for bundle in bundles:
            codes = cls._bundle_codes((bundle,))
            if any(code in codes for code in ("sed_non_speech", "semantic_non_speech")):
                sources.append(str(getattr(bundle, "source", "secondary")))
        return tuple(dict.fromkeys(sources))

    def _should_drop(
        self,
        candidate: CandidateEvidence,
        *,
        assessment: RiskAssessment,
        alternatives: Sequence[CandidateEvidence],
        bundles: Sequence[Any],
        physical_timeline: Any,
    ) -> bool:
        if assessment.level not in {"high", "critical"}:
            return False
        sources = self._drop_sources(bundles)
        if self.config.require_multi_source_drop and len(sources) < 2:
            return False
        if not sources or any(item.text.strip() for item in alternatives):
            return False
        if physical_timeline is None:
            return False
        spans = getattr(physical_timeline, "speech_evidence_spans", ()) or ()
        if not spans:
            return False
        overlap = sum(
            max(0.0, min(candidate.end, span.end) - max(candidate.start, span.start))
            for span in spans
        )
        return overlap <= 0.0

    def _drop_from_evidence(
        self,
        candidate: CandidateEvidence,
        *,
        assessment: RiskAssessment,
        evidence_codes: Sequence[str],
        physical_timeline: Any,
        sources: Sequence[str],
    ) -> EvidenceDecision:
        physical = self._validate_physical(candidate, physical_timeline)
        physical = {
            **physical,
            "valid": True,
            "status": "blank_allowed",
            "drop_overlap_seconds": 0.0,
        }
        return EvidenceDecision(
            candidate_ids=(candidate.id,),
            decision="drop",
            final_text="",
            final_words=(),
            start=None,
            end=None,
            time_source=self._time_source(candidate),
            confidence=None,
            risk_score=assessment.score,
            risk_level=assessment.level,
            evidence_codes=tuple(dict.fromkeys(evidence_codes)),
            physical_validation=physical,
            revision_trace=({
                "stage": "secondary_drop",
                "sources": list(sources),
                "reason": "independent_non_speech_evidence",
            },),
        )

    @staticmethod
    def _time_source(candidate: CandidateEvidence) -> str:
        for word in candidate.words:
            if word.start is not None and word.end is not None:
                return word.time_source
        return "segment_boundary"

    def _validate_physical(self, candidate: CandidateEvidence, timeline: Any) -> dict[str, Any]:
        if timeline is None:
            return {
                "valid": True,
                "status": "not_available",
                "overlap_seconds": None,
                "physical_clip_id": candidate.physical_clip_id,
            }
        spans = getattr(timeline, "speech_evidence_spans", ()) or ()
        tolerance = self.config.physical_tolerance
        candidate_start = max(0.0, candidate.start - tolerance)
        candidate_end = candidate.end + tolerance
        duration = getattr(timeline, "duration", None)
        outside_timeline = duration is not None and candidate.end > float(duration) + tolerance
        overlap = 0.0
        for span in spans:
            overlap += max(0.0, min(candidate_end, span.end) - max(candidate_start, span.start))
        word_checks = self._validate_word_times(candidate, timeline, spans, tolerance)
        valid = (
            not outside_timeline
            and (overlap > 0.0 or not spans)
            and not word_checks["invalid_word_ids"]
        )
        return {
            "valid": valid,
            "status": (
                "supported"
                if valid
                else "outside_timeline"
                if outside_timeline
                else "outside_physical_evidence"
            ),
            "overlap_seconds": round(overlap, 6),
            "candidate_start": candidate.start,
            "candidate_end": candidate.end,
            "timeline_duration": duration,
            "physical_clip_id": candidate.physical_clip_id,
            "tolerance_seconds": tolerance,
            "word_validation": word_checks,
        }

    @staticmethod
    def _validate_word_times(
        candidate: CandidateEvidence,
        timeline: Any,
        spans: Sequence[Any],
        tolerance: float,
    ) -> dict[str, Any]:
        """Check timed words against candidate and physical boundaries."""
        timed_words = [
            word for word in candidate.words
            if word.start is not None and word.end is not None
        ]
        invalid_word_ids: list[str] = []
        invalid_reasons: dict[str, list[str]] = {}
        duration = getattr(timeline, "duration", None)
        clip_bounds = {
            getattr(clip, "id", ""): (clip.start, clip.end)
            for clip in (getattr(timeline, "physical_clips", ()) or ())
        }
        for word in timed_words:
            reasons: list[str] = []
            if word.start < candidate.start - tolerance or word.end > candidate.end + tolerance:
                reasons.append("outside_candidate")
            if duration is not None and (
                word.start < -tolerance or word.end > float(duration) + tolerance
            ):
                reasons.append("outside_timeline")
            if spans and not any(
                min(word.end + tolerance, span.end)
                > max(word.start - tolerance, span.start)
                for span in spans
            ):
                reasons.append("outside_physical_evidence")
            if candidate.physical_clip_id in clip_bounds:
                clip_start, clip_end = clip_bounds[candidate.physical_clip_id]
                if word.start < clip_start - tolerance or word.end > clip_end + tolerance:
                    reasons.append("outside_physical_clip")
            if reasons:
                invalid_word_ids.append(word.id)
                invalid_reasons[word.id] = reasons
        return {
            "checked_word_count": len(timed_words),
            "missing_time_word_count": len(candidate.words) - len(timed_words),
            "invalid_word_ids": invalid_word_ids,
            "invalid_reasons": invalid_reasons,
        }


def decisions_to_subtitle_events(decisions: Sequence[EvidenceDecision]) -> list[Any]:
    """Convert accepted decisions to the existing SubtitleEvent type."""
    from .base import WordTimestamp
    from ..mapping.time_mapper import SubtitleEvent

    events = []
    for index, decision in enumerate(decisions, start=1):
        if decision.decision == "drop" or not decision.final_text or decision.start is None or decision.end is None:
            continue
        words = []
        for word in decision.final_words:
            if word.start is None or word.end is None:
                continue
            words.append(WordTimestamp(
                word=word.text,
                start=word.start - decision.start,
                end=word.end - decision.start,
                confidence=word.confidence,
                speaker_id=word.speaker_id,
            ))
        source_word_ids = [word.id for word in decision.final_words]
        if not source_word_ids:
            source_word_ids = list(decision.candidate_ids)
        events.append(SubtitleEvent(
            index=index,
            start=decision.start,
            end=decision.end,
            text=decision.final_text,
            words=words,
            asr_text=decision.final_text,
            source_word_ids=source_word_ids,
            physical_start=decision.start,
            physical_end=decision.end,
            physical_region_id=decision.physical_validation.get("physical_clip_id"),
            time_source=decision.time_source,
            alignment_warning=(
                "unresolved evidence conflict" if decision.decision == "unresolved" else None
            ),
            revision_trace=list(decision.revision_trace),
        ))
    return events


__all__ = ["DecisionConfig", "EvidenceDecisionEngine", "decisions_to_subtitle_events"]
