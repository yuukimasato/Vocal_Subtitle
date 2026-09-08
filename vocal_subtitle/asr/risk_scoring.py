"""Explainable risk scoring for candidate evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from .evidence import CandidateEvidence


_REPEATED_SPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\u3400-\u9fff\u3040-\u30ff]+", re.UNICODE)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = _PUNCTUATION.sub(" ", value)
    return _REPEATED_SPACE.sub(" ", value).strip()


def _similarity(left: str, right: str) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class RiskAssessment:
    candidate_id: str
    score: float
    level: str
    evidence_codes: tuple[str, ...] = ()
    factors: dict[str, float] = field(default_factory=dict)
    review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "level": self.level,
            "evidence_codes": list(self.evidence_codes),
            "factors": dict(self.factors),
            "review_required": self.review_required,
        }


@dataclass(frozen=True)
class RiskScoringConfig:
    short_duration_seconds: float = 0.8
    high_cps: float = 15.0
    critical_cps: float = 20.0
    medium_threshold: float = 0.25
    high_threshold: float = 0.50
    critical_threshold: float = 0.75
    repeated_phrase_gap_seconds: float = 3.0


class EvidenceRiskScorer:
    """Score candidates without deciding whether they should be deleted."""

    def __init__(self, config: Optional[RiskScoringConfig] = None):
        self.config = config or RiskScoringConfig()

    def score_bundle(
        self,
        candidates: Iterable[CandidateEvidence],
        *,
        global_evidence: Iterable[CandidateEvidence] = (),
        physical_timeline: Any = None,
    ) -> list[RiskAssessment]:
        all_candidates = tuple(candidates)
        all_global = tuple(global_evidence)
        return [
            self.score(
                candidate,
                global_evidence=all_global,
                physical_timeline=physical_timeline,
                repeated_candidates=all_candidates,
            )
            for candidate in all_candidates
        ]

    def score(
        self,
        candidate: CandidateEvidence,
        *,
        global_evidence: Iterable[CandidateEvidence] = (),
        physical_timeline: Any = None,
        repeated_candidates: Iterable[CandidateEvidence] = (),
    ) -> RiskAssessment:
        factors: dict[str, float] = {}
        codes: list[str] = []
        score = 0.0

        duration = max(candidate.duration, 0.001)
        meaningful_chars = len(normalize_text(candidate.text).replace(" ", ""))
        cps = meaningful_chars / duration
        if duration < self.config.short_duration_seconds:
            score += 0.12
            codes.append("short_duration")
            factors["short_duration"] = 0.12
        if cps >= self.config.critical_cps and meaningful_chars >= 8:
            score += 0.35
            codes.append("critical_cps")
            factors["cps"] = min(1.0, cps / self.config.critical_cps)
        elif cps >= self.config.high_cps:
            score += 0.20
            codes.append("high_cps")
            factors["cps"] = min(1.0, cps / self.config.high_cps)

        if candidate.confidence is None:
            score += 0.15
            codes.append("missing_word_confidence")
            factors["missing_confidence"] = 0.15
        elif candidate.confidence < 0.45:
            score += 0.20
            codes.append("low_confidence")
            factors["confidence"] = 1.0 - candidate.confidence

        if not candidate.has_word_times:
            score += 0.15
            codes.append("missing_word_time")
            factors["missing_word_time"] = 0.15

        if candidate.no_speech_prob is not None and candidate.no_speech_prob >= 0.6:
            score += 0.22
            codes.append("high_no_speech_prob")
            factors["no_speech_prob"] = candidate.no_speech_prob
        if candidate.avg_logprob is not None and candidate.avg_logprob < -1.0:
            score += 0.12
            codes.append("low_avg_logprob")
            factors["avg_logprob"] = min(1.0, abs(candidate.avg_logprob) / 3.0)
        if candidate.compression_ratio is not None and candidate.compression_ratio >= 2.4:
            score += 0.15
            codes.append("high_compression_ratio")
            factors["compression_ratio"] = min(1.0, candidate.compression_ratio / 4.0)

        overlap_conflict = self._global_conflict(candidate, global_evidence)
        if overlap_conflict:
            score += 0.25
            codes.append("global_text_conflict")
            factors["global_conflict"] = overlap_conflict

        if physical_timeline is not None and not self._has_physical_support(candidate, physical_timeline):
            score += 0.20
            codes.append("physical_coverage_gap")
            factors["physical_coverage"] = 0.20

        repeat_gap = self._repeated_phrase_gap(candidate, repeated_candidates)
        if repeat_gap is not None:
            score += 0.18
            codes.append("repeated_phrase")
            factors["repeated_phrase_gap"] = repeat_gap

        score = min(1.0, score)
        if score >= self.config.critical_threshold:
            level = "critical"
        elif score >= self.config.high_threshold:
            level = "high"
        elif score >= self.config.medium_threshold:
            level = "medium"
        else:
            level = "low"
        return RiskAssessment(
            candidate_id=candidate.id,
            score=round(score, 6),
            level=level,
            evidence_codes=tuple(dict.fromkeys(codes)),
            factors=factors,
            review_required=level in {"medium", "high", "critical"},
        )

    @staticmethod
    def _global_conflict(candidate: CandidateEvidence, global_evidence: Iterable[CandidateEvidence]) -> float:
        best = 0.0
        for observation in global_evidence:
            if observation.source != "global":
                continue
            overlap = min(candidate.end, observation.end) - max(candidate.start, observation.start)
            if overlap <= 0:
                continue
            similarity = _similarity(candidate.text, observation.text)
            conflict = 1.0 - similarity
            best = max(best, conflict)
        return best if best >= 0.45 else 0.0

    @staticmethod
    def _has_physical_support(candidate: CandidateEvidence, timeline: Any) -> bool:
        spans = getattr(timeline, "speech_evidence_spans", ()) or ()
        for span in spans:
            if min(candidate.end, span.end) > max(candidate.start, span.start):
                return True
        return False

    def _repeated_phrase_gap(
        self,
        candidate: CandidateEvidence,
        candidates: Iterable[CandidateEvidence],
    ) -> Optional[float]:
        """Return the largest configured non-contiguous duplicate gap."""
        normalized = normalize_text(candidate.text)
        if not normalized:
            return None
        gaps = []
        for other in candidates:
            if other.id == candidate.id or normalize_text(other.text) != normalized:
                continue
            if other.end <= candidate.start:
                gap = candidate.start - other.end
            elif other.start >= candidate.end:
                gap = other.start - candidate.end
            else:
                continue
            if gap >= self.config.repeated_phrase_gap_seconds:
                gaps.append(gap)
        return round(max(gaps), 6) if gaps else None


__all__ = ["EvidenceRiskScorer", "RiskAssessment", "RiskScoringConfig", "normalize_text"]
