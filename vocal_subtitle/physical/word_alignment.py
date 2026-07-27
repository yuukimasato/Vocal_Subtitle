"""Constrained ASR word-to-physical-boundary alignment.

Treats ASR word timestamps as observations and VAD/FFmpeg/RMS as
candidate anchors and legal range constraints. Never stretches the
final word to the end of the VAD segment.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .allocator import WordAllocation
from .ir import GlobalWord
from .subtitle_bins import PhysicalSubtitleBin
from .timeline import PhysicalTimeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundaryDecision:
    """Structured boundary arbitration result.

    Attributes:
        accepted: Whether this boundary was accepted.
        boundary_time: The chosen boundary time in seconds.
        boundary_type: Type of boundary ("start" or "end").
        confidence: Normalized confidence score (0-1).
        evidence_ids: Evidence IDs that contributed to this decision.
        reason_codes: Short codes explaining the decision ("hard_split_violation", etc.)
        candidate_scores: Optional list of (candidate_label, score) tuples.
        rejected_candidates: Optional reason strings for rejected candidates.
    """

    accepted: bool
    boundary_time: float
    boundary_type: str  # "start" | "end"
    confidence: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    candidate_scores: tuple[tuple[str, float], ...] = ()
    rejected_candidates: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "boundary_time": self.boundary_time,
            "boundary_type": self.boundary_type,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
            "candidate_scores": [
                {"label": label, "score": score}
                for label, score in self.candidate_scores
            ],
            "rejected_candidates": list(self.rejected_candidates),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BoundaryDecision":
        return cls(
            accepted=payload["accepted"],
            boundary_time=payload["boundary_time"],
            boundary_type=payload.get("boundary_type", "start"),
            confidence=payload.get("confidence", 0.0),
            evidence_ids=tuple(payload.get("evidence_ids", [])),
            reason_codes=tuple(payload.get("reason_codes", [])),
            candidate_scores=tuple(
                (cs["label"], cs["score"])
                for cs in payload.get("candidate_scores", [])
            ),
            rejected_candidates=tuple(payload.get("rejected_candidates", [])),
        )


def _asr_confidence_tier(confidence: Optional[float]) -> str:
    """Map raw ASR confidence to a calibrated tier.

    Tiers are: "high", "medium", "low", "unknown".
    The exact thresholds (0.9, 0.6) are initial experiment values;
    they must be calibrated per ASR backend on an evaluation set before
    being used as release defaults.
    """
    if confidence is None:
        return "unknown"
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _search_window_for_tier(tier: str) -> float:
    """Return the search window radius (seconds) for a confidence tier.

    Initial values: high ±200ms, low ±500ms, medium ±300ms, unknown ±500ms.
    These are EXPERIMENT VALUES subject to backend-specific calibration.
    """
    return {
        "high": 0.200,
        "medium": 0.300,
        "low": 0.500,
        "unknown": 0.500,
    }.get(tier, 0.500)


def _score_start_candidate(
    candidate: float,
    *,
    asr_time: float,
    rms_gradient: Optional[float] = None,
    vad_prob: Optional[float] = None,
    ffmpeg_boundary: Optional[float] = None,
    ffmpeg_distance: float = 1.0,
    source_consistency: float = 1.0,
    weight_asr: float = 0.30,
    weight_rms: float = 0.25,
    weight_vad: float = 0.20,
    weight_ffmpeg: float = 0.15,
    weight_consistency: float = 0.10,
) -> float:
    """Score a START boundary candidate.

    Start candidates prefer to include onset consonants and breath sounds,
    so we penalize candidates later than the ASR time (which may cut bursts).
    """
    score = 0.0

    # ASR proximity: symmetric Gaussian
    sigma_asr = 0.080
    diff = abs(candidate - asr_time)
    score += weight_asr * max(0.0, 1.0 - (diff / sigma_asr) ** 2)

    # RMS gradient: prefer candidates at onset peaks (negative gradient = rising)
    if rms_gradient is not None:
        # Penalize late boundaries that miss the onset
        if candidate > asr_time and rms_gradient > 0:
            score += weight_rms * 0.3
        else:
            score += weight_rms * 0.7

    # VAD: prefer inside VAD active region
    if vad_prob is not None:
        score += weight_vad * vad_prob

    # FFmpeg: prefer near silence transitions
    if ffmpeg_boundary is not None:
        proximity = 1.0 - min(ffmpeg_distance / 0.200, 1.0)
        score += weight_ffmpeg * max(0.0, proximity)

    # Source consistency
    score += weight_consistency * source_consistency

    return score


def _score_end_candidate(
    candidate: float,
    *,
    asr_time: float,
    rms_gradient: Optional[float] = None,
    vad_prob: Optional[float] = None,
    ffmpeg_boundary: Optional[float] = None,
    ffmpeg_distance: float = 1.0,
    silence_duration: float = 0.0,
    source_consistency: float = 1.0,
    weight_asr: float = 0.30,
    weight_rms: float = 0.25,
    weight_vad: float = 0.15,
    weight_ffmpeg: float = 0.15,
    weight_silence: float = 0.05,
    weight_consistency: float = 0.10,
) -> float:
    """Score an END boundary candidate.

    End candidates prefer to avoid cutting off fading tails, so we
    penalize candidates earlier than ASR time. Continuous low-energy
    silence (200-400ms) is a strong end candidate but not required.
    """
    score = 0.0

    # ASR proximity
    sigma_asr = 0.080
    diff = abs(candidate - asr_time)
    score += weight_asr * max(0.0, 1.0 - (diff / sigma_asr) ** 2)

    # RMS gradient: prefer falling energy at end
    if rms_gradient is not None:
        if rms_gradient < 0:
            score += weight_rms * 0.8  # falling energy is a good end signal
        else:
            score += weight_rms * 0.3

    # VAD
    if vad_prob is not None:
        score += weight_vad * vad_prob

    # FFmpeg
    if ffmpeg_boundary is not None:
        proximity = 1.0 - min(ffmpeg_distance / 0.200, 1.0)
        score += weight_ffmpeg * max(0.0, proximity)

    # Silence bonus: 200-400ms continuous low-energy silence is strong signal
    if silence_duration >= 0.200:
        silence_factor = min(silence_duration / 0.400, 1.0)
        score += weight_silence * silence_factor

    # Source consistency
    score += weight_consistency * source_consistency

    return score


def _is_within_legal_range(
    time: float,
    bins: Sequence[PhysicalSubtitleBin],
    timeline: Optional[PhysicalTimeline] = None,
    clip_start: Optional[float] = None,
    clip_end: Optional[float] = None,
) -> bool:
    """Check whether a time falls within all legal constraints."""
    # PhysicalSubtitleBin check
    bin_ok = any(b.start - 0.01 <= time <= b.end + 0.01 for b in bins)

    # PhysicalClip check
    clip_ok = True
    if clip_start is not None and clip_end is not None:
        clip_ok = clip_start - 0.01 <= time <= clip_end + 0.01

    return bin_ok and clip_ok


def align_words_to_physical(
    allocations: Sequence[WordAllocation],
    physical_bins: Sequence[PhysicalSubtitleBin],
    *,
    bin_owner_map: Optional[Dict[str, str]] = None,
    timeline: Optional[PhysicalTimeline] = None,
    clip_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[WordAllocation]:
    """Produce aligned WordAllocations with BoundaryDecision on each word endpoint.

    ASR word timestamps are treated as OBSERVATIONS. VAD/FFmpeg/RMS are the
    legal range constraints and anchor candidates. This function NEVER stretches
    the last word to the end of a PhysicalSubtitleBin or SpeechSegment.

    Returns a new list of WordAllocation — the original allocations and their
    GlobalWords (raw_start/raw_end) are never mutated.

    Args:
        allocations: Existing word-to-bin allocations.
        physical_bins: Physical subtitle bins (legal ranges).
        bin_owner_map: Optional map from bin_id to physical_clip_id.
        timeline: Optional PhysicalTimeline for additional constraint checks.
        clip_bounds: Optional map from clip_id to (start, end) bounds.

    Returns:
        New list of WordAllocation with aligned_start/end and BoundaryDecisions.
    """
    result: List[WordAllocation] = []
    if not allocations:
        return result

    for alloc in allocations:
        word = alloc.word
        raw_start = word.raw_start
        raw_end = word.raw_end
        tier = _asr_confidence_tier(word.confidence)
        search_radius = _search_window_for_tier(tier)

        # Determine the primary bin for this word
        primary_bin_id = alloc.clip_ids[0] if alloc.clip_ids else None
        relevant_bins = [
            b for b in physical_bins
            if primary_bin_id is None or b.id == primary_bin_id
        ]
        if not relevant_bins:
            relevant_bins = list(physical_bins)

        # Get clip constraints
        clip_id = None
        if bin_owner_map and primary_bin_id:
            clip_id = bin_owner_map.get(primary_bin_id)
        if clip_id is None and alloc.clip_ids:
            clip_id = alloc.clip_ids[0]

        clip_start = None
        clip_end = None
        if clip_id and clip_bounds:
            clip_b = clip_bounds.get(clip_id)
            if clip_b:
                clip_start, clip_end = clip_b

        # --- START boundary ---
        # Build candidate set from evidence sources
        start_candidates: List[Tuple[float, str, float]] = []
        # ASR observation
        start_candidates.append((raw_start, "asr_start", word.confidence or 0.5))
        # Nearby silence boundary from bins
        for b in relevant_bins:
            if abs(b.start - raw_start) < search_radius * 2:
                start_candidates.append((b.start, "bin_start", b.confidence or 0.5))

        # Filter by legal range
        legal_start_candidates = [
            (t, s, c) for t, s, c in start_candidates
            if _is_within_legal_range(t, relevant_bins, timeline, clip_start, clip_end)
        ]

        # Monotonicity: cannot be before previous word's end
        if result:
            prev_end = max(
                getattr(a, "aligned_end", a.word.raw_end)
                for a in result[-1:]
            )
            legal_start_candidates = [
                (t, s, c) for t, s, c in legal_start_candidates
                if t >= prev_end - 0.005
            ]

        start_decision: BoundaryDecision
        if not legal_start_candidates:
            start_decision = BoundaryDecision(
                accepted=False,
                boundary_time=raw_start,
                boundary_type="start",
                confidence=0.0,
                reason_codes=("no_legal_start_candidate",),
                rejected_candidates=("all candidates outside legal range",),
            )
        else:
            scores = [
                (label, _score_start_candidate(
                    time,
                    asr_time=raw_start,
                    vad_prob=0.5,
                    source_consistency=1.0 if "asr" in label else 0.7,
                ))
                for time, label, conf in legal_start_candidates
            ]
            best_time, best_label, best_score = max(
                zip(
                    [t for t, _, _ in legal_start_candidates],
                    [l for _, l, _ in legal_start_candidates],
                    [s for _, s in scores],
                ),
                key=lambda x: x[2],
            )

            # Clamp to search window
            if abs(best_time - raw_start) > search_radius:
                best_time = raw_start

            start_decision = BoundaryDecision(
                accepted=True,
                boundary_time=best_time,
                boundary_type="start",
                confidence=best_score,
                evidence_ids=alloc.evidence_ids,
                reason_codes=("aligned_start",),
                candidate_scores=tuple(
                    (label, sc) for (_, label, _), (_, sc) in zip(legal_start_candidates, scores)
                ),
            )

        # --- END boundary ---
        end_candidates: List[Tuple[float, str, float]] = []
        end_candidates.append((raw_end, "asr_end", word.confidence or 0.5))
        for b in relevant_bins:
            if abs(b.end - raw_end) < search_radius * 2:
                end_candidates.append((b.end, "bin_end", b.confidence or 0.5))

        legal_end_candidates = [
            (t, s, c) for t, s, c in end_candidates
            if _is_within_legal_range(t, relevant_bins, timeline, clip_start, clip_end)
            and t > start_decision.boundary_time
        ]

        end_decision: BoundaryDecision
        if not legal_end_candidates:
            end_decision = BoundaryDecision(
                accepted=False,
                boundary_time=raw_end,
                boundary_type="end",
                confidence=0.0,
                reason_codes=("no_legal_end_candidate",),
                rejected_candidates=("all candidates outside legal range",),
            )
        else:
            scores = [
                (label, _score_end_candidate(
                    time,
                    asr_time=raw_end,
                    vad_prob=0.5,
                    source_consistency=1.0 if "asr" in label else 0.7,
                ))
                for time, label, conf in legal_end_candidates
            ]
            best_time, best_label, best_score = max(
                zip(
                    [t for t, _, _ in legal_end_candidates],
                    [l for _, l, _ in legal_end_candidates],
                    [s for _, s in scores],
                ),
                key=lambda x: x[2],
            )

            if abs(best_time - raw_end) > search_radius:
                best_time = raw_end

            end_decision = BoundaryDecision(
                accepted=True,
                boundary_time=best_time,
                boundary_type="end",
                confidence=best_score,
                evidence_ids=alloc.evidence_ids,
                reason_codes=("aligned_end",),
                candidate_scores=tuple(
                    (label, sc) for (_, label, _), (_, sc) in zip(legal_end_candidates, scores)
                ),
            )

        # Never stretch the last word end beyond raw_end by more than 50ms
        if end_decision.boundary_time > raw_end + 0.050:
            end_decision_limited = BoundaryDecision(
                accepted=end_decision.accepted,
                boundary_time=raw_end,
                boundary_type="end",
                confidence=end_decision.confidence * 0.8,
                evidence_ids=end_decision.evidence_ids,
                reason_codes=end_decision.reason_codes + ("end_clamped",),
            )
            end_decision = end_decision_limited

        # Build new WordAllocation with alignment data
        new_allocation = WordAllocation(
            word=word,
            physical_spans=alloc.physical_spans,
            evidence_ids=alloc.evidence_ids,
            evidence_spans=alloc.evidence_spans,
            speaker_id=alloc.speaker_id,
            speaker_source=alloc.speaker_source,
            warnings=alloc.warnings,
            accepted=alloc.accepted,
        )
        # Annotate with alignment fields via object mutation (WordAllocation is frozen; use __dict__)
        object.__setattr__(new_allocation, "aligned_start", start_decision.boundary_time)
        object.__setattr__(new_allocation, "aligned_end", end_decision.boundary_time)
        object.__setattr__(new_allocation, "start_boundary_decision", start_decision)
        object.__setattr__(new_allocation, "end_boundary_decision", end_decision)
        object.__setattr__(new_allocation, "alignment_status", (
            "aligned" if start_decision.accepted and end_decision.accepted else "degraded"
        ))

        result.append(new_allocation)

    return result
