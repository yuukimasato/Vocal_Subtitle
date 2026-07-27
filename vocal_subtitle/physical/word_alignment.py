"""Constrained ASR word-to-physical-boundary alignment.

Treats ASR word timestamps as observations and VAD/FFmpeg/RMS as
candidate anchors and legal range constraints. Never stretches the
final word to the end of the VAD segment.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .allocator import WordAllocation
from .boundary_arbiter import (
    BoundaryArbiter,
    BoundaryCandidate,
    BoundaryDecision,
)
from .subtitle_bins import PhysicalSubtitleBin, assign_word_to_bin
from .timeline import PhysicalTimeline

logger = logging.getLogger(__name__)


def _dedupe_times(values: Sequence[tuple[float, str]]) -> list[tuple[float, str]]:
    """Deduplicate candidate timestamps without losing source provenance."""
    grouped: dict[float, list[str]] = {}
    for value, label in values:
        if not math.isfinite(value) or value < 0:
            continue
        key = round(float(value), 4)
        grouped.setdefault(key, []).append(label)
    return [
        (time, "+".join(dict.fromkeys(labels)))
        for time, labels in sorted(grouped.items())
    ]


def _rms_features(
    audio: np.ndarray | None,
    sample_rate: int,
    time: float,
    *,
    window: float = 0.02,
) -> dict[str, float]:
    """Measure local level, slope and valley evidence around one candidate."""
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return {}
    center = max(0, min(len(audio), int(time * sample_rate)))
    width = max(1, int(window * sample_rate))

    def rms(start: int, end: int) -> float:
        frame = audio[max(0, start):min(len(audio), end)]
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

    before = rms(center - width, center)
    current = rms(center - width // 2, center + width // 2)
    after = rms(center, center + width)
    neighbor = max(before, after, 1e-8)
    return {
        "rms_level": current,
        "rms_gradient": after - before,
        "rms_valley": max(0.0, min(1.0, 1.0 - current / neighbor)),
        "rms_before": before,
        "rms_after": after,
    }


def _vad_features(
    time: float,
    vad_segments: Sequence[Any] | None,
) -> dict[str, float]:
    if vad_segments is None:
        return {}
    active = [
        float(getattr(item, "confidence", 1.0))
        for item in vad_segments
        if float(getattr(item, "start", 0.0)) <= time <= float(
            getattr(item, "end", 0.0)
        )
    ]
    return {
        "vad_probability": max(0.0, min(1.0, max(active, default=0.0))),
        "vad_active": 1.0 if active else 0.0,
    }


def _ffmpeg_boundaries(result: Mapping[str, Any] | None) -> list[tuple[float, str]]:
    if not isinstance(result, Mapping):
        return []
    values: list[tuple[float, str]] = []
    for field, label in (
        ("raw_silence_intervals", "ffmpeg_silence"),
        ("skeleton", "ffmpeg_skeleton"),
        ("coarse_speech", "ffmpeg_coarse"),
    ):
        for item in result.get(field, ()) or ():
            if isinstance(item, Mapping):
                start, end = item.get("start"), item.get("end")
            else:
                start = getattr(item, "start", item[0] if isinstance(item, (tuple, list)) else None)
                end = getattr(item, "end", item[1] if isinstance(item, (tuple, list)) else None)
            if start is None or end is None:
                continue
            values.extend(((float(start), f"{label}_start"), (float(end), f"{label}_end")))
    return values


def _ffmpeg_features(time: float, result: Mapping[str, Any] | None) -> dict[str, float]:
    boundaries = _ffmpeg_boundaries(result)
    if not boundaries:
        return {}
    distances = [abs(time - value) for value, _ in boundaries]
    silence = result.get("raw_silence_intervals", ()) if isinstance(result, Mapping) else ()
    in_silence = any(
        isinstance(item, (tuple, list)) and len(item) >= 2
        and float(item[0]) <= time <= float(item[1])
        for item in silence or ()
    )
    return {
        "ffmpeg_boundary_distance": min(distances),
        "ffmpeg_boundary_proximity": max(0.0, 1.0 - min(distances) / 0.2),
        "ffmpeg_silence": 1.0 if in_silence else 0.0,
    }


def _evidence_features(
    time: float,
    timeline: Optional[PhysicalTimeline],
) -> tuple[dict[str, float], tuple[str, ...]]:
    if timeline is None:
        return {}, ()
    nearby = [
        item for item in timeline.speech_evidence_spans
        if item.start - 0.2 <= time <= item.end + 0.2
    ]
    distances = [min(abs(time - item.start), abs(time - item.end)) for item in nearby]
    sources = {item.source for item in nearby}
    return (
        {
            "evidence_boundary_distance": min(distances) if distances else 1.0,
            "evidence_source_count": float(len(sources)),
            "source_consistency": min(1.0, len(sources) / 3.0),
        },
        tuple(item.id for item in nearby),
    )


def _candidate_features(
    time: float,
    *,
    audio: np.ndarray | None,
    sample_rate: int,
    vad_segments: Sequence[Any] | None,
    ffmpeg_result: Mapping[str, Any] | None,
    noise_profile: Any | None,
    timeline: Optional[PhysicalTimeline],
) -> tuple[dict[str, float], tuple[str, ...]]:
    features: dict[str, float] = {}
    features.update(_rms_features(audio, sample_rate, time))
    features.update(_vad_features(time, vad_segments))
    features.update(_ffmpeg_features(time, ffmpeg_result))
    evidence_features, evidence_ids = _evidence_features(time, timeline)
    features.update(evidence_features)
    if noise_profile is not None and hasattr(noise_profile, "candidate_features"):
        features.update(noise_profile.candidate_features(time))
    return features, evidence_ids


def _score_with_features(
    boundary_type: str,
    candidate: float,
    *,
    asr_time: float,
    features: Mapping[str, float],
    asr_confidence: float,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Score all available signals and return auditable components."""
    asr_proximity = max(0.0, 1.0 - abs(candidate - asr_time) / 0.2)
    rms_gradient = float(features.get("rms_gradient", 0.0))
    rms_scale = max(abs(float(features.get("rms_before", 0.0))), abs(float(features.get("rms_after", 0.0))), 1e-6)
    gradient = max(0.0, min(1.0, (rms_gradient / rms_scale + 1.0) / 2.0))
    if boundary_type == "end":
        gradient = 1.0 - gradient
    rms_score = max(float(features.get("rms_valley", 0.0)), gradient)
    vad_score = float(features.get("vad_probability", 0.5))
    ffmpeg_score = float(features.get("ffmpeg_boundary_proximity", 0.0))
    source_score = float(features.get("source_consistency", 0.0))
    noise_stability = float(features.get("noise_stability", 0.0))
    noise_score = 1.0 - float(features.get("noise_fallback", 0.0))
    components = {
        "asr": max(0.0, min(1.0, asr_proximity * max(0.0, min(1.0, asr_confidence)))),
        "rms": max(0.0, min(1.0, rms_score)),
        "vad": max(0.0, min(1.0, vad_score)),
        "ffmpeg": max(0.0, min(1.0, ffmpeg_score)),
        "noise": max(0.0, min(1.0, (noise_stability + noise_score) / 2.0)),
        "source_consistency": max(0.0, min(1.0, source_score)),
    }
    weights = {
        "asr": 0.28,
        "rms": 0.22,
        "vad": 0.16,
        "ffmpeg": 0.14,
        "noise": 0.10,
        "source_consistency": 0.10,
    }
    score = sum(weights[key] * components[key] for key in weights)
    return score, tuple((key, round(value, 6)) for key, value in components.items())


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
    audio: np.ndarray | None = None,
    sample_rate: int = 16000,
    vad_segments: Sequence[Any] | None = None,
    ffmpeg_result: Mapping[str, Any] | None = None,
    noise_profile: Any | None = None,
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
    arbiter = BoundaryArbiter()

    for alloc in allocations:
        word = alloc.word
        raw_start = word.raw_start
        raw_end = word.raw_end
        tier = _asr_confidence_tier(word.confidence)
        search_radius = _search_window_for_tier(tier)

        # Determine the primary bin for this word
        assigned_bin = assign_word_to_bin(word, physical_bins)
        primary_bin_id = (
            alloc.physical_bin_id
            or (assigned_bin.id if assigned_bin is not None else None)
        )
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
        # Candidate sources are deliberately local: ASR, bin/evidence edges,
        # VAD and FFmpeg transitions. RMS is sampled at every candidate and
        # contributes a feature rather than silently creating a second timeline.
        start_candidates: list[tuple[float, str]] = [(raw_start, "asr_start")]
        for b in relevant_bins:
            if abs(b.start - raw_start) < search_radius * 2:
                start_candidates.append((b.start, "bin_start"))
        for item in timeline.speech_evidence_spans if timeline else ():
            for value, label in ((item.start, f"{item.source}_start"), (item.end, f"{item.source}_end")):
                if abs(value - raw_start) < search_radius * 2:
                    start_candidates.append((value, label))
        for value, label in _ffmpeg_boundaries(ffmpeg_result):
            if abs(value - raw_start) < search_radius * 2:
                start_candidates.append((value, label))
        for item in vad_segments or ():
            for value, label in ((float(item.start), "vad_start"), (float(item.end), "vad_end")):
                if abs(value - raw_start) < search_radius * 2:
                    start_candidates.append((value, label))

        previous_end = None
        if result:
            previous_end = max(
                getattr(a, "aligned_end", a.word.raw_end)
                for a in result[-1:]
            )
        scored_start_candidates = []
        for time, label in _dedupe_times(start_candidates):
            rejection_reasons = []
            if not _is_within_legal_range(
                time, relevant_bins, timeline, clip_start, clip_end
            ):
                rejection_reasons.append("outside_physical_owner")
            if previous_end is not None and time < previous_end - 0.005:
                rejection_reasons.append("breaks_monotonicity")
            if abs(time - raw_start) > search_radius:
                rejection_reasons.append("outside_search_window")
            features, feature_evidence_ids = _candidate_features(
                time,
                audio=audio,
                sample_rate=sample_rate,
                vad_segments=vad_segments,
                ffmpeg_result=ffmpeg_result,
                noise_profile=noise_profile,
                timeline=timeline,
            )
            score, score_components = _score_with_features(
                "start",
                time,
                asr_time=raw_start,
                features=features,
                asr_confidence=word.confidence or 0.5,
            )
            scored_start_candidates.append(BoundaryCandidate(
                label=label,
                time=time,
                score=score,
                evidence_ids=tuple(dict.fromkeys(alloc.evidence_ids + feature_evidence_ids)),
                rejection_reasons=tuple(rejection_reasons),
                features=tuple(sorted((key, float(value)) for key, value in features.items())),
                score_components=score_components,
            ))
        start_decision = arbiter.decide(
            "start",
            scored_start_candidates,
            fallback_time=raw_start,
            accepted_reason="aligned_start",
            missing_reason="no_legal_start_candidate",
        )

        # --- END boundary ---
        end_candidates: list[tuple[float, str]] = [(raw_end, "asr_end")]
        for b in relevant_bins:
            if abs(b.end - raw_end) < search_radius * 2:
                end_candidates.append((b.end, "bin_end"))
        for item in timeline.speech_evidence_spans if timeline else ():
            for value, label in ((item.start, f"{item.source}_start"), (item.end, f"{item.source}_end")):
                if abs(value - raw_end) < search_radius * 2:
                    end_candidates.append((value, label))
        for value, label in _ffmpeg_boundaries(ffmpeg_result):
            if abs(value - raw_end) < search_radius * 2:
                end_candidates.append((value, label))
        for item in vad_segments or ():
            for value, label in ((float(item.start), "vad_start"), (float(item.end), "vad_end")):
                if abs(value - raw_end) < search_radius * 2:
                    end_candidates.append((value, label))

        scored_end_candidates = []
        for time, label in _dedupe_times(end_candidates):
            rejection_reasons = []
            if not _is_within_legal_range(
                time, relevant_bins, timeline, clip_start, clip_end
            ):
                rejection_reasons.append("outside_physical_owner")
            if time <= start_decision.boundary_time:
                rejection_reasons.append("non_positive_word_duration")
            if abs(time - raw_end) > search_radius:
                rejection_reasons.append("outside_search_window")
            features, feature_evidence_ids = _candidate_features(
                time,
                audio=audio,
                sample_rate=sample_rate,
                vad_segments=vad_segments,
                ffmpeg_result=ffmpeg_result,
                noise_profile=noise_profile,
                timeline=timeline,
            )
            score, score_components = _score_with_features(
                "end",
                time,
                asr_time=raw_end,
                features=features,
                asr_confidence=word.confidence or 0.5,
            )
            scored_end_candidates.append(BoundaryCandidate(
                label=label,
                time=time,
                score=score,
                evidence_ids=tuple(dict.fromkeys(alloc.evidence_ids + feature_evidence_ids)),
                rejection_reasons=tuple(rejection_reasons),
                features=tuple(sorted((key, float(value)) for key, value in features.items())),
                score_components=score_components,
            ))
        end_decision = arbiter.decide(
            "end",
            scored_end_candidates,
            fallback_time=raw_end,
            accepted_reason="aligned_end",
            missing_reason="no_legal_end_candidate",
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
                candidate_scores=end_decision.candidate_scores,
                rejected_candidates=end_decision.rejected_candidates,
                candidate_diagnostics=end_decision.candidate_diagnostics,
            )
            end_decision = end_decision_limited

        new_allocation = replace(
            alloc,
            aligned_start=start_decision.boundary_time,
            aligned_end=end_decision.boundary_time,
            physical_bin_id=primary_bin_id,
            boundary_confidence=min(
                start_decision.confidence, end_decision.confidence
            ),
            alignment_status=(
                "aligned"
                if start_decision.accepted and end_decision.accepted
                else "degraded"
            ),
            start_boundary_decision=start_decision,
            end_boundary_decision=end_decision,
            boundary_evidence_ids=tuple(dict.fromkeys(
                start_decision.evidence_ids + end_decision.evidence_ids
            )),
        )

        result.append(new_allocation)

    return result
