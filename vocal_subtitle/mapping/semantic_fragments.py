"""Adaptive pause classification and semantic PhysicalFragment segmentation.

Converts a continuous stream of aligned WordAllocations into short
PhysicalFragment candidates suitable for LLM semantic decision-making.
Physical hard boundaries (PhysicalClip, confirmed speaker change, long
pause, genuine overlap) are established first; only then can semantic
grouping combine fragments within those safe boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..physical.word_alignment import WordAllocation

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class AdaptivePauseThresholds:
    """Speaker-specific pause thresholds calibrated from the word stream.

    All values in seconds. Each speaker's distribution is estimated
    using robust quantiles and clamped to design ranges.
    """

    speaker_id: Optional[int] = None
    micro_pause_max: float = 0.150    # initial ≈120-250ms, calibrated
    sentence_pause_max: float = 0.350  # initial ≈250-500ms, calibrated
    long_pause_min: float = 0.600      # initial ≈500ms+, calibrated
    speech_rate_wps: float = 2.5       # words per second estimate
    sample_count: int = 0
    calibration_quality: str = "default"  # "calibrated" | "default" | "fallback"


@dataclass
class PhysicalFragment:
    """A contiguous short-segment candidate for semantic decision-making.

    Contains ordered word IDs, physical span, candidate speaker, pause
    classification, and evidence references. The LLM operates on these
    fragments but never receives their numerical times.
    """

    id: str
    word_ids: List[str] = field(default_factory=list)
    physical_start: float = 0.0
    physical_end: float = 0.0
    candidate_speaker: Optional[int] = None
    speaker_confidence: float = 0.0
    speaker_status: str = "unknown"
    pause_class: str = ""  # "micro" | "sentence" | "long" | "hard_split" | "overlap"
    pause_duration: float = 0.0
    hard_split_before: bool = False
    hard_split_reason: str = ""
    genuine_overlap: bool = False
    language: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "word_ids": list(self.word_ids),
            "physical_start": self.physical_start,
            "physical_end": self.physical_end,
            "candidate_speaker": self.candidate_speaker,
            "speaker_confidence": self.speaker_confidence,
            "speaker_status": self.speaker_status,
            "pause_class": self.pause_class,
            "pause_duration": self.pause_duration,
            "hard_split_before": self.hard_split_before,
            "hard_split_reason": self.hard_split_reason,
            "genuine_overlap": self.genuine_overlap,
            "language": self.language,
            "evidence_ids": list(self.evidence_ids),
        }


# ------------------------------------------------------------------
# Pause calibration
# ------------------------------------------------------------------


def _estimate_gap_distribution(
    gaps: np.ndarray,
) -> Tuple[float, float, float]:
    """Estimate micro/sentence/long thresholds from gap data.

    Uses robust quantiles: p25 for micro, p60 for sentence, p85 for long.
    Clamped into design ranges.
    """
    if len(gaps) < 2:
        return 0.150, 0.350, 0.600

    # Remove extreme outliers (top 5%)
    cutoff = np.percentile(gaps, 95)
    filtered = gaps[gaps <= cutoff]

    if len(filtered) < 2:
        filtered = gaps

    p25 = float(np.percentile(filtered, 25))
    p60 = float(np.percentile(filtered, 60))
    p85 = float(np.percentile(filtered, 85))

    micro = np.clip(p25, 0.100, 0.250)
    sentence = np.clip(p60, 0.200, 0.500)
    long_ = np.clip(p85, 0.400, 0.800)

    # Maintain ordering
    if micro >= sentence:
        sentence = micro + 0.050
    if sentence >= long_:
        long_ = sentence + 0.100

    return float(micro), float(sentence), float(long_)


def calibrate_speaker_thresholds(
    allocations: Sequence[WordAllocation],
    *,
    speaker_id: Optional[int] = None,
    min_samples: int = 3,
) -> AdaptivePauseThresholds:
    """Calibrate pause thresholds per speaker from aligned word gaps.

    For each speaker (or all words if speaker_id is None), computes
    inter-word gap quantiles and maps them to micro/sentence/long
    thresholds clamped to design ranges.
    """
    # Filter by speaker
    if speaker_id is not None:
        relevant = [a for a in allocations if a.speaker_id == speaker_id]
    else:
        relevant = list(allocations)

    if not relevant:
        return AdaptivePauseThresholds(speaker_id=speaker_id, calibration_quality="fallback")

    # Extract gaps from aligned times
    gaps: List[float] = []
    words = sorted(relevant, key=lambda a: getattr(a, "aligned_start", a.word.raw_start))

    for i in range(1, len(words)):
        prev_end = getattr(words[i - 1], "aligned_end", words[i - 1].word.raw_end)
        curr_start = getattr(words[i], "aligned_start", words[i].word.raw_start)
        gap = curr_start - prev_end
        if 0 < gap < 10.0:  # keep reasonable gaps
            gaps.append(gap)

    if len(gaps) < min_samples:
        return AdaptivePauseThresholds(speaker_id=speaker_id, calibration_quality="fallback")

    gaps_arr = np.array(gaps, dtype=np.float64)
    micro, sentence, long_ = _estimate_gap_distribution(gaps_arr)

    # Estimate speech rate
    total_duration = words[-1].word.raw_end - words[0].word.raw_start
    word_count = len(words)
    speech_rate = word_count / max(total_duration, 0.1)

    return AdaptivePauseThresholds(
        speaker_id=speaker_id,
        micro_pause_max=micro,
        sentence_pause_max=sentence,
        long_pause_min=long_,
        speech_rate_wps=round(speech_rate, 1),
        sample_count=len(gaps),
        calibration_quality="calibrated" if len(gaps) >= min_samples else "default",
    )


# ------------------------------------------------------------------
# Fragment segmentation
# ------------------------------------------------------------------


def build_physical_fragments(
    allocations: Sequence[WordAllocation],
    *,
    thresholds: Optional[AdaptivePauseThresholds] = None,
    fragment_id_prefix: str = "frag",
    min_fragment_words: int = 1,
    max_fragment_words: int = 15,
    max_fragment_duration: float = 8.0,
    hard_split_speaker_change: bool = True,
) -> List[PhysicalFragment]:
    """Build PhysicalFragments from aligned word allocations.

    Hard boundaries are established first:
    1. PhysicalClip changes
    2. Confirmed speaker changes
    3. Long pauses (>= long_pause_min)
    4. Genuine overlap track boundaries

    Within hard boundaries, micro/sentence pauses mark soft candidate
    split points for the LLM semantic stage.

    Args:
        allocations: Aligned word allocations, time-ordered.
        thresholds: Pre-calibrated pause thresholds; auto-calibrated if None.
        fragment_id_prefix: Prefix for fragment IDs.
        min_fragment_words: Minimum words per fragment.
        max_fragment_words: Maximum words per fragment (soft limit).
        max_fragment_duration: Maximum fragment duration in seconds.
        hard_split_speaker_change: Treat confirmed speaker change as hard split.

    Returns:
        Ordered list of PhysicalFragment.
    """
    if not allocations:
        return []

    if thresholds is None:
        thresholds = calibrate_speaker_thresholds(allocations)

    words = sorted(allocations, key=lambda a: getattr(a, "aligned_start", a.word.raw_start))

    # Group words by their aligned positions
    fragments: List[PhysicalFragment] = []
    current_word_ids: List[str] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_speaker: Optional[int] = None
    current_speaker_status: str = "unknown"
    current_language: Optional[str] = None
    current_evidence_ids: List[str] = []
    fragment_index = 0

    def _emit_fragment(hard_split: bool, reason: str) -> None:
        nonlocal fragment_index, current_start, current_end
        if not current_word_ids:
            return
        fragment_index += 1
        fragments.append(PhysicalFragment(
            id=f"{fragment_id_prefix}-{fragment_index:04d}",
            word_ids=list(current_word_ids),
            physical_start=current_start or 0.0,
            physical_end=current_end or 0.0,
            candidate_speaker=current_speaker,
            speaker_status=current_speaker_status,
            pause_class="hard_split" if hard_split else "",
            hard_split_before=False,
            hard_split_reason=reason if hard_split else "",
            genuine_overlap=False,
            language=current_language,
            evidence_ids=list(current_evidence_ids),
        ))
        current_word_ids.clear()
        current_start = None
        current_end = None

    previous_clip: Optional[str] = None
    previous_confirmed_speaker: Optional[int] = None
    previous_end: Optional[float] = None

    for alloc in words:
        word = alloc.word
        aligned_start = getattr(alloc, "aligned_start", word.raw_start)
        aligned_end = getattr(alloc, "aligned_end", word.raw_end)

        # Determine clip
        current_clip = alloc.clip_ids[0] if alloc.clip_ids else None

        # Determine speaker
        spk = alloc.speaker_id
        spk_status = "confirmed" if spk is not None and alloc.speaker_source == "diarization" else "unknown"

        # Hard boundary checks
        hard_reason = ""
        is_hard = False

        # 1. PhysicalClip change
        if previous_clip is not None and current_clip is not None and current_clip != previous_clip:
            is_hard = True
            hard_reason = f"PhysicalClip: {previous_clip} -> {current_clip}"

        # 2. Confirmed speaker change
        if hard_split_speaker_change and previous_confirmed_speaker is not None and spk is not None:
            if spk != previous_confirmed_speaker and spk_status == "confirmed":
                is_hard = True
                hard_reason = f"Speaker: {previous_confirmed_speaker} -> {spk}"

        # 3. Long pause
        if previous_end is not None:
            gap_duration = aligned_start - previous_end
            if gap_duration >= thresholds.long_pause_min:
                is_hard = True
                hard_reason = f"LongPause: {gap_duration:.3f}s >= {thresholds.long_pause_min:.3f}s"

        # 4. Genuine overlap (WordAllocation overlap detection)
        if alloc.warnings and any("overlap" in w.lower() for w in alloc.warnings):
            is_hard = True
            hard_reason = "OverlapRegion"

        # Emit fragment on hard boundary
        if is_hard and current_word_ids:
            _emit_fragment(True, hard_reason)
            if fragments:
                fragments[-1].hard_split_before = False  # reset for next fragment
            if is_hard and hard_reason:
                pass  # mark next fragment start as hard

        # Add word to current fragment
        if not current_word_ids:
            current_start = aligned_start
            current_speaker = spk
            current_speaker_status = spk_status
            current_language = word.language
            if is_hard:
                pass  # the next fragment will have hard_split_before=True

        current_word_ids.append(word.id)
        current_end = aligned_end
        if current_speaker is None and spk is not None:
            current_speaker = spk
        current_evidence_ids.append(f"evidence-{word.id}")

        # Soft split: max words or duration exceeded
        duration = (current_end or 0) - (current_start or 0)
        soft_reason = ""
        if len(current_word_ids) >= max_fragment_words:
            soft_reason = "max_words"
        elif duration >= max_fragment_duration:
            soft_reason = "max_duration"
        elif previous_end is not None:
            gap = aligned_start - previous_end
            if gap >= thresholds.sentence_pause_max:
                soft_reason = f"sentence_pause: {gap:.3f}s"

        if soft_reason and current_word_ids:
            pause_dur = aligned_start - (previous_end or aligned_start) if previous_end else 0.0
            _emit_fragment(False, soft_reason)
            if fragments:
                fragments[-1].pause_duration = pause_dur
                if "sentence_pause" in soft_reason:
                    fragments[-1].pause_class = "sentence"
                elif "micro" in soft_reason:
                    fragments[-1].pause_class = "micro"
                else:
                    fragments[-1].pause_class = "sentence"

        previous_clip = current_clip
        if spk_status == "confirmed":
            previous_confirmed_speaker = spk
        previous_end = aligned_end

    # Emit final fragment
    if current_word_ids:
        _emit_fragment(False, "final")

    # Post-process: mark hard_split_before on fragments after hard boundaries
    for i in range(1, len(fragments)):
        if fragments[i - 1].hard_split_reason:
            fragments[i].hard_split_before = True
            # Copy hard_split_reason from previous as the cause
            if not fragments[i].hard_split_reason:
                pass  # reason already on the previous fragment

    return fragments
