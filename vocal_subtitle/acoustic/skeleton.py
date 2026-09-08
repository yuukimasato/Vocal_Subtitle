"""Pure acoustic skeleton queries using absolute-second coordinates."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..utils.audio_utils import AudioUtils


DEFAULT_HARD_SILENCE_SECONDS = 0.4


def group_speech_intervals(
    intervals: Iterable[Tuple[float, float]],
    *,
    max_gap: float = DEFAULT_HARD_SILENCE_SECONDS,
) -> List[Tuple[float, float]]:
    """Group intervals for ASR context without crossing hard silence.

    The result is an ASR input policy only. Callers must retain the original
    intervals for physical projection and coverage auditing.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    grouped: List[Tuple[float, float]] = []
    for start, end in ordered:
        if not grouped or start - grouped[-1][1] > max_gap:
            grouped.append((start, end))
            continue
        grouped[-1] = (grouped[-1][0], max(grouped[-1][1], end))
    return grouped


def is_time_in_speech(t: float, skeleton: List[Tuple[float, float]]) -> bool:
    return any(start <= t <= end for start, end in skeleton)


def has_speech_in_range(start: float, end: float, skeleton: List[Tuple[float, float]]) -> bool:
    return any(speech_start < end and speech_end > start for speech_start, speech_end in skeleton)


def rms_energy_check(audio: np.ndarray, sample_rate: int, time_point: float, window_ms: int = 50, threshold_ratio: float = 2.0) -> bool:
    silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)
    half_window = window_ms / 2000.0
    start = max(0.0, time_point - half_window)
    end = min(len(audio) / sample_rate, time_point + half_window)
    gap_rms = AudioUtils.get_segment_rms(audio, start, end, sample_rate)
    return gap_rms > silence_rms * threshold_ratio


def silence_confirmed(audio: Optional[np.ndarray], sample_rate: int, time_point: float, *, distance: float) -> bool:
    if audio is None:
        return distance <= 0.03
    return not rms_energy_check(audio, sample_rate, time_point, window_ms=50, threshold_ratio=2.0)


def compute_vad_overlap(start: float, end: float, vad_segments: List) -> float:
    duration = end - start
    if duration <= 0:
        return 0.0
    overlap_total = 0.0
    for segment in vad_segments:
        segment_start = segment.start if hasattr(segment, "start") else segment[0]
        segment_end = segment.end if hasattr(segment, "end") else segment[1]
        overlap_start = max(start, segment_start)
        overlap_end = min(end, segment_end)
        if overlap_start < overlap_end:
            overlap_total += overlap_end - overlap_start
    return min(1.0, overlap_total / duration)


_is_time_in_speech = is_time_in_speech
_has_speech_in_range = has_speech_in_range
_rms_energy_check = rms_energy_check
_silence_confirmed = silence_confirmed
_compute_vad_overlap = compute_vad_overlap
