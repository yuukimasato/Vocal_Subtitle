"""Classification of acoustic energy and VAD-confirmed speech events."""

from __future__ import annotations

import logging

import numpy as np

from .skeleton import compute_vad_overlap

logger = logging.getLogger(__name__)


def classify_energy_type(
    audio: np.ndarray, sample_rate: int, start: float, end: float
) -> str:
    segment = audio[int(start * sample_rate) : int(end * sample_rate)]
    if len(segment) < 256:
        return "unknown"
    try:
        autocorr = np.correlate(segment, segment, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        peaks = []
        for i in range(1, min(len(autocorr) - 1, sample_rate // 50)):
            if (
                autocorr[i] > autocorr[i - 1]
                and autocorr[i] > autocorr[i + 1]
                and autocorr[i] > 0.15
            ):
                peaks.append((i, autocorr[i]))
        if not peaks:
            return "transient_noise"
        harmonics_ratio = sum(value for _, value in peaks[:10]) / len(peaks[:10])
        return "music_or_tonal" if harmonics_ratio > 0.3 else "transient_noise"
    except Exception:
        return "unknown"


def classify_acoustic_events(
    skeleton: list[tuple[float, float]],
    silero_segments: list,
    audio: np.ndarray,
    sample_rate: int,
) -> list[dict]:
    classified = []
    for start, end in skeleton:
        silero_overlap = compute_vad_overlap(start, end, silero_segments)
        if silero_overlap > 0.5:
            event_type, confidence = "human_speech", "high"
        elif silero_overlap > 0.1:
            event_type, confidence = "human_speech", "low"
        else:
            event_type, confidence = "non_human_energy", "high"
        entry = {
            "start": start,
            "end": end,
            "type": event_type,
            "confidence": confidence,
            "silero_overlap_ratio": round(silero_overlap, 2),
        }
        if event_type == "non_human_energy":
            entry["energy_subtype"] = classify_energy_type(
                audio, sample_rate, start, end
            )
        classified.append(entry)
    non_human_count = sum(item["type"] == "non_human_energy" for item in classified)
    if non_human_count:
        logger.info(
            "Acoustic event classification: %d total, %d non-human energy (%.0f%%), %d human speech",
            len(classified),
            non_human_count,
            non_human_count / max(len(classified), 1) * 100,
            sum(item["type"] == "human_speech" for item in classified),
        )
    return classified


_classify_energy_type = classify_energy_type
_compute_vad_overlap = compute_vad_overlap
