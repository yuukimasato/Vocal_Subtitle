"""Global diarization identity normalization.

This module only merges identities that the diarization backend already
returned. It never invents speakers from pauses or subtitle ordering.
"""

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from .base import DiarizationResult, SpeakerTurn


def _ordered_speakers(result: DiarizationResult) -> list[int]:
    turns = list(result.turns) + list(result.exclusive_turns)
    return list(dict.fromkeys(
        turn.speaker_id
        for turn in sorted(turns, key=lambda item: (item.start, item.end))
    ))


def _profile(audio: np.ndarray, sample_rate: int, turns: list[SpeakerTurn]) -> np.ndarray:
    """Build a small dependency-free acoustic profile for one speaker."""
    samples = []
    total = len(audio)
    for turn in turns:
        start = max(0, min(total, int(turn.start * sample_rate)))
        end = max(start, min(total, int(turn.end * sample_rate)))
        if end > start:
            samples.append(np.asarray(audio[start:end], dtype=np.float32))
    if not samples:
        return np.empty(0, dtype=np.float32)

    signal = np.concatenate(samples)
    if signal.size == 0:
        return np.empty(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(signal * signal)))
    zcr = float(np.mean(signal[:-1] * signal[1:] < 0)) if signal.size > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(signal[: min(signal.size, sample_rate * 2)]))
    freqs = np.arange(spectrum.size, dtype=np.float32)
    spectral_total = float(spectrum.sum())
    centroid = float((freqs * spectrum).sum() / spectral_total) if spectral_total else 0.0
    return np.array([rms, zcr, centroid / max(sample_rate, 1)], dtype=np.float32)


def canonicalize_diarization_result(
    result: DiarizationResult,
    *,
    max_speakers: Optional[int] = None,
    audio: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
) -> DiarizationResult:
    """Normalize speaker IDs and enforce an optional maximum.

    The first-seen speaker order defines canonical IDs. When the backend
    returns too many identities, profiles are used when audio is available;
    otherwise the overflow is assigned deterministically to the least-used
    canonical identity. A smaller result is never split or padded.
    """
    ordered = _ordered_speakers(result)
    raw_count = len(ordered)
    diagnostics = dict(result.diagnostics or {})
    diagnostics["raw_diarization_speaker_count"] = raw_count

    if not ordered:
        diagnostics.update({
            "canonical_speaker_count": 0,
            "speaker_merge_map": {},
            "canonicalization_status": "ok",
        })
        return replace(result, speaker_count=0, diagnostics=diagnostics)

    if max_speakers is not None and max_speakers <= 0:
        raise ValueError("max_speakers must be a positive integer")

    target_count = min(raw_count, max_speakers) if max_speakers else raw_count
    roots = ordered[:target_count]
    turns_by_speaker: Dict[int, list[SpeakerTurn]] = {speaker: [] for speaker in ordered}
    for turn in list(result.turns) + list(result.exclusive_turns):
        turns_by_speaker.setdefault(turn.speaker_id, []).append(turn)

    profiles: Dict[int, np.ndarray] = {}
    if audio is not None and max_speakers and raw_count > max_speakers:
        array = np.asarray(audio, dtype=np.float32)
        for speaker in ordered:
            profiles[speaker] = _profile(array, sample_rate, turns_by_speaker[speaker])

    merge_map: Dict[int, int] = {speaker: index for index, speaker in enumerate(roots)}
    root_durations = {
        index: sum(turn.duration for turn in turns_by_speaker[speaker])
        for index, speaker in enumerate(roots)
    }
    root_first_start = {
        index: min((turn.start for turn in turns_by_speaker[speaker]), default=0.0)
        for index, speaker in enumerate(roots)
    }

    for speaker in ordered[target_count:]:
        candidates = roots
        source_profile = profiles.get(speaker, np.empty(0, dtype=np.float32))
        distances = []
        for index, root in enumerate(candidates):
            root_profile = profiles.get(root, np.empty(0, dtype=np.float32))
            if source_profile.size and root_profile.size:
                distance = (
                    float(np.linalg.norm(source_profile - root_profile)),
                    0.0,
                    index,
                )
            else:
                first_start = min(
                    (turn.start for turn in turns_by_speaker[speaker]), default=0.0
                )
                distance = (
                    root_durations[index],
                    abs(root_first_start[index] - first_start),
                    index,
                )
            distances.append((distance, index))
        target_index = min(distances, key=lambda item: item[0])[1]
        merge_map[speaker] = target_index
        root_durations[target_index] += sum(
            turn.duration for turn in turns_by_speaker[speaker]
        )

    def remap(turn: SpeakerTurn) -> SpeakerTurn:
        return replace(turn, speaker_id=merge_map[turn.speaker_id])

    turns = [remap(turn) for turn in result.turns]
    exclusive_turns = [remap(turn) for turn in result.exclusive_turns]
    merged = any(source != target for source, target in merge_map.items())
    status = "degraded" if merged else (result.status or "ok")
    diagnostics.update({
        "canonical_speaker_count": target_count,
        "speaker_merge_map": merge_map,
        "canonicalization_status": status,
    })
    if merged:
        diagnostics["canonicalization_reason"] = (
            f"backend returned {raw_count} speakers; constrained to {target_count}"
        )

    return replace(
        result,
        turns=turns,
        exclusive_turns=exclusive_turns,
        speaker_count=target_count,
        status=status,
        diagnostics=diagnostics,
    )
