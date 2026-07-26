"""Physical subtitle bins derived from speech evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .timeline import PhysicalTimeline, SpeechEvidenceSpan


_SOURCE_PRIORITY = (
    "ffmpeg_skeleton",
    "boundary_fusion",
    "silero",
    "rms",
    "ffmpeg_coarse",
    "ten",
    "webrtc",
)


@dataclass(frozen=True)
class PhysicalSubtitleBin:
    """A non-overlapping physical speech range used as a subtitle container."""

    id: str
    start: float
    end: float
    source: str
    confidence: float | None = None
    evidence_ids: tuple[str, ...] = ()
    physical_clip_id: str | None = None
    boundary_resolution_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "physical_clip_id": self.physical_clip_id,
            "boundary_resolution_ms": self.boundary_resolution_ms,
        }


def build_physical_subtitle_bins(
    timeline: PhysicalTimeline,
    *,
    merge_gap: float = 0.05,
    boundary_resolution_ms: float | None = None,
    audio: Any | None = None,
    sample_rate: int = 16000,
    min_internal_silence: float = 0.04,
    min_split_bin_duration: float = 2.5,
) -> list[PhysicalSubtitleBin]:
    """Build speech containers from the most precise available evidence source.

    Macro ``PhysicalClip`` objects describe ownership/context and are not used
    as subtitle containers. A single preferred evidence source is selected so
    coarse VAD output cannot swallow the more precise fused boundaries.
    """
    if not isinstance(timeline, PhysicalTimeline):
        raise ValueError("timeline must be a PhysicalTimeline")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if min_internal_silence < 0:
        raise ValueError("min_internal_silence must be non-negative")
    if min_split_bin_duration < 0:
        raise ValueError("min_split_bin_duration must be non-negative")

    evidence = list(timeline.speech_evidence_spans)
    if not evidence:
        return []
    source_rank = {source: index for index, source in enumerate(_SOURCE_PRIORITY)}
    selected_source = min(
        evidence,
        key=lambda item: source_rank.get(item.source, len(source_rank)),
    ).source
    selected = [item for item in evidence if item.source == selected_source]
    selected.sort(key=lambda item: (item.start, item.end, item.id))

    bins: list[PhysicalSubtitleBin] = []
    current: list[SpeechEvidenceSpan] = []
    for item in selected:
        if current:
            previous = current[-1]
            same_clip = previous.physical_clip_id == item.physical_clip_id
            if not same_clip or item.start - previous.end > merge_gap:
                bins.extend(
                    _make_bins_with_energy_gaps(
                        len(bins) + 1,
                        current,
                        boundary_resolution_ms,
                        audio=audio,
                        sample_rate=sample_rate,
                        min_internal_silence=min_internal_silence,
                        min_split_bin_duration=min_split_bin_duration,
                    )
                )
                current = []
        current.append(item)
    if current:
        bins.extend(
            _make_bins_with_energy_gaps(
                len(bins) + 1,
                current,
                boundary_resolution_ms,
                audio=audio,
                sample_rate=sample_rate,
                min_internal_silence=min_internal_silence,
                min_split_bin_duration=min_split_bin_duration,
            )
        )
    return _renumber_bins(bins)


def assign_word_to_bin(word: Any, bins: Sequence[PhysicalSubtitleBin]) -> PhysicalSubtitleBin | None:
    """Return the bin with the largest positive overlap for a global word."""
    raw_start = float(getattr(word, "raw_start"))
    raw_end = float(getattr(word, "raw_end"))
    candidates = []
    midpoint = (raw_start + raw_end) / 2.0
    for item in bins:
        overlap = max(0.0, min(raw_end, item.end) - max(raw_start, item.start))
        if overlap > 0:
            distance = abs(midpoint - ((item.start + item.end) / 2.0))
            candidates.append((overlap, -distance, item))
    if not candidates:
        return None
    return max(candidates, key=lambda value: (value[0], value[1], value[2].id))[2]


def _make_bin(
    index: int,
    evidence: Sequence[SpeechEvidenceSpan],
    boundary_resolution_ms: float | None,
    *,
    start: float | None = None,
    end: float | None = None,
    source_suffix: str | None = None,
) -> PhysicalSubtitleBin:
    clip_ids = {item.physical_clip_id for item in evidence}
    confidences = [item.confidence for item in evidence if item.confidence is not None]
    sources = list(dict.fromkeys(item.source for item in evidence))
    return PhysicalSubtitleBin(
        id=f"subtitle-bin-{index:06d}",
        start=min(item.start for item in evidence) if start is None else start,
        end=max(item.end for item in evidence) if end is None else end,
        source="+".join(sources + ([source_suffix] if source_suffix else [])),
        confidence=max(confidences) if confidences else None,
        evidence_ids=tuple(item.id for item in evidence),
        physical_clip_id=next(iter(clip_ids)) if len(clip_ids) == 1 else None,
        boundary_resolution_ms=boundary_resolution_ms,
    )


def _make_bins_with_energy_gaps(
    index: int,
    evidence: Sequence[SpeechEvidenceSpan],
    boundary_resolution_ms: float | None,
    *,
    audio: Any | None,
    sample_rate: int,
    min_internal_silence: float,
    min_split_bin_duration: float,
) -> list[PhysicalSubtitleBin]:
    """Split long evidence runs at measured low-energy valleys.

    FFmpeg skeletons reliably capture long pauses but can keep adjacent
    speakers in one speech run. A short, deep energy valley inside a long run
    is therefore an additional physical boundary candidate. The split is
    conservative: it is disabled without audio and only applies to runs long
    enough to plausibly contain multiple subtitle events.
    """
    base_start = min(item.start for item in evidence)
    base_end = max(item.end for item in evidence)
    if audio is None or base_end - base_start < min_split_bin_duration:
        return [_make_bin(index, evidence, boundary_resolution_ms)]

    gaps = _find_energy_gaps(
        audio,
        sample_rate,
        base_start,
        base_end,
        min_duration=min_internal_silence,
    )
    if not gaps:
        return [_make_bin(index, evidence, boundary_resolution_ms)]
    # A long physical run can contain ordinary intra-sentence pauses. Keep
    # only its deepest valley as the conservative hand-off candidate; the
    # later word-safe segmenter handles display-length limits without moving
    # the rest of the timeline through a chain of speculative cuts.
    gaps = sorted(gaps, key=lambda item: item[2])[:1]
    gaps = sorted(gaps, key=lambda item: item[0])

    ranges: list[tuple[float, float]] = []
    cursor = base_start
    for gap_start, gap_end, _ in gaps:
        if gap_start - cursor > 0.02:
            ranges.append((cursor, gap_start))
        cursor = gap_end
    if base_end - cursor > 0.02:
        ranges.append((cursor, base_end))
    if len(ranges) < 2:
        return [_make_bin(index, evidence, boundary_resolution_ms)]

    return [
        _make_bin(
            index + offset,
            evidence,
            boundary_resolution_ms,
            start=start,
            end=end,
            source_suffix="energy_valley",
        )
        for offset, (start, end) in enumerate(ranges)
    ]


def _find_energy_gaps(
    audio: Any,
    sample_rate: int,
    start: float,
    end: float,
    *,
    min_duration: float,
) -> list[tuple[float, float, float]]:
    """Return internal low-energy runs at the audio frame resolution."""
    try:
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    except (ImportError, TypeError, ValueError):
        return []
    if samples.size == 0:
        return []

    frame_size = max(1, int(round(sample_rate * 0.01)))
    frame_count = samples.size // frame_size
    if frame_count < 1:
        return []
    framed = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(framed), axis=1))
    db = 20.0 * np.log10(rms + 1e-6)
    noise_floor = float(np.percentile(db, 10))
    # Cap the adaptive threshold: a short silence may be below the percentile
    # used for the noise estimate, especially in synthetic or very clean
    # recordings where most frames are full-scale speech.
    threshold_db = max(-45.0, min(noise_floor + 10.0, -30.0))
    low = db <= threshold_db
    first = max(0, int(start * sample_rate / frame_size))
    last = min(frame_count, int(np.ceil(end * sample_rate / frame_size)))
    min_frames = max(1, int(np.ceil(min_duration / 0.01)))
    gaps: list[tuple[float, float, float]] = []
    run_start: int | None = None
    for index in range(first, last + 1):
        is_low = index < last and bool(low[index])
        if is_low and run_start is None:
            run_start = index
        if not is_low and run_start is not None:
            run_end = index
            run_frames = run_end - run_start
            valley_db = float(np.min(db[run_start:run_end]))
            # Very deep valleys can be shorter than a normal pause, which is
            # common at an immediate speaker hand-off.
            deep_valley = valley_db <= threshold_db - 12.0
            if run_frames >= min_frames or (deep_valley and run_frames >= 4):
                gap_start = max(start, run_start * frame_size / sample_rate)
                gap_end = min(end, run_end * frame_size / sample_rate)
                if (
                    gap_start - start >= 0.12
                    and end - gap_end >= 0.12
                    and gap_end > gap_start
                ):
                    gaps.append((gap_start, gap_end, valley_db))
            run_start = None
    return gaps


def _renumber_bins(bins: Sequence[PhysicalSubtitleBin]) -> list[PhysicalSubtitleBin]:
    return [
        PhysicalSubtitleBin(
            id=f"subtitle-bin-{index:06d}",
            start=item.start,
            end=item.end,
            source=item.source,
            confidence=item.confidence,
            evidence_ids=item.evidence_ids,
            physical_clip_id=item.physical_clip_id,
            boundary_resolution_ms=item.boundary_resolution_ms,
        )
        for index, item in enumerate(bins, start=1)
    ]
