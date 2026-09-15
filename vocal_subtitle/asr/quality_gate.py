"""Physical-evidence quality checks for ASR subtitle candidates."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ASRQualityResult:
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": dict(self.metrics),
            "reasons": list(self.reasons),
        }


def _merge_intervals(intervals: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    result: list[tuple[float, float]] = []
    for start, end in ordered:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _intersection_duration(
    intervals: Sequence[tuple[float, float]],
    start: float,
    end: float,
) -> float:
    return sum(
        max(0.0, min(end, right) - max(start, left)) for left, right in intervals
    )


def evaluate_asr_quality(
    events: Iterable[Any],
    speech_intervals: Iterable[Sequence[float]],
    audio_duration: float,
    *,
    min_coverage_ratio: float = 0.80,
    min_text_density: float = 0.20,
    max_event_duration: float = 12.0,
    long_audio_seconds: float = 60.0,
    long_audio_min_text_chars: int = 12,
    max_overlap_ratio: float = 0.35,
) -> ASRQualityResult:
    """Evaluate event coverage using physical speech time, not file duration."""
    physical = _merge_intervals(speech_intervals)
    candidates = list(events)
    physical_duration = sum(end - start for start, end in physical)
    valid = []
    invalid = 0
    for event in candidates:
        text = str(getattr(event, "text", "") or "").strip()
        start = float(
            getattr(event, "physical_start", None)
            or getattr(event, "start", None)
            or 0.0
        )
        end = float(
            getattr(event, "physical_end", None) or getattr(event, "end", None) or 0.0
        )
        if not text or end <= start or end < 0 or start > float(audio_duration):
            invalid += 1
            continue
        valid.append((max(0.0, start), min(float(audio_duration), end), text))

    covered = _merge_intervals((start, end) for start, end, _ in valid)
    covered_physical = sum(
        _intersection_duration(physical, start, end) for start, end in covered
    )
    coverage = covered_physical / physical_duration if physical_duration else 0.0
    text_chars = sum(len(text) for _, _, text in valid)
    density = text_chars / covered_physical if covered_physical > 0 else 0.0
    max_duration = max((end - start for start, end, _ in valid), default=0.0)
    long_events = sum(
        1 for start, end, _ in valid if end - start > float(max_event_duration)
    )

    overlap_duration = 0.0
    for index, (start, end, _) in enumerate(valid):
        for other_start, other_end, _ in valid[index + 1 :]:
            overlap_duration += max(0.0, min(end, other_end) - max(start, other_start))
    total_event_duration = sum(end - start for start, end, _ in valid)
    overlap_ratio = (
        overlap_duration / total_event_duration if total_event_duration else 0.0
    )

    metrics = {
        "coverage_ratio": round(coverage, 6),
        "text_density": round(density, 6),
        "max_event_duration": round(max_duration, 6),
        "long_event_count": long_events,
        "overlap_ratio": round(overlap_ratio, 6),
        "invalid_event_count": invalid,
        "event_count": len(candidates),
        "valid_event_count": len(valid),
        "physical_speech_duration": round(physical_duration, 6),
        "text_char_count": text_chars,
    }
    reasons = []
    if not valid:
        reasons.append("empty_or_invalid_events")
    if invalid:
        reasons.append("invalid_events")
    if physical_duration and coverage < float(min_coverage_ratio):
        reasons.append("low_physical_coverage")
    if physical_duration and density < float(min_text_density):
        reasons.append("low_text_density")
    if overlap_ratio > float(max_overlap_ratio):
        reasons.append("excessive_overlap")
    if long_events and float(audio_duration) >= float(long_audio_seconds):
        reasons.append("abnormally_long_events")
    if float(audio_duration) >= float(long_audio_seconds) and text_chars < int(
        long_audio_min_text_chars
    ):
        reasons.append("long_audio_low_text_amount")

    status = "pass"
    if reasons:
        status = "failed"
    elif valid and (coverage < 0.95 or density < max(float(min_text_density) * 2, 0.5)):
        status = "warning"
    return ASRQualityResult(status=status, metrics=metrics, reasons=tuple(reasons))
