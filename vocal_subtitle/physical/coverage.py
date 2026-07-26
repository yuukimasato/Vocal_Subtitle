"""Physical speech-bin coverage auditing for global subtitle generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .allocator import WordAllocation
from .subtitle_bins import PhysicalSubtitleBin


@dataclass(frozen=True)
class PhysicalCoverageRange:
    """A contiguous group of physical bins without an accepted ASR word."""

    start: float
    end: float
    bin_ids: tuple[str, ...]
    physical_clip_id: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "bin_ids": list(self.bin_ids),
            "physical_clip_id": self.physical_clip_id,
        }


@dataclass(frozen=True)
class PhysicalCoverageReport:
    """Coverage facts used by recovery, validation, and the WebUI."""

    physical_bin_count: int
    covered_physical_bin_count: int
    uncovered_physical_bin_count: int
    uncovered_physical_bins: tuple[dict[str, Any], ...]
    recovery_ranges: tuple[PhysicalCoverageRange, ...]
    transcript_end: float | None
    last_physical_speech_end: float | None
    tail_gap_seconds: float

    @property
    def complete(self) -> bool:
        return self.uncovered_physical_bin_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_bin_count": self.physical_bin_count,
            "covered_physical_bin_count": self.covered_physical_bin_count,
            "uncovered_physical_bin_count": self.uncovered_physical_bin_count,
            "uncovered_physical_bins": list(self.uncovered_physical_bins),
            "recovery_ranges": [item.to_dict() for item in self.recovery_ranges],
            "transcript_end": self.transcript_end,
            "last_physical_speech_end": self.last_physical_speech_end,
            "tail_gap_seconds": self.tail_gap_seconds,
            "complete": self.complete,
        }


def audit_physical_coverage(
    bins: Sequence[PhysicalSubtitleBin],
    allocations: Iterable[WordAllocation],
    *,
    min_required_duration: float = 0.08,
    merge_gap: float = 1.0,
) -> PhysicalCoverageReport:
    """Report physical bins that have no accepted word overlap.

    ``missing_word_ids`` cannot detect audio that ASR never returned. This
    audit starts from physical speech bins and therefore catches that case.
    Very short bins are still reported but are excluded from recovery ranges
    because they are commonly boundary noise or punctuation-sized fragments.
    """
    if min_required_duration < 0:
        raise ValueError("min_required_duration must be non-negative")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")

    ordered_bins = sorted(
        list(bins), key=lambda item: (float(item.start), float(item.end), item.id)
    )
    accepted_words = [
        item.word
        for item in allocations
        if getattr(item, "accepted", False)
    ]
    transcript_end = max(
        (float(getattr(word, "raw_end")) for word in accepted_words),
        default=None,
    )

    uncovered: list[PhysicalSubtitleBin] = []
    covered_count = 0
    for bin_item in ordered_bins:
        covered = any(
            float(getattr(word, "raw_start")) < float(bin_item.end)
            and float(getattr(word, "raw_end")) > float(bin_item.start)
            for word in accepted_words
        )
        if covered:
            covered_count += 1
        else:
            uncovered.append(bin_item)

    last_physical_end = max(
        (float(item.end) for item in ordered_bins), default=None
    )
    tail_gap = max(
        0.0,
        (last_physical_end or 0.0) - (transcript_end or 0.0),
    )

    uncovered_payload = tuple(
        _bin_payload(item) for item in uncovered
    )
    recovery_bins = [
        item
        for item in uncovered
        if float(item.end) - float(item.start) >= min_required_duration
    ]
    recovery_ranges = tuple(
        _merge_ranges(recovery_bins, merge_gap=merge_gap)
    )

    return PhysicalCoverageReport(
        physical_bin_count=len(ordered_bins),
        covered_physical_bin_count=covered_count,
        uncovered_physical_bin_count=len(uncovered),
        uncovered_physical_bins=uncovered_payload,
        recovery_ranges=recovery_ranges,
        transcript_end=transcript_end,
        last_physical_speech_end=last_physical_end,
        tail_gap_seconds=tail_gap,
    )


def _merge_ranges(
    bins: Sequence[PhysicalSubtitleBin],
    *,
    merge_gap: float,
) -> list[PhysicalCoverageRange]:
    ranges: list[PhysicalCoverageRange] = []
    for bin_item in sorted(
        bins, key=lambda item: (float(item.start), float(item.end), item.id)
    ):
        if not ranges:
            ranges.append(
                PhysicalCoverageRange(
                    start=float(bin_item.start),
                    end=float(bin_item.end),
                    bin_ids=(bin_item.id,),
                    physical_clip_id=bin_item.physical_clip_id,
                )
            )
            continue

        previous = ranges[-1]
        same_clip = previous.physical_clip_id == bin_item.physical_clip_id
        if same_clip and float(bin_item.start) - previous.end <= merge_gap:
            ranges[-1] = PhysicalCoverageRange(
                start=previous.start,
                end=max(previous.end, float(bin_item.end)),
                bin_ids=(*previous.bin_ids, bin_item.id),
                physical_clip_id=previous.physical_clip_id,
            )
            continue

        ranges.append(
            PhysicalCoverageRange(
                start=float(bin_item.start),
                end=float(bin_item.end),
                bin_ids=(bin_item.id,),
                physical_clip_id=bin_item.physical_clip_id,
            )
        )
    return ranges


def _bin_payload(bin_item: PhysicalSubtitleBin) -> dict[str, Any]:
    return {
        "id": bin_item.id,
        "start": float(bin_item.start),
        "end": float(bin_item.end),
        "duration": max(0.0, float(bin_item.end) - float(bin_item.start)),
        "source": bin_item.source,
        "physical_clip_id": bin_item.physical_clip_id,
    }
