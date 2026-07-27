"""Tests for adaptive pause classification and PhysicalFragment segmentation."""

import pytest

from vocal_subtitle.physical.allocator import PhysicalSpan, WordAllocation
from vocal_subtitle.physical.ir import GlobalWord
from vocal_subtitle.mapping.semantic_fragments import (
    AdaptivePauseThresholds,
    PhysicalFragment,
    _estimate_gap_distribution,
    build_physical_fragments,
    calibrate_speaker_thresholds,
)


def _make_word(
    word_id: str,
    text: str,
    raw_start: float,
    raw_end: float,
    confidence: float = 0.95,
    speaker_id: int | None = None,
) -> GlobalWord:
    return GlobalWord(
        id=word_id,
        text=text,
        raw_start=raw_start,
        raw_end=raw_end,
        confidence=confidence,
        source_window_id="win-1",
        segment_id="seg-1",
        speaker_id=speaker_id,
    )


def _make_alloc(
    word: GlobalWord,
    aligned_start: float | None = None,
    aligned_end: float | None = None,
    speaker_id: int | None = None,
    speaker_source: str = "unknown",
    warnings: tuple = (),
) -> WordAllocation:
    ws = aligned_start if aligned_start is not None else word.raw_start
    we = aligned_end if aligned_end is not None else word.raw_end
    alloc = WordAllocation(
        word=word,
        physical_spans=(
            PhysicalSpan(clip_id="clip-1", start=word.raw_start, end=word.raw_end),
        ),
        evidence_ids=("e1",),
        speaker_id=speaker_id,
        speaker_source=speaker_source,
        warnings=warnings,
        accepted=True,
    )
    object.__setattr__(alloc, "aligned_start", ws)
    object.__setattr__(alloc, "aligned_end", we)
    object.__setattr__(alloc, "alignment_status", "aligned")
    return alloc


class TestGapDistribution:
    def test_estimates_from_typical_gaps(self):
        import numpy as np
        gaps = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.25, 0.35, 0.45, 0.55, 0.70])
        micro, sentence, long_ = _estimate_gap_distribution(gaps)

        assert 0.100 <= micro <= 0.250
        assert 0.200 <= sentence <= 0.500
        assert 0.400 <= long_ <= 0.800
        assert micro < sentence < long_

    def test_few_gaps_returns_defaults(self):
        import numpy as np
        micro, sentence, long_ = _estimate_gap_distribution(np.array([0.1]))

        assert micro == 0.150
        assert sentence == 0.350
        assert long_ == 0.600


class TestCalibrateThresholds:
    def test_calibrates_from_aligned_words(self):
        words = []
        for i in range(10):
            t = i * 0.5
            word = _make_word(f"w{i}", str(i), t, t + 0.4)
            words.append(_make_alloc(word, aligned_start=t, aligned_end=t + 0.4))

        thresholds = calibrate_speaker_thresholds(words)

        # Gaps are ~0.1s between words, so micro ≈ 0.1, sentence ≈ 0.1+0.05, long ≈ 0.1+0.1
        assert 0.100 <= thresholds.micro_pause_max <= 0.250
        assert thresholds.micro_pause_max < thresholds.sentence_pause_max
        assert thresholds.sentence_pause_max < thresholds.long_pause_min
        assert thresholds.calibration_quality == "calibrated"

    def test_insufficient_data_is_fallback(self):
        word = _make_word("w1", "solo", 0.0, 0.5)
        alloc = _make_alloc(word)

        thresholds = calibrate_speaker_thresholds([alloc], min_samples=5)

        assert thresholds.calibration_quality == "fallback"


class TestBuildFragments:
    def test_single_word_yields_one_fragment(self):
        word = _make_word("w1", "hello", 0.0, 0.5)
        alloc = _make_alloc(word)

        fragments = build_physical_fragments([alloc])

        assert len(fragments) == 1
        assert fragments[0].word_ids == ["w1"]
        assert fragments[0].physical_start == 0.0
        assert fragments[0].physical_end == 0.5

    def test_long_pause_creates_hard_split(self):
        w1 = _make_word("w1", "first", 0.0, 0.5)
        w2 = _make_word("w2", "second", 5.0, 5.5)
        a1 = _make_alloc(w1)
        a2 = _make_alloc(w2)

        fragments = build_physical_fragments([a1, a2])

        # Long pause between 0.5 and 5.0 should split
        assert len(fragments) == 2
        assert fragments[0].word_ids == ["w1"]
        assert fragments[1].word_ids == ["w2"]
        assert fragments[1].hard_split_before is True

    def test_close_words_share_fragment(self):
        w1 = _make_word("w1", "hello", 0.0, 0.5)
        w2 = _make_word("w2", "world", 0.55, 1.0)
        a1 = _make_alloc(w1, aligned_start=0.0, aligned_end=0.5)
        a2 = _make_alloc(w2, aligned_start=0.55, aligned_end=1.0)

        fragments = build_physical_fragments([a1, a2])

        assert len(fragments) == 1
        assert fragments[0].word_ids == ["w1", "w2"]
        assert fragments[0].physical_start == 0.0
        assert fragments[0].physical_end == 1.0

    def test_speaker_change_splits_when_confirmed(self):
        w1 = _make_word("w1", "a", 0.0, 0.5, speaker_id=0)
        w2 = _make_word("w2", "b", 0.55, 1.0, speaker_id=1)
        a1 = _make_alloc(w1, speaker_id=0, speaker_source="diarization")
        a2 = _make_alloc(w2, speaker_id=1, speaker_source="diarization")

        fragments = build_physical_fragments([a1, a2])

        assert len(fragments) == 2
        assert fragments[1].hard_split_before is True

    def test_empty_input(self):
        assert build_physical_fragments([]) == []

    def test_fragment_serialization(self):
        frag = PhysicalFragment(
            id="frag-0001",
            word_ids=["w1", "w2"],
            physical_start=0.0,
            physical_end=1.0,
            candidate_speaker=0,
            pause_class="sentence",
            hard_split_before=False,
        )

        payload = frag.to_dict()

        assert payload["id"] == "frag-0001"
        assert payload["word_ids"] == ["w1", "w2"]
        assert payload["physical_start"] == 0.0
        assert payload["physical_end"] == 1.0
