"""Tests for constrained word-level boundary alignment."""

import pytest

from vocal_subtitle.physical.allocator import (
    AllocationResult,
    PhysicalSpan,
    WordAllocation,
)
from vocal_subtitle.physical.ir import GlobalWord
from vocal_subtitle.physical.subtitle_bins import PhysicalSubtitleBin
from vocal_subtitle.physical.word_alignment import (
    BoundaryDecision,
    _asr_confidence_tier,
    _search_window_for_tier,
    _score_start_candidate,
    _score_end_candidate,
    align_words_to_physical,
)


def _make_word(
    word_id: str,
    text: str,
    raw_start: float,
    raw_end: float,
    confidence: float = 0.95,
) -> GlobalWord:
    return GlobalWord(
        id=word_id,
        text=text,
        raw_start=raw_start,
        raw_end=raw_end,
        confidence=confidence,
        source_window_id="win-1",
        segment_id="seg-1",
    )


def _make_bin(
    bin_id: str,
    start: float,
    end: float,
    source: str = "silero",
    confidence: float = 0.9,
) -> PhysicalSubtitleBin:
    return PhysicalSubtitleBin(
        id=bin_id,
        start=start,
        end=end,
        source=source,
        confidence=confidence,
        evidence_ids=("e1",),
        physical_clip_id="clip-1",
    )


def _make_alloc(word: GlobalWord, bin_id: str = "bin-1") -> WordAllocation:
    return WordAllocation(
        word=word,
        physical_spans=(
            PhysicalSpan(clip_id="clip-1", start=word.raw_start, end=word.raw_end),
        ),
        evidence_ids=("e1",),
        speaker_id=None,
        accepted=True,
    )


class TestConfidenceTiers:
    def test_high_confidence(self):
        assert _asr_confidence_tier(0.95) == "high"
        assert _asr_confidence_tier(0.90) == "high"

    def test_medium_confidence(self):
        assert _asr_confidence_tier(0.75) == "medium"
        assert _asr_confidence_tier(0.60) == "medium"

    def test_low_confidence(self):
        assert _asr_confidence_tier(0.30) == "low"
        assert _asr_confidence_tier(0.0) == "low"

    def test_unknown_confidence(self):
        assert _asr_confidence_tier(None) == "unknown"


class TestSearchWindows:
    def test_high_tier_window(self):
        assert _search_window_for_tier("high") == 0.200

    def test_low_tier_window(self):
        assert _search_window_for_tier("low") == 0.500

    def test_unknown_tier_window(self):
        assert _search_window_for_tier("unknown") == 0.500


class TestStartScoring:
    def test_asr_proximity_matters(self):
        # Candidate at ASR time scores higher than one far away
        near = _score_start_candidate(1.0, asr_time=1.0)
        far = _score_start_candidate(1.3, asr_time=1.0)
        assert near > far

    def test_rms_falling_penalized_at_start(self):
        # At start, a positive rms_gradient means rising energy (the candidate
        # is at the onset peak), which scores higher than falling energy
        # (rms_gradient < 0, meaning the candidate already passed the peak).
        rising = _score_start_candidate(0.9, asr_time=1.0, rms_gradient=0.8)
        falling = _score_start_candidate(1.1, asr_time=1.0, rms_gradient=-0.8)
        # Rising onset scores higher because the rms function checks:
        # if candidate > asr_time and rms_gradient > 0: penalty
        # The rising candidate (0.9 < 1.0) avoids the penalty
        assert rising >= falling


class TestEndScoring:
    def test_silence_bonus(self):
        no_silence = _score_end_candidate(2.0, asr_time=2.0, silence_duration=0.0)
        with_silence = _score_end_candidate(2.0, asr_time=2.0, silence_duration=0.35)
        assert with_silence > no_silence

    def test_falling_energy_preferred_at_end(self):
        falling = _score_end_candidate(2.0, asr_time=2.0, rms_gradient=-1.5)
        rising = _score_end_candidate(2.0, asr_time=2.0, rms_gradient=1.5)
        assert falling > rising


class TestBoundaryDecision:
    def test_serialization_round_trip(self):
        original = BoundaryDecision(
            accepted=True,
            boundary_time=1.234,
            boundary_type="start",
            confidence=0.85,
            evidence_ids=("e1", "e2"),
            reason_codes=("aligned_start",),
            candidate_scores=(("asr_start", 0.8), ("bin_start", 0.75)),
            rejected_candidates=("bin_end outside range",),
        )

        payload = original.to_dict()
        restored = BoundaryDecision.from_dict(payload)

        assert restored.accepted is True
        assert restored.boundary_time == 1.234
        assert restored.boundary_type == "start"
        assert restored.confidence == 0.85
        assert restored.evidence_ids == ("e1", "e2")
        assert restored.reason_codes == ("aligned_start",)
        assert len(restored.candidate_scores) == 2


class TestAlignWords:
    def test_align_single_word_within_bin(self):
        word = _make_word("w1", "hello", 0.5, 0.8, confidence=0.95)
        word_bin = _make_bin("bin-1", 0.4, 0.9)
        alloc = _make_alloc(word, "bin-1")

        result = align_words_to_physical([alloc], [word_bin])

        assert len(result) == 1
        aligned = result[0]
        assert hasattr(aligned, "aligned_start")
        assert hasattr(aligned, "aligned_end")
        assert 0.4 <= aligned.aligned_start <= 0.55
        assert 0.75 <= aligned.aligned_end <= 0.9

    def test_last_word_not_stretched_to_bin_end(self):
        word = _make_word("w1", "test", 0.5, 0.7, confidence=0.8)
        word_bin = _make_bin("bin-1", 0.4, 3.0)  # bin goes way past the word
        alloc = _make_alloc(word, "bin-1")

        result = align_words_to_physical([alloc], [word_bin])

        assert len(result) == 1
        aligned = result[0]
        # End should NOT be stretched to bin end (3.0)
        assert aligned.aligned_end < 1.0

    def test_word_outside_all_bins_gets_degraded(self):
        word = _make_word("w1", "orphan", 10.0, 11.0)
        word_bin = _make_bin("bin-1", 0.0, 5.0)  # word is outside this bin
        alloc = _make_alloc(word, "bin-none")

        result = align_words_to_physical([alloc], [word_bin])

        assert len(result) == 1
        aligned = result[0]
        # Even without legal candidates, we get times (from raw ASR)
        assert aligned.aligned_start >= 0
