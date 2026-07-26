from copy import deepcopy

from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
from vocal_subtitle.asr.hallucination import (
    HallucinationFilterPolicy,
    filter_transcription_segments,
)


def _segment(text, **kwargs):
    return TranscriptionSegment(
        text=text,
        start=kwargs.pop("start", 0.0),
        end=kwargs.pop("end", 1.0),
        words=kwargs.pop("words", []),
        **kwargs,
    )


def _word(text="word", confidence=0.9):
    return WordTimestamp(text, 0.1, 0.4, confidence=confidence)


def test_filters_empty_and_training_phrases():
    result = filter_transcription_segments(
        [_segment("  "), _segment("感谢观看。"), _segment("Thanks for watching!")],
        HallucinationFilterPolicy(),
    )

    assert result.segments == []
    assert result.counts["empty_text"] == 1
    assert result.counts["training_phrase"] == 2


def test_quality_thresholds_require_missing_word_evidence():
    result = filter_transcription_segments(
        [
            _segment("silent", no_speech_prob=0.9),
            _segment("weak", avg_logprob=-2.0),
            _segment("real", words=[_word()], no_speech_prob=0.9, avg_logprob=-2.0),
        ],
        HallucinationFilterPolicy(),
    )

    assert [item.text for item in result.segments] == ["real"]
    assert result.counts["high_no_speech_without_word_evidence"] == 1
    assert result.counts["low_logprob_without_word_evidence"] == 1
    assert result.counts["high_no_speech_with_word_evidence"] == 1
    assert result.counts["low_logprob_with_word_evidence"] == 1


def test_repetitive_compression_is_filtered_but_high_compression_alone_warns():
    result = filter_transcription_segments(
        [
            _segment("哈哈哈哈", compression_ratio=3.0),
            _segment("unusual output", compression_ratio=3.0),
        ],
        HallucinationFilterPolicy(),
    )

    assert [item.text for item in result.segments] == ["unusual output"]
    assert result.counts["repetitive_compression"] == 1
    assert result.counts["high_compression_without_repetition"] == 1


def test_valid_short_words_are_preserved():
    result = filter_transcription_segments(
        [
            _segment("Good", end=0.1, words=[_word("Good")], no_speech_prob=0.95),
            _segment("我", end=0.1, words=[_word("我")], avg_logprob=-3.0),
            _segment("唉", end=0.1, words=[_word("唉")]),
        ],
        HallucinationFilterPolicy(),
    )

    assert [item.text for item in result.segments] == ["Good", "我", "唉"]


def test_adjacent_duplicate_keeps_more_complete_result():
    result = filter_transcription_segments(
        [
            _segment("hello", start=0.0, end=1.0),
            _segment("hello world", start=1.05, end=1.8),
        ],
        HallucinationFilterPolicy(),
    )

    assert [item.text for item in result.segments] == ["hello world"]
    assert result.counts["adjacent_duplicate"] == 1


def test_filter_does_not_mutate_input():
    original = [_segment("真实", words=[_word("真实")])]
    snapshot = deepcopy(original)

    filter_transcription_segments(original, HallucinationFilterPolicy())

    assert original == snapshot


def test_disabled_filter_keeps_all_results():
    segments = [_segment("感谢观看"), _segment("  ")]

    result = filter_transcription_segments(
        segments,
        HallucinationFilterPolicy(enabled=False),
    )

    assert result.segments == segments
    assert result.counts == {"disabled": 1}
