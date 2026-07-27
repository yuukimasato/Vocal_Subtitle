"""Tests for local ASR recovery engine."""

import numpy as np
import pytest

from vocal_subtitle.asr.base import ASREngine, TranscriptionSegment, WordTimestamp
from vocal_subtitle.asr.local_recovery import (
    LocalRecoveryConfig,
    LocalRecoveryRequest,
    LocalRecoveryResult,
    LocalRecoveryEngine,
    RecoveryCandidate,
)
from vocal_subtitle.physical.ir import GlobalWord


class _StubASR(ASREngine):
    """ASR engine that returns configurable per-call results."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
        self._call_index = 0

    @property
    def name(self):
        return "stub-recovery-asr"

    @property
    def model_name(self):
        return "stub-recovery-asr"

    def load_model(self):
        return None

    def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
        self.calls.append({"audio_len": len(audio), "sample_rate": sample_rate})
        if self._call_index < len(self.responses):
            result = self.responses[self._call_index]
            self._call_index += 1
            return result
        return []


# ── RecoveryCandidate ────────────────────────────────────────────────

def test_recovery_candidate_requires_non_empty_word_id():
    with pytest.raises(ValueError, match="word_id"):
        RecoveryCandidate(word_id="", text="hello", start=1.0, end=1.3, confidence=0.9)


def test_recovery_candidate_rejects_inverted_time():
    with pytest.raises(ValueError, match="start.*end"):
        RecoveryCandidate(word_id="w1", text="hello", start=1.3, end=1.0, confidence=0.9)


def test_recovery_candidate_rejects_negative_time():
    with pytest.raises(ValueError):
        RecoveryCandidate(word_id="w1", text="hello", start=-0.1, end=1.0, confidence=0.9)


def test_recovery_candidate_accepts_default_confidence():
    candidate = RecoveryCandidate(word_id="w1", text="hello", start=0.5, end=0.8)
    assert candidate.confidence == 0.0


def test_recovery_candidate_clamps_confidence():
    candidate = RecoveryCandidate(word_id="w1", text="hello", start=0.5, end=0.8, confidence=1.5)
    assert candidate.confidence == 1.0
    candidate2 = RecoveryCandidate(word_id="w2", text="hi", start=0.5, end=0.8, confidence=-0.3)
    assert candidate2.confidence == 0.0


def test_recovery_candidate_to_dict():
    candidate = RecoveryCandidate(word_id="w1", text="hello", start=0.5, end=0.8, confidence=0.85)
    d = candidate.to_dict()
    assert d["word_id"] == "w1"
    assert d["text"] == "hello"
    assert d["start"] == 0.5
    assert d["end"] == 0.8
    assert d["confidence"] == 0.85


# ── LocalRecoveryConfig ──────────────────────────────────────────────

def test_config_defaults_are_valid():
    cfg = LocalRecoveryConfig()
    assert cfg.max_attempts_per_range >= 1
    assert cfg.min_confidence >= 0.0
    assert cfg.context_window >= 0.0
    assert cfg.max_attempts_per_range <= 5


def test_config_rejects_zero_max_attempts():
    with pytest.raises(ValueError):
        LocalRecoveryConfig(max_attempts_per_range=0)


def test_config_rejects_negative_min_confidence():
    with pytest.raises(ValueError):
        LocalRecoveryConfig(min_confidence=-0.1)


# ── LocalRecoveryRequest ─────────────────────────────────────────────

def test_request_requires_at_least_one_reason():
    with pytest.raises(ValueError, match="reason"):
        LocalRecoveryRequest(start=0.0, end=1.0, reasons=[])


def test_request_normalizes_reasons():
    req = LocalRecoveryRequest(start=0.5, end=1.5, reasons=[" low_conf ", " LOW_CONF "])
    assert req.reasons == ("low_conf",)


def test_request_requires_positive_range():
    with pytest.raises(ValueError, match="start.*end"):
        LocalRecoveryRequest(start=1.5, end=0.5, reasons=["uncovered"])


# ── LocalRecoveryResult ──────────────────────────────────────────────

def test_result_marks_outcome():
    """An outcome of 'recovered' with candidates is success."""
    result = LocalRecoveryResult(
        request=LocalRecoveryRequest(start=0.0, end=1.0, reasons=["uncovered"]),
        candidates=[RecoveryCandidate(word_id="w1", text="hello", start=0.2, end=0.5, confidence=0.9)],
        attempt_count=1,
        outcome="recovered",
    )
    assert result.success is True
    assert result.outcome == "recovered"


def test_result_failed_has_no_candidates():
    result = LocalRecoveryResult(
        request=LocalRecoveryRequest(start=0.0, end=1.0, reasons=["uncovered"]),
        candidates=[],
        attempt_count=2,
        outcome="max_attempts_exceeded",
    )
    assert result.success is False
    assert len(result.candidates) == 0


def test_result_requires_valid_outcome():
    with pytest.raises(ValueError, match="outcome"):
        LocalRecoveryResult(
            request=LocalRecoveryRequest(start=0.0, end=1.0, reasons=["uncovered"]),
            candidates=[],
            attempt_count=1,
            outcome="invalid",
        )


def test_result_requires_positive_attempt_count():
    with pytest.raises(ValueError):
        LocalRecoveryResult(
            request=LocalRecoveryRequest(start=0.0, end=1.0, reasons=["uncovered"]),
            candidates=[],
            attempt_count=0,
        )


# ── LocalRecoveryEngine ──────────────────────────────────────────────

def test_engine_processes_uncovered_range():
    """ASR returns words within the request range — they should be accepted."""
    # Segment-local word at 1.0-1.3s, request is [0.5, 1.5], context_start=0.0.
    # Global word position: [1.0, 1.3] which overlaps with [0.5, 1.5].
    engine_stub = _StubASR(responses=[
        [
            TranscriptionSegment(
                text="recovered text",
                start=1.0,
                end=1.3,
                words=[
                    WordTimestamp("recovered", 1.0, 1.15, confidence=0.9),
                    WordTimestamp("text", 1.15, 1.3, confidence=0.9),
                ],
            )
        ]
    ])
    engine = LocalRecoveryEngine(asr_engine=engine_stub)
    audio = np.zeros(48000, dtype=np.float32)  # 3s @ 16kHz

    requests = [
        LocalRecoveryRequest(start=0.5, end=1.5, reasons=["uncovered"]),
    ]
    results = engine.process_requests(requests, audio, sample_rate=16000)

    assert len(results) == 1
    assert results[0].success
    assert len(results[0].candidates) == 2
    assert results[0].candidates[0].text == "recovered"
    assert results[0].candidates[1].text == "text"


def test_engine_respects_max_attempts():
    # Engine that always returns empty — should hit max attempts
    engine_stub = _StubASR(responses=[[], [], []])
    engine = LocalRecoveryEngine(
        asr_engine=engine_stub,
        config=LocalRecoveryConfig(max_attempts_per_range=2),
    )
    audio = np.zeros(16000, dtype=np.float32)

    requests = [
        LocalRecoveryRequest(start=0.2, end=0.8, reasons=["uncovered"]),
    ]
    results = engine.process_requests(requests, audio, sample_rate=16000)

    assert len(results) == 1
    assert not results[0].success
    assert results[0].outcome == "max_attempts_exceeded"
    assert results[0].attempt_count == 2


def test_engine_rejects_candidates_below_min_confidence():
    engine_stub = _StubASR(responses=[
        [
            TranscriptionSegment(
                text="low conf",
                start=0.0,
                end=0.3,
                words=[WordTimestamp("low", 0.0, 0.15, confidence=0.3)],
            )
        ]
    ])
    engine = LocalRecoveryEngine(
        asr_engine=engine_stub,
        config=LocalRecoveryConfig(min_confidence=0.5),
    )
    audio = np.zeros(16000, dtype=np.float32)

    results = engine.process_requests(
        [LocalRecoveryRequest(start=0.0, end=0.3, reasons=["low_conf"])],
        audio, sample_rate=16000,
    )
    assert len(results[0].candidates) == 0


def test_engine_stops_early_on_success():
    """When first attempt recovers, don't try again."""
    # Word returned at local [0.3, 0.5], request [0.2, 0.8], context_start=0.0.
    # Global: [0.3, 0.5] overlaps with request [0.2, 0.8].
    engine_stub = _StubASR(responses=[
        [
            TranscriptionSegment(
                text="ok",
                start=0.3,
                end=0.5,
                words=[WordTimestamp("ok", 0.3, 0.5, confidence=0.9)],
            )
        ],
        [],  # second response would be empty but should never be called
    ])
    engine = LocalRecoveryEngine(
        asr_engine=engine_stub,
        config=LocalRecoveryConfig(max_attempts_per_range=3),
    )
    audio = np.zeros(16000, dtype=np.float32)

    engine.process_requests(
        [LocalRecoveryRequest(start=0.2, end=0.8, reasons=["uncovered"])],
        audio, sample_rate=16000,
    )
    assert len(engine_stub.calls) == 1


def test_engine_applies_context_window():
    engine_stub = _StubASR(responses=[
        [TranscriptionSegment(text="x", start=0.0, end=0.1,
                              words=[WordTimestamp("x", 0.0, 0.1)])]
    ])
    engine = LocalRecoveryEngine(
        asr_engine=engine_stub,
        config=LocalRecoveryConfig(context_window=0.5),
    )
    audio = np.zeros(160000, dtype=np.float32)  # 10s

    engine.process_requests(
        [LocalRecoveryRequest(start=1.0, end=2.0, reasons=["uncovered"])],
        audio, sample_rate=16000,
    )
    # Window should be [0.5, 2.5] = 2.0s = 32000 samples
    call = engine_stub.calls[0]
    # Allow small rounding difference
    assert abs(call["audio_len"] - 32000) <= 1


def test_engine_skips_empty_requests():
    engine_stub = _StubASR()
    engine = LocalRecoveryEngine(asr_engine=engine_stub)
    audio = np.zeros(16000, dtype=np.float32)

    results = engine.process_requests([], audio, sample_rate=16000)
    assert results == []
    assert len(engine_stub.calls) == 0


def test_engine_maps_recovery_to_global_words():
    """ASR returns words — they should map to global coordinates via context_start."""
    # Segment audio from context_start=0.0 to context_end=2.0.
    # ASR returns words at local [0.6, 0.8] and [0.8, 1.0].
    # Global: [0.6, 0.8] and [0.8, 1.0], both overlap with request [0.5, 1.5].
    engine_stub = _StubASR(responses=[
        [
            TranscriptionSegment(
                text="found word",
                start=0.6,
                end=1.0,
                words=[WordTimestamp("found", 0.6, 0.8, confidence=0.88),
                       WordTimestamp("word", 0.8, 1.0, confidence=0.92)],
            )
        ]
    ])
    engine = LocalRecoveryEngine(asr_engine=engine_stub, word_id_prefix="rec")
    audio = np.zeros(48000, dtype=np.float32)  # 3s

    results = engine.process_requests(
        [LocalRecoveryRequest(start=0.5, end=1.5, reasons=["uncovered"])],
        audio, sample_rate=16000,
    )

    assert len(results[0].candidates) == 2
    assert results[0].candidates[0].word_id.startswith("rec:")
    assert results[0].candidates[0].start == 0.6
    assert results[0].candidates[0].confidence == 0.88


def test_engine_fallback_asr():
    """When primary engine returns empty, fallback engine is tried."""
    primary = _StubASR(responses=[])  # always empty
    # Fallback returns word at local [0.3, 0.5], request [0.2, 0.8]
    # Global: [0.3, 0.5] overlaps with [0.2, 0.8].
    fallback = _StubASR(responses=[
        [TranscriptionSegment(text="fb", start=0.3, end=0.5,
                              words=[WordTimestamp("fb", 0.3, 0.5)])]
    ])
    engine = LocalRecoveryEngine(
        asr_engine=primary,
        fallback_asr=fallback,
        config=LocalRecoveryConfig(max_attempts_per_range=2),
    )
    audio = np.zeros(16000, dtype=np.float32)

    results = engine.process_requests(
        [LocalRecoveryRequest(start=0.2, end=0.8, reasons=["uncovered"])],
        audio, sample_rate=16000,
    )
    assert results[0].success
    assert results[0].candidates[0].text == "fb"
    assert engine._fallback is not None


def test_engine_rejects_overlapping_candidates_without_physical_evidence():
    """Candidate words outside the request range must be rejected."""
    # ASR returns word far from the request range
    engine_stub = _StubASR(responses=[
        [
            TranscriptionSegment(
                text="far away",
                start=5.0, end=5.5,
                words=[WordTimestamp("far", 5.0, 5.25, confidence=0.9)],
            )
        ]
    ])
    engine = LocalRecoveryEngine(
        asr_engine=engine_stub,
        config=LocalRecoveryConfig(min_confidence=0.5),
    )
    audio = np.zeros(160000, dtype=np.float32)  # 10s

    # Request [0.5, 1.5], context_window=0.5 → context [0.0, 2.0].
    # ASR returns word at local 5.0s, global [5.0, 5.25], far outside [0.5, 1.5].
    results = engine.process_requests(
        [LocalRecoveryRequest(start=0.5, end=1.5, reasons=["uncovered"])],
        audio, sample_rate=16000,
    )
    # Word is outside request range — should be rejected
    assert len(results[0].candidates) == 0


def test_recovery_global_word_conversion():
    """RecoveryCandidate.to_global_word produces a valid GlobalWord."""
    candidate = RecoveryCandidate(word_id="rec:001", text="hello", start=1.0, end=1.3, confidence=0.85)
    gw = candidate.to_global_word(source_window_id="recovery", segment_id="recovery-seg")
    assert gw.id == "rec:001"
    assert gw.text == "hello"
    assert gw.raw_start == 1.0
    assert gw.raw_end == 1.3
    assert gw.confidence == 0.85
    assert gw.source_window_id == "recovery"
    assert gw.metadata.get("source") == "local_recovery"


def test_engine_handles_asr_exception_gracefully():
    class _FailingASR(ASREngine):
        @property
        def name(self):
            return "failing"
        @property
        def model_name(self):
            return "failing"
        def load_model(self):
            return None
        def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
            raise RuntimeError("ASR crash")

    engine = LocalRecoveryEngine(
        asr_engine=_FailingASR(),
        config=LocalRecoveryConfig(max_attempts_per_range=2),
    )
    audio = np.zeros(16000, dtype=np.float32)

    results = engine.process_requests(
        [LocalRecoveryRequest(start=0.2, end=0.8, reasons=["uncovered"])],
        audio, sample_rate=16000,
    )
    assert not results[0].success
    assert results[0].outcome == "asr_error"
    assert results[0].attempt_count == 2
