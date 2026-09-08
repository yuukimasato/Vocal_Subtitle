from types import SimpleNamespace

import pytest

from vocal_subtitle.asr.evidence import (
    CandidateEvidence,
    EvidenceBundle,
    EvidenceDecision,
    EvidenceWord,
    candidate_from_subtitle_event,
    candidates_from_global_transcript,
    candidates_from_segments,
    evidence_word_from_asr,
)
from vocal_subtitle.asr.evidence_decision import EvidenceDecisionEngine
from vocal_subtitle.asr.evidence_review import EvidenceReviewRuntimePorts, EvidenceReviewService
from vocal_subtitle.asr.evidence_cache import EvidenceCacheKeyContext
from vocal_subtitle.asr.contracts import EvidenceReviewRequest
from vocal_subtitle.asr.review_scheduler import ReviewScheduler, ReviewSchedulerConfig
from vocal_subtitle.asr.risk_scoring import EvidenceRiskScorer, RiskScoringConfig


def candidate(identifier="seg-1", text="hello", start=1.0, end=2.0, **kwargs):
    return CandidateEvidence(
        id=identifier,
        source=kwargs.pop("source", "segmented"),
        text=text,
        start=start,
        end=end,
        **kwargs,
    )


def candidate(identifier="seg-1", text="hello", start=1.0, end=2.0, **kwargs):
    return CandidateEvidence(
        id=identifier,
        source=kwargs.pop("source", "segmented"),
        text=text,
        start=start,
        end=end,
        **kwargs,
    )


def test_evidence_cache_key_changes_with_window_and_policy():
    base = dict(
        input_hash="input-a",
        audio_hash="audio-a",
        physical_timeline_version="physical-timeline-v1",
        window_start=1.0,
        window_end=2.0,
        phase="context_reasr",
        engine="faster-whisper",
        model="large-v3",
        language="en",
        route_version="route-v1",
    )
    changed_window = EvidenceCacheKeyContext(**{**base, "window_end": 2.1})
    changed_engine = EvidenceCacheKeyContext(**{**base, "engine": "funasr"})
    changed_sample_rate = EvidenceCacheKeyContext(**{**base, "sample_rate": 8000})

    assert EvidenceCacheKeyContext(**base).key() != changed_window.key()
    assert EvidenceCacheKeyContext(**base).key() != changed_engine.key()
    assert EvidenceCacheKeyContext(**base).key() != changed_sample_rate.key()


def test_evidence_cache_key_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        EvidenceCacheKeyContext(
            input_hash="input-a",
            audio_hash="",
            physical_timeline_version="physical-timeline-v1",
            window_start=1.0,
            window_end=2.0,
            phase="context_reasr",
            engine="faster-whisper",
            model="large-v3",
            language="en",
            route_version="route-v1",
            sample_rate=0,
        )


def test_evidence_review_service_uses_window_cache():
    class MemoryCache:
        def __init__(self):
            self.values = {}

        def get(self, stage, key):
            return self.values.get((stage, key))

        def set(self, stage, key, value, ttl=None):
            self.values[(stage, key)] = value

    class EmptyReviewEngine:
        name = "fake-review"

        def __init__(self):
            self.calls = 0

        def review(self, audio, sample_rate, window, *, language=None):
            self.calls += 1
            return []

    config = SimpleNamespace(
        enabled=True,
        shadow_mode=True,
        context_reasr_enabled=True,
        unresolved_keeps_candidate=True,
        left_context=0.8,
        right_context=0.8,
        max_group_duration=12.0,
        max_window_duration=15.0,
        medium_threshold=0.25,
        high_threshold=0.50,
        critical_threshold=0.75,
    )
    request = EvidenceReviewRequest(
        events=[SimpleNamespace(index=1, start=1.0, end=1.3, text="hello", words=[])],
        audio=__import__("numpy").zeros(32000, dtype="float32"),
        input_hash="input-a",
        engine="faster-whisper",
        model="large-v3",
        route_version="route-v1",
    )
    cache = MemoryCache()
    engine = EmptyReviewEngine()
    ports = EvidenceReviewRuntimePorts(
        config=config,
        context_reasr=engine,
        cache=cache,
    )

    first = EvidenceReviewService().run(request, ports)
    second = EvidenceReviewService().run(request, ports)

    assert engine.calls == 1
    assert first.diagnostics["review"]["windows"][0]["wall_time_seconds"] >= 0.0
    assert "rss_mb" in first.diagnostics["review"]["windows"][0]["resources"]
    assert second.diagnostics["review"]["windows"][0]["cache_hit"] is True


def test_risk_scoring_marks_short_low_evidence_candidate_for_review():
    item = candidate(text="very fast phrase", start=1.0, end=1.3)

    assessment = EvidenceRiskScorer().score(item)

    assert assessment.level in {"medium", "high", "critical"}
    assert assessment.review_required is True
    assert "short_duration" in assessment.evidence_codes
    assert "missing_word_confidence" in assessment.evidence_codes


def test_risk_scoring_can_use_custom_thresholds():
    item = candidate(text="stable", start=1.0, end=2.0, confidence=0.95)
    assessment = EvidenceRiskScorer(
        RiskScoringConfig(medium_threshold=0.9, high_threshold=0.95, critical_threshold=0.99)
    ).score(item)

    assert assessment.level == "low"
    assert assessment.review_required is False


def test_risk_scoring_marks_non_contiguous_repeated_phrase():
    items = [
        candidate("first", text="same phrase", start=1.0, end=1.8, confidence=0.9),
        candidate("second", text="SAME phrase", start=6.0, end=6.8, confidence=0.9),
    ]

    assessments = EvidenceRiskScorer().score_bundle(items)

    assert all("repeated_phrase" in item.evidence_codes for item in assessments)
    assert all(item.review_required for item in assessments)


def test_scheduler_merges_adjacent_high_risk_candidates_within_same_clip():
    candidates = [
        candidate("a", text="first", start=1.0, end=1.4, physical_clip_id="clip-1"),
        candidate("b", text="second", start=1.5, end=1.9, physical_clip_id="clip-1"),
    ]
    assessments = [
        SimpleNamespace(candidate_id="a", review_required=True),
        SimpleNamespace(candidate_id="b", review_required=True),
    ]

    windows = ReviewScheduler(ReviewSchedulerConfig()).schedule(candidates, assessments)

    assert len(windows) == 1
    assert windows[0].candidate_ids == ("a", "b")
    assert windows[0].start == pytest.approx(0.2)
    assert windows[0].end == pytest.approx(2.7)


def test_scheduler_does_not_cross_long_speech_gap():
    candidates = [
        candidate("a", text="first", start=1.0, end=1.4),
        candidate("b", text="second", start=3.0, end=3.4),
    ]
    assessments = [
        SimpleNamespace(candidate_id="a", review_required=True),
        SimpleNamespace(candidate_id="b", review_required=True),
    ]
    timeline = SimpleNamespace(
        speech_evidence_spans=[
            SimpleNamespace(start=1.0, end=1.4),
            SimpleNamespace(start=3.0, end=3.4),
        ]
    )

    windows = ReviewScheduler(ReviewSchedulerConfig()).schedule(
        candidates, assessments, timeline=timeline
    )

    assert len(windows) == 2


def test_decision_engine_keeps_low_risk_candidate():
    item = candidate(confidence=0.95, words=(
        EvidenceWord("w1", "hello", 1.1, 1.5, 0.95),
    ))
    assessment = EvidenceRiskScorer().score(item)
    decision = EvidenceDecisionEngine().decide(item, assessment=assessment)

    assert isinstance(decision, EvidenceDecision)
    assert decision.decision == "keep"
    assert decision.final_text == "hello"
    assert decision.time_source == "native_word_timestamp"


def test_decision_engine_replaces_with_timed_context_candidate():
    item = candidate(text="helo", confidence=0.4)
    replacement = candidate(
        "ctx-1",
        text="hello",
        start=1.0,
        end=2.0,
        source="context_reasr",
        confidence=0.9,
        words=(EvidenceWord("w1", "hello", 1.1, 1.8, 0.9),),
    )
    assessment = EvidenceRiskScorer().score(item)
    decision = EvidenceDecisionEngine().decide(
        item,
        assessment=assessment,
        alternatives=[replacement],
    )

    assert decision.decision == "replace"
    assert decision.final_text == "hello"
    assert "context_reasr_agreement" in decision.evidence_codes


def test_decision_engine_marks_high_risk_without_review_unresolved():
    item = candidate(text="training phrase", start=1.0, end=1.2)
    assessment = EvidenceRiskScorer(
        RiskScoringConfig(critical_threshold=0.2)
    ).score(item)
    decision = EvidenceDecisionEngine().decide(item, assessment=assessment)

    assert decision.decision == "unresolved"
    assert decision.final_text == item.text
    assert "review_unavailable_or_conflicting" in decision.evidence_codes


def test_decision_engine_requires_high_risk_for_explicit_drop():
    item = candidate(confidence=0.1)
    low = EvidenceRiskScorer(RiskScoringConfig(medium_threshold=0.99)).score(item)
    with pytest.raises(ValueError, match="high or critical"):
        EvidenceDecisionEngine().drop(item, assessment=low, evidence_codes=("sed_non_speech",))


def test_decision_engine_can_emit_multiple_split_decisions():
    item = candidate(text="hello world", confidence=0.4)
    parts = [
        candidate(
            "part-1",
            text="hello",
            start=1.0,
            end=1.4,
            confidence=0.9,
            words=(EvidenceWord("p1", "hello", 1.0, 1.4, 0.9),),
        ),
        candidate(
            "part-2",
            text="world",
            start=1.5,
            end=2.0,
            confidence=0.9,
            words=(EvidenceWord("p2", "world", 1.5, 2.0, 0.9),),
        ),
    ]
    assessment = EvidenceRiskScorer().score(item)

    decisions = EvidenceDecisionEngine().split(item, parts, assessment=assessment)

    assert [decision.decision for decision in decisions] == ["split", "split"]
    assert [decision.final_text for decision in decisions] == ["hello", "world"]


def test_non_shadow_decision_conversion_preserves_word_and_physical_provenance():
    from vocal_subtitle.asr.evidence_decision import decisions_to_subtitle_events

    item = candidate(
        words=(EvidenceWord("word-1", "hello", 1.1, 1.5, 0.9),),
        physical_clip_id="clip-1",
        confidence=0.9,
    )
    assessment = EvidenceRiskScorer().score(item)
    decision = EvidenceDecisionEngine().decide(item, assessment=assessment)

    event = decisions_to_subtitle_events([decision])[0]

    assert event.source_word_ids == ["word-1"]
    assert event.physical_region_id == "clip-1"
    assert event.words[0].start == pytest.approx(0.1)


def test_physical_validation_rejects_candidate_beyond_timeline_duration():
    from vocal_subtitle.physical.timeline import PhysicalTimeline

    item = candidate(start=1.0, end=2.0, confidence=0.9)
    timeline = PhysicalTimeline.from_duration(1.5)
    timeline.add_evidence(1.0, 1.4, "silero")
    assessment = EvidenceRiskScorer().score(item)

    decision = EvidenceDecisionEngine().decide(
        item,
        assessment=assessment,
        physical_timeline=timeline,
    )

    assert decision.decision == "unresolved"
    assert decision.physical_validation["status"] == "outside_timeline"


def test_physical_validation_rejects_word_outside_candidate_and_audio_evidence():
    from vocal_subtitle.physical.timeline import PhysicalTimeline

    item = candidate(
        start=1.0,
        end=1.5,
        confidence=0.9,
        words=(EvidenceWord("w1", "hello", 1.7, 1.9, 0.9),),
    )
    timeline = PhysicalTimeline.from_duration(2.0)
    timeline.add_evidence(1.0, 1.5, "silero")
    assessment = EvidenceRiskScorer().score(item)

    decision = EvidenceDecisionEngine().decide(
        item,
        assessment=assessment,
        physical_timeline=timeline,
    )

    assert decision.decision == "unresolved"
    assert decision.physical_validation["word_validation"]["invalid_word_ids"] == ["w1"]

