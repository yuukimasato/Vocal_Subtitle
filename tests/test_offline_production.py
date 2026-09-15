from types import SimpleNamespace

import pytest

from vocal_subtitle.application.offline_production import (
    OfflineProductionCoordinator,
    OfflineProductionRequest,
)
from vocal_subtitle.application.pipeline_result import PipelineStats
from vocal_subtitle.asr.contracts import EvidenceReviewResult
from vocal_subtitle.asr.evidence import EvidenceDecision, EvidenceWord
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.physical.decision_projection import DecisionEventProjector
from vocal_subtitle.physical.timeline import PhysicalTimeline


def _config(*, shadow_mode: bool, fallback_to_segmented: bool = True):
    return SimpleNamespace(
        enabled=True,
        shadow_mode=shadow_mode,
        fallback_to_segmented=fallback_to_segmented,
    )


def _event(text="main"):
    return SubtitleEvent(index=1, start=1.0, end=1.4, text=text)


def _decision(text="reviewed", physical_clip_id="clip-1"):
    word = EvidenceWord(
        id="segmented:event:000001:word:0000",
        text=text,
        start=1.05,
        end=1.35,
        confidence=0.9,
    )
    return EvidenceDecision(
        candidate_ids=("segmented:event:000001",),
        decision="keep",
        final_text=text,
        final_words=(word,),
        start=1.0,
        end=1.4,
        time_source="native_word_timestamp",
        confidence=0.9,
        risk_score=0.1,
        risk_level="low",
        physical_validation={"valid": True, "physical_clip_id": physical_clip_id},
        revision_trace=({"stage": "test"},),
    )


class FakeReviewService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def run(self, request, ports):
        self.calls.append((request, ports))
        if self.error:
            raise self.error
        return self.result


def test_projector_builds_events_without_legacy_conversion_helper():
    event = DecisionEventProjector().project([_decision()])[0]

    assert event.text == "reviewed"
    assert event.start == 1.0
    assert event.end == 1.4
    assert event.words[0].start == pytest.approx(0.05)
    assert event.source_word_ids == ["segmented:event:000001:word:0000"]
    assert event.physical_region_id == "clip-1"
    assert event.revision_trace[-1]["stage"] == "decision_event_projection"


def test_projector_binds_events_to_physical_evidence():
    timeline = PhysicalTimeline.from_duration(3.0)
    clip_id = timeline.physical_clips[0].id
    timeline.add_evidence(
        0.9,
        1.2,
        "silero",
        physical_clip_id=clip_id,
    )
    event = DecisionEventProjector().project(
        [_decision(physical_clip_id=clip_id)], timeline
    )[0]

    assert event.physical_region_id == clip_id
    assert event.physical_start == pytest.approx(1.0)
    assert event.physical_end == pytest.approx(1.2)
    assert event.physical_spans[0]["source"] == "silero"


def test_coordinator_shadow_preserves_baseline_and_records_decisions():
    baseline = _event("baseline")
    review = FakeReviewService(
        EvidenceReviewResult(
            events=[_event("legacy-review-event")],
            decisions=[_decision()],
            diagnostics={"status": "ok"},
        )
    )
    coordinator = OfflineProductionCoordinator(review_service=review)

    result = coordinator.run(
        OfflineProductionRequest(events=[baseline]),
        ports=SimpleNamespace(config=_config(shadow_mode=True)),
    )

    assert result.events == [baseline]
    assert result.decisions == [_decision()]
    assert result.diagnostics["production_path"] == "shadow"
    assert result.diagnostics["decision_count"] == 1
    assert result.diagnostics["physical_projection"]["mode"] == "shadow_observed"
    assert result.diagnostics["physical_projection"]["returned_event_count"] == 1


def test_coordinator_authoritative_uses_projector_result():
    baseline = _event("baseline")
    review = FakeReviewService(
        EvidenceReviewResult(
            events=[baseline],
            decisions=[_decision()],
            diagnostics={"status": "ok"},
        )
    )
    coordinator = OfflineProductionCoordinator(review_service=review)

    result = coordinator.run(
        OfflineProductionRequest(events=[baseline]),
        ports=SimpleNamespace(config=_config(shadow_mode=False)),
    )

    assert [item.text for item in result.events] == ["reviewed"]
    assert result.events[0] is not baseline
    assert result.diagnostics["production_path"] == "authoritative"
    assert result.diagnostics["status"] == "completed"


def test_coordinator_falls_back_to_segmented_events_with_diagnostics():
    baseline = _event()
    coordinator = OfflineProductionCoordinator(
        review_service=FakeReviewService(error=RuntimeError("review failed"))
    )

    result = coordinator.run(
        OfflineProductionRequest(events=[baseline]),
        ports=SimpleNamespace(config=_config(shadow_mode=False)),
    )

    assert len(result.events) == 1
    assert result.events[0].text == baseline.text
    assert result.events[0].start == pytest.approx(baseline.start)
    assert result.events[0].end == pytest.approx(baseline.end)
    assert result.events[0].revision_trace[-1]["stage"] == "decision_event_projection"
    assert result.diagnostics["production_path"] == "segmented_fallback"
    assert result.diagnostics["status"] == "degraded"
    assert result.diagnostics["review_status"] == "failed"
    assert (
        result.diagnostics["physical_projection"]["mode"]
        == "segmented_fallback_projected"
    )
    assert result.diagnostics["physical_projection"]["raw_event_bypass_count"] == 0
    assert result.diagnostics["decision_count"] == 1


def test_coordinator_empty_input_has_complete_zero_projection_diagnostics():
    coordinator = OfflineProductionCoordinator()

    result = coordinator.run(
        OfflineProductionRequest(events=[]),
        ports=SimpleNamespace(config=_config(shadow_mode=False)),
    )

    assert result.events == []
    assert result.diagnostics["production_path"] == "empty_input"
    assert result.diagnostics["review_status"] == "ok"
    assert result.diagnostics["physical_projection"] == {
        "mode": "empty_input",
        "decision_count": 0,
        "event_count": 0,
        "physical_violation_count": 0,
        "cross_silence_count": 0,
        "decision_trace_missing_count": 0,
        "raw_event_bypass_count": 0,
    }


def _physical_decision(
    decision_id: str,
    text: str,
    start: float,
    end: float,
    *,
    words=(),
    decision: str = "keep",
):
    return EvidenceDecision(
        candidate_ids=(decision_id,),
        decision=decision,
        final_text="" if decision == "drop" else text,
        final_words=tuple(words),
        start=None if decision == "drop" else start,
        end=None if decision == "drop" else end,
        time_source="native_word_timestamp" if words else "segment_boundary",
        confidence=0.9 if decision != "drop" else None,
        risk_score=0.8 if decision == "unresolved" else 0.1,
        risk_level="high" if decision == "unresolved" else "low",
        physical_validation={"valid": True},
        revision_trace=({"stage": "test", "decision_id": decision_id},),
    )


def test_physical_projection_keeps_cross_clip_word_inside_one_bin():
    timeline = PhysicalTimeline(2.0)
    timeline.add_clip(0.0, 1.0, clip_id="clip-a")
    timeline.add_clip(1.0, 2.0, clip_id="clip-b")
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-a")
    timeline.add_evidence(1.0, 2.0, "ffmpeg_skeleton", physical_clip_id="clip-b")
    decision = _physical_decision(
        "candidate-1",
        "crossing",
        0.8,
        1.2,
        words=(EvidenceWord("source-word", "crossing", 0.8, 1.2, 0.9),),
    )

    events = DecisionEventProjector().project([decision], timeline)

    assert len(events) == 1
    assert events[0].text == "crossing"
    assert events[0].source_word_ids == ["source-word"]
    assert {item["physical_clip_id"] for item in events[0].physical_spans} == {"clip-b"}
    assert "cross_physical_boundary" in (events[0].alignment_warning or "")
    assert events[0].revision_trace[0]["stage"] == "evidence_decision"


def test_physical_projection_aggregates_multiple_decisions_into_one_bin_event():
    timeline = PhysicalTimeline(2.0)
    timeline.add_clip(0.0, 2.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 1.5, "ffmpeg_skeleton", physical_clip_id="clip-a")
    decisions = [
        _physical_decision(
            "decision-a",
            "第一句",
            0.1,
            0.5,
            words=(EvidenceWord("word-a", "第一句", 0.1, 0.5, 0.9),),
        ),
        _physical_decision(
            "decision-b",
            "第二句",
            0.5,
            0.9,
            words=(EvidenceWord("word-b", "第二句", 0.5, 0.9, 0.9),),
        ),
    ]

    result = DecisionEventProjector().project_with_diagnostics(decisions, timeline)

    assert len(result.events) == 1
    assert result.events[0].text == "第一句第二句"
    assert result.events[0].source_word_ids == ["word-a", "word-b"]
    assert result.events[0].physical_bin_id == "subtitle-bin-000001"
    assert result.events[0].physical_start >= result.events[0].physical_bin_start
    assert result.events[0].physical_end <= result.events[0].physical_bin_end
    assert {
        item["decision_id"]
        for item in result.events[0].revision_trace
        if item["stage"] == "evidence_decision"
    } == {
        "decision:000001",
        "decision:000002",
    }


def test_physical_projection_drops_silent_word_and_reports_coverage_gap():
    timeline = PhysicalTimeline(2.0)
    timeline.add_clip(0.0, 2.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 0.4, "ffmpeg_skeleton", physical_clip_id="clip-a")
    timeline.add_evidence(1.0, 1.4, "ffmpeg_skeleton", physical_clip_id="clip-a")
    decisions = [
        _physical_decision(
            "spoken",
            "spoken",
            0.1,
            0.3,
            words=(EvidenceWord("spoken-word", "spoken", 0.1, 0.3, 0.9),),
        ),
        _physical_decision(
            "hallucination",
            "hallucination",
            0.6,
            0.8,
            words=(EvidenceWord("silent-word", "hallucination", 0.6, 0.8, 0.9),),
        ),
    ]

    projector = DecisionEventProjector()
    result = projector.project_with_diagnostics(decisions, timeline)

    assert [event.text for event in result.events] == ["spoken"]
    assert result.diagnostics["rejected_word_ids"] == [
        "decision:000002:word:0000:silent-word"
    ]
    assert result.diagnostics["coverage"]["complete"] is False


def test_physical_projection_marks_missing_word_times_and_unresolved():
    timeline = PhysicalTimeline(2.0)
    timeline.add_clip(0.0, 2.0, clip_id="clip-a")
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton", physical_clip_id="clip-a")
    decisions = [
        _physical_decision(
            "missing-times",
            "no timing",
            0.2,
            0.6,
            words=(EvidenceWord("missing", "no", time_source="segment_boundary"),),
        ),
        _physical_decision(
            "unresolved",
            "uncertain",
            0.7,
            0.9,
            words=(EvidenceWord("uncertain-word", "uncertain", 0.7, 0.9, 0.5),),
            decision="unresolved",
        ),
        _physical_decision(
            "dropped",
            "removed",
            0.9,
            1.0,
            decision="drop",
        ),
    ]

    events = DecisionEventProjector().project(decisions, timeline)

    assert [event.text for event in events] == ["no uncertain"]
    assert "missing_word_timestamps" in events[0].alignment_warning
    assert "unresolved evidence conflict" in (events[0].alignment_warning or "")


def test_stats_persist_production_diagnostics():
    stats = PipelineStats(input_path="input.wav", duration_seconds=1.0)
    stats.production_path = "shadow"
    stats.review_status = "ok"
    stats.decision_count = 2
    stats.status = "degraded"
    stats.error_category = "model_unavailable"
    stats.diagnostics_complete = True

    restored = PipelineStats.from_dict(stats.input_path, stats.to_dict())

    assert restored.production_path == "shadow"
    assert restored.review_status == "ok"
    assert restored.decision_count == 2
    assert restored.status == "degraded"
    assert restored.error_category == "model_unavailable"
    assert restored.diagnostics_complete is True
