from pathlib import Path
from types import SimpleNamespace

from vocal_subtitle.asr.evidence import CandidateEvidence, EvidenceDecision, EvidenceWord
from vocal_subtitle.quality.golden_gate import evaluate_golden_set
from vocal_subtitle.quality.provenance import (
    attribute_case_misses,
    trace_context_for,
)
from vocal_subtitle.reporting.capability_maturity import build_capability_maturity
from vocal_subtitle.reporting.feedback_profile import inspect_feedback_profile
from vocal_subtitle.reporting.noise_shadow import build_noise_shadow
from vocal_subtitle.asr.trace_contract import attach_event_trace
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.quality.provenance import normalize_trace_context


def _expected():
    return {"id": "expected:1", "start": 1.0, "end": 2.0, "text": "hello"}


def test_normalize_trace_context_recovers_ids_from_span_payload_dicts():
    normalized = normalize_trace_context({
        "physical_span_ids": [
            "clip-000001",
            {
                "physical_clip_id": "clip-000002",
                "start": 0.0,
                "end": 1.0,
                "evidence_ids": ["evidence:ffmpeg_skeleton:000000:000016"],
                "source": "ffmpeg_skeleton",
            },
            {"evidence_ids": ["evidence:rms:000000:000004"], "start": 2.0, "end": 3.0},
            None,
        ],
    })
    assert normalized["physical_span_ids"] == [
        "clip-000001",
        "clip-000002",
        "evidence:rms:000000:000004",
    ]


def test_attribution_tolerates_span_payload_dicts_in_final_event_traces():
    predicted = [{
        "id": "final:event:000001",
        "start": 0.0,
        "end": 1.007,
        "text": "这是我朋友的店我",
        "trace_context": {
            "final_event_ids": ["final:event:000001"],
            "physical_span_ids": [{
                "physical_clip_id": "clip-000001",
                "start": 0.0,
                "end": 1.007,
                "evidence_ids": ["evidence:ffmpeg_skeleton:000000:000016"],
                "source": "ffmpeg_skeleton",
            }],
        },
    }]
    misses = attribute_case_misses(
        [_expected()],
        predicted,
        physical_spans=[{"id": "physical:1", "start": 0.0, "end": 1.007}],
        final_events=predicted,
    )
    assert misses[0]["final_event_ids"] == ["final:event:000001"]
    assert misses[0]["physical_span_ids"] == ["physical:1"]


def test_provenance_attributes_physical_gap_and_strict_only_failure():
    missing_physical = attribute_case_misses(
        [_expected()],
        [],
        physical_spans=[{"id": "physical:other", "start": 3.0, "end": 4.0}],
    )
    assert missing_physical[0]["stage"] == "physical_coverage_missing"
    assert missing_physical[0]["physical_span_ids"] == []

    strict_only = attribute_case_misses(
        [_expected()],
        [{"id": "final:1", "start": 1.999, "end": 2.4, "text": "hello"}],
    )
    assert strict_only[0]["stage"] == "strict_match_only_failure"


def test_candidate_decision_and_final_trace_ids_are_serializable_and_stable():
    word = EvidenceWord("word:1", "hello", 1.1, 1.5, 0.9)
    candidate = CandidateEvidence(
        id="segmented:1",
        source="segmented",
        text="hello",
        start=1.0,
        end=2.0,
        words=(word,),
        offset_id="offset:1",
        window_id="window:1",
    )
    decision = EvidenceDecision(
        candidate_ids=(candidate.id,),
        decision="keep",
        final_text=candidate.text,
        final_words=(word,),
        start=1.0,
        end=2.0,
        time_source="native_word_timestamp",
        confidence=0.9,
        risk_score=0.1,
        risk_level="low",
        physical_validation={"valid": True},
    )
    assert candidate.to_dict()["trace_context"]["candidate_id"] == candidate.id
    assert candidate.to_dict()["words"][0]["dedup_key"] == word.dedup_key
    assert decision.to_dict()["decision_id"] == decision.decision_id
    assert decision.revision_trace[-1]["stage"] == "evidence_decision"
    assert trace_context_for(decision)["decision_id"] == decision.decision_id


def test_global_role_statistics_are_reported_without_replacing_legacy_metrics():
    report = evaluate_golden_set([{
        "expected_events": [],
        "predicted_events": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
            "global_evidence": {
                "signal_count": 2,
                "considered_alternative_count": 1,
                "accepted_alternative_count": 1,
                "selected_global_count": 0,
            },
        },
    }])
    assert report["metrics"]["global_evidence"] == {
        "signal_count": 2,
        "considered_alternative_count": 1,
        "accepted_alternative_count": 1,
        "selected_global_count": 0,
        "rejected_count": 0,
    }


def test_phase_six_shadow_outputs_do_not_apply_configuration(tmp_path: Path):
    noise = build_noise_shadow(
        noise_floor_db=-52.0,
        current_vad_threshold=0.5,
        current_skeleton_noise_db=-40.0,
    )
    assert noise["status"] == "advisory"
    assert noise["suggested_skeleton_noise_db"] == -42.0
    assert noise["applied"] is False
    assert noise["overrides"] == {}

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile = profile_dir / "demo.yaml"
    profile.write_text("profile_id: demo\noverrides:\n  vad.threshold: 0.4\nupdated_at: v1\n", encoding="utf-8")
    config = SimpleNamespace(
        feedback=SimpleNamespace(
            enabled=True,
            active_profile="demo",
            user_profile_dir=str(profile_dir),
        )
    )
    status = inspect_feedback_profile(config)
    assert status["loaded"] is True
    assert status["applied"] is False
    assert status["overrides_hash"]


def test_capability_maturity_is_independent_from_release_status():
    maturity = build_capability_maturity(
        evidence={"status": "ok", "candidate_count": 3, "optional_engines": {}},
        quality={},
        config=SimpleNamespace(feedback=SimpleNamespace(enabled=False, active_profile="user_default")),
    )
    assert maturity["release_status_semantics"] == "independent_from_capability_maturity"
    assert maturity["capabilities"]["segmented_primary"]["release_default"] is True
    assert maturity["capabilities"]["qwen_review"]["release_default"] is False


def test_chunk_and_skeleton_events_share_the_same_trace_contract():
    event = SubtitleEvent(index=1, start=0.0, end=0.5, text="hello")
    attach_event_trace(
        event,
        source_id="skeleton",
        offset_id="skeleton:0",
        window_id="skeleton-window:000000",
    )
    restored = SubtitleEvent.from_dict(event.to_dict())
    assert restored.trace_context["schema_version"] == "offline-trace-v1"
    assert restored.trace_context["offset_id"] == "skeleton:0"
    assert any(item["stage"] == "evidence_contract" for item in restored.revision_trace)
