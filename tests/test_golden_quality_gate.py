from vocal_subtitle.quality.golden_gate import (
    audit_case_coverage,
    classify_expected_match,
    evaluate_golden_set,
)


def test_golden_gate_reports_production_metrics_and_required_categories():
    report = evaluate_golden_set(
        [{
            "id": "zh-dialogue",
            "categories": ["chinese", "multi_speaker"],
            "expected_events": [
                {"start": 0.0, "end": 1.0, "text": "你好", "kind": "speech"},
                {"start": 2.0, "end": 2.5, "text": "music", "kind": "non_speech"},
            ],
            "predicted_events": [
                {"start": 0.0, "end": 1.0, "text": "你好"},
            ],
            "decisions": [{"decision": "keep", "revision_trace": [{"stage": "test"}]}],
            "diagnostics": {
                "physical_violation_count": 0,
                "cross_silence_count": 0,
                "raw_event_bypass_count": 0,
            },
        }],
        required_categories=["chinese", "multi_speaker"],
    )

    assert report["publishable"] is True
    assert report["metrics"]["real_speech_drop_rate"] == 0.0
    assert report["metrics"]["hallucination_retention_rate"] == 0.0


def test_golden_gate_blocks_hallucination_retention_and_raw_bypass():
    report = evaluate_golden_set([{
        "categories": ["non_speech"],
        "expected_events": [{"start": 0.0, "end": 1.0, "text": "music", "kind": "non_speech"}],
        "predicted_events": [{"start": 0.0, "end": 1.0, "text": "music"}],
        "decisions": [{"decision": "drop"}],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 1,
        },
    }], required_categories=["english"])

    assert report["publishable"] is False
    assert report["checks"]["hallucination_retention"] is False
    assert report["checks"]["raw_bypass"] is False
    assert report["checks"]["required_categories"] is False


def test_golden_gate_blocks_missing_runtime_diagnostics():
    report = evaluate_golden_set([{
        "categories": ["chinese"],
        "expected_events": [],
        "predicted_events": [],
        "decisions": [],
        "diagnostics": {},
    }], required_categories=["chinese"])

    assert report["publishable"] is False
    assert report["checks"]["diagnostics_complete"] is False
    assert report["metrics"]["missing_diagnostics"] == {
        "physical_violation_count": 1,
        "cross_silence_count": 1,
        "raw_event_bypass_count": 1,
    }


def test_golden_gate_action_rates_include_keep_and_replace():
    report = evaluate_golden_set([{
        "categories": ["chinese"],
        "expected_events": [],
        "predicted_events": [],
        "decisions": [
            {"decision": "keep", "revision_trace": [{"stage": "test"}]},
            {"decision": "replace", "revision_trace": [{"stage": "test"}]},
            {"decision": "unresolved", "revision_trace": [{"stage": "test"}]},
        ],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    assert report["metrics"]["action_counts"] == {
        "keep": 1,
        "replace": 1,
        "unresolved": 1,
        "split": 0,
        "drop": 0,
    }
    assert report["metrics"]["unresolved_rate"] == 0.333333


def test_golden_gate_preserves_production_pair_metadata():
    report = evaluate_golden_set(
        [],
        metadata={
            "primary_engine": "faster-whisper",
            "secondary_engine": "qwen",
            "review_policy": "risk_only",
            "pair_route_version": "asr-pair-v1",
        },
    )

    assert report["metadata"]["primary_engine"] == "faster-whisper"
    assert report["metadata"]["secondary_engine"] == "qwen"
    assert report["metadata"]["review_policy"] == "risk_only"


def test_golden_gate_reports_event_level_miss_attribution():
    report = evaluate_golden_set([{
        "id": "debug-case",
        "categories": ["english_or_mixed"],
        "expected_events": [
            {"start": 1.0, "end": 2.0, "text": "hello world", "kind": "speech"},
            {"start": 3.0, "end": 4.0, "text": "goodbye", "kind": "speech"},
        ],
        "predicted_events": [
            {"start": 1.1, "end": 1.8, "text": "zzzzzz"},
            {"start": 4.1, "end": 4.7, "text": "goodbye"},
        ],
        "decisions": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    assert report["metrics"]["miss_attribution_counts"] == {
        "asr_text_mismatch": 1,
        "timeline_shift_or_boundary_mismatch": 1,
    }
    summary = report["case_summaries"][0]
    assert summary["id"] == "debug-case"
    assert summary["miss_count"] == 2
    assert summary["miss_attribution"][0]["best_overlap"]["overlap_seconds"] > 0


def test_classify_expected_match_keeps_gate_strict_but_explains_nearby_text():
    result = classify_expected_match(
        {"start": 1.0, "end": 2.0, "text": "same phrase"},
        [{"start": 2.2, "end": 2.8, "text": "same phrase"}],
    )

    assert result["matched"] is False
    assert result["stage"] == "timeline_shift_or_boundary_mismatch"
    assert result["best_text"]["text_similarity"] == 1.0


def test_advisory_reference_miss_does_not_block_default_safety_gate():
    report = evaluate_golden_set([{
        "id": "advisory-miss",
        "reference_role": "advisory",
        "reference_status": "manual_reference",
        "expected_events": [{
            "start": 0.0,
            "end": 1.0,
            "text": "人工参考内容",
            "kind": "speech",
        }],
        "predicted_events": [],
        "decisions": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    assert report["gate_mode"] == "safety"
    assert report["publishable"] is True
    assert report["safety_gate"]["passed"] is True
    assert report["reference_quality"]["status"] == "fail"
    assert report["release_status"] == "reference_improvement_required"
    assert report["checks"]["real_speech_drop"] is False


def test_strict_reference_blocks_advisory_quality_miss():
    report = evaluate_golden_set([{
        "reference_role": "advisory",
        "expected_events": [{"start": 0.0, "end": 1.0, "text": "参考"}],
        "predicted_events": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }], gate_mode="strict-reference")

    assert report["publishable"] is False
    assert report["status"] == "fail"
    assert report["safety_gate"]["passed"] is True
    assert report["reference_quality"]["passed"] is False


def test_no_reference_speech_events_are_not_counted_as_reference_quality():
    report = evaluate_golden_set([{
        "reference_role": "none",
        "reference_status": "no_manual_reference",
        "expected_events": [{"start": 0.0, "end": 1.0, "text": "not used"}],
        "predicted_events": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    assert report["metrics"]["expected_speech_count"] == 0
    assert report["reference_quality"]["status"] == "not_applicable"


def test_strict_match_is_reported_alongside_legacy_match():
    expected = {"start": 1.0, "end": 2.0, "text": "same phrase"}
    predicted = [{"start": 1.999, "end": 2.5, "text": "same phrase"}]

    legacy = classify_expected_match(expected, predicted)
    report = evaluate_golden_set([{
        "reference_role": "strict",
        "reference_status": "manual_reference",
        "expected_events": [expected],
        "predicted_events": predicted,
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    assert legacy["matched"] is True
    assert report["metrics"]["real_speech_drop_rate"] == 0.0
    assert report["metrics"]["strict_speech_drop_rate"] == 1.0
    assert report["metrics"]["strict_match_only_failure_count"] == 1


def test_physical_coverage_reports_blank_and_trailing_silence_independently():
    coverage = audit_case_coverage(
        [{"id": "speech-1", "start": 1.0, "end": 2.0}],
        [{"start": 1.0, "end": 2.5, "text": "hello"}],
    )

    assert coverage["status"] == "pass"
    assert coverage["audible_blank_rate"] == 0.0
    assert coverage["trailing_silence_ms"] == 500.0

    report = evaluate_golden_set([{
        "expected_events": [],
        "predicted_events": [],
        "physical_speech_spans": [{"start": 1.0, "end": 2.0}],
        "final_cues": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])
    assert report["coverage_quality"]["audible_blank_rate"] == 1.0
    assert report["coverage_quality"]["case_counts"]["fail"] == 1
    assert report["publishable"] is True


def test_physical_coverage_is_not_evaluable_without_speech_spans():
    coverage = audit_case_coverage([], [{"start": 0.0, "end": 1.0, "text": "hello"}])

    assert coverage["status"] == "not_evaluable"
    assert coverage["audible_blank_rate"] is None
    assert coverage["trailing_silence_ms"] is None


def test_engine_status_is_aggregated_and_missing_cases_are_explicit():
    report = evaluate_golden_set([{
        "engine_status": {
            "faster-whisper": {
                "selected": True,
                "enabled": True,
                "available": True,
                "windows_processed": 2,
                "windows_failed": 0,
            },
        },
        "expected_events": [],
        "predicted_events": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }, {
        "expected_events": [],
        "predicted_events": [],
        "diagnostics": {
            "physical_violation_count": 0,
            "cross_silence_count": 0,
            "raw_event_bypass_count": 0,
        },
    }])

    assert report["engine_status"]["faster-whisper"]["selected_count"] == 1
    assert report["metrics"]["engine_status_missing_case_count"] == 1
