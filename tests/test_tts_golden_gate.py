"""TTS 专项黄金集门禁(高精度方案 Task 9 / 优化方案 §10.3)。

TTS 骨架优先场景单独报告与门禁,不与复杂真人录音混为一组:
- 骨架 start/end 偏差(P50/P90);
- 语音覆盖率(骨架段被 cue 覆盖的比例);
- 跨骨架合并数与微停顿拆分率;
- 未达标只阻止 TTS profile 默认化,不影响通用模式门禁。
"""

import json

from vocal_subtitle.quality.golden_gate import (
    TTSGoldenThresholds,
    evaluate_tts_golden_set,
)


def _case(
    case_id,
    *,
    start_delta_ms=20.0,
    end_delta_ms=10.0,
    coverage=1.0,
    cross_merge=0,
    micro_splits=0,
    event_count=2,
):
    return {
        "id": case_id,
        "diagnostics": {
            "skeleton_priority": True,
            "skeleton_start_delta_ms": start_delta_ms,
            "skeleton_end_delta_ms": end_delta_ms,
            "skeleton_coverage_rate": coverage,
            "cross_skeleton_merge_count": cross_merge,
            "micro_pause_split_count": micro_splits,
            "event_count": event_count,
        },
    }


def test_tts_gate_reports_skeleton_metrics():
    report = evaluate_tts_golden_set(
        [
            _case("tts-1", start_delta_ms=20.0, end_delta_ms=10.0),
            _case("tts-2", start_delta_ms=40.0, end_delta_ms=30.0),
        ]
    )

    metrics = report["metrics"]
    assert metrics["case_count"] == 2
    # P50 = 两个用例的中位数。
    assert metrics["skeleton_start_delta_p50_ms"] == 30.0
    assert metrics["skeleton_start_delta_p90_ms"] >= 30.0
    assert metrics["skeleton_end_delta_p50_ms"] == 20.0
    assert metrics["cross_skeleton_merge_count"] == 0
    assert "micro_pause_split_rate" in metrics
    assert "skeleton_coverage_rate" in metrics


def test_tts_gate_passes_within_thresholds():
    report = evaluate_tts_golden_set(
        [_case("tts-1", start_delta_ms=30.0, end_delta_ms=20.0, coverage=1.0)],
        thresholds=TTSGoldenThresholds(),
    )

    assert report["status"] == "pass"
    assert report["publishable"] is True


def test_tts_gate_blocks_boundary_delta_and_coverage_failures():
    thresholds = TTSGoldenThresholds(
        max_skeleton_start_delta_p90_ms=50.0,
        max_skeleton_end_delta_p90_ms=50.0,
        min_skeleton_coverage_rate=0.98,
        max_cross_skeleton_merge_count=0,
        max_micro_pause_split_rate=0.2,
    )

    report = evaluate_tts_golden_set(
        [
            _case("tts-1", start_delta_ms=80.0, end_delta_ms=90.0, coverage=0.9),
            _case(
                "tts-2",
                start_delta_ms=120.0,
                end_delta_ms=60.0,
                coverage=0.95,
                cross_merge=1,
                micro_splits=1,
                event_count=2,
            ),
        ],
        thresholds=thresholds,
    )

    assert report["status"] == "fail"
    assert report["publishable"] is False
    checks = report["checks"]
    assert checks["skeleton_start_delta"] is False
    assert checks["skeleton_end_delta"] is False
    assert checks["skeleton_coverage"] is False
    assert checks["cross_skeleton_merge"] is False
    assert checks["micro_pause_split"] is False


def test_tts_gate_report_is_reproducible():
    cases = [_case("tts-1"), _case("tts-2", start_delta_ms=35.0)]

    first = evaluate_tts_golden_set(cases)
    second = evaluate_tts_golden_set(list(cases))

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
