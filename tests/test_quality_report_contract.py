"""阶段质量报告契约(2026-09-15 重构计划 Task 1)。

不加载任何可选模型,仅从 ``PipelineStats`` 提取:
- 阶段名与耗时(stage_timings);
- 降级类别与原因(fallback_category / fallback_reason);
- 质量状态与诊断(quality_status / quality_diagnostics)。
"""

import pytest

from vocal_subtitle.application.pipeline_result import PipelineStats


@pytest.fixture()
def stage_report():
    """把 PipelineStats 折叠成可序列化的阶段质量摘要。"""

    def _capture(stats: PipelineStats) -> dict:
        return {
            "stage_timings": dict(getattr(stats, "stage_timings", {}) or {}),
            "fallback_category": str(getattr(stats, "fallback_category", "") or ""),
            "fallback_reason": str(getattr(stats, "fallback_reason", "") or ""),
            "quality_status": str(getattr(stats, "quality_status", "") or ""),
            "quality_diagnostics": dict(
                getattr(stats, "quality_diagnostics", {}) or {}
            ),
        }

    return _capture


def test_stats_defaults_capture_to_clean_report(stage_report):
    report = stage_report(
        PipelineStats(input_path="in.wav", duration_seconds=1.0)
    )

    assert report["stage_timings"] == {}
    assert report["fallback_category"] == ""
    assert report["fallback_reason"] == ""
    assert report["quality_status"] == "pass"
    assert report["quality_diagnostics"] == {}


def test_stage_report_captures_timings_and_degradation(stage_report):
    stats = PipelineStats(input_path="in.wav", duration_seconds=1.0)
    stats.stage_timings = {"vad": 0.42, "asr": 3.18, "acoustic": 0.05}
    stats.fallback_category = "execution_failed"
    stats.fallback_reason = "model crashed"
    stats.quality_status = "degraded"
    stats.quality_diagnostics = {"asr_quality_gate": {"status": "degraded"}}

    report = stage_report(stats)

    assert report["stage_timings"] == {"vad": 0.42, "asr": 3.18, "acoustic": 0.05}
    assert report["fallback_category"] == "execution_failed"
    assert report["quality_status"] == "degraded"
    assert report["quality_diagnostics"]["asr_quality_gate"]["status"] == "degraded"


def test_stage_report_is_json_serializable(stage_report):
    import json

    stats = PipelineStats(input_path="in.wav", duration_seconds=1.0)
    stats.stage_timings = {"asr": 1.5}
    stats.quality_diagnostics = {"review": {"status": "ok"}}

    payload = json.dumps(stage_report(stats), ensure_ascii=False)

    assert "asr" in payload
    restored = json.loads(payload)
    assert restored["stage_timings"]["asr"] == 1.5
