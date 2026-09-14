"""阶段质量报告聚合(2026-09-15 重构计划 Task 6)。"""

import json

from vocal_subtitle.application.run_context import RunContext
from vocal_subtitle.quality.stage_report import (
    StageReport,
    StageReportAggregator,
    aggregate_run_diagnostics,
)


def test_aggregator_counts_statuses_and_elapsed():
    aggregator = StageReportAggregator()
    aggregator.add(StageReport("preflight", "ok", 0.10))
    aggregator.add(StageReport("asr", "degraded", 3.50, {"fallback": "funasr"}))
    aggregator.add(StageReport("export", "failed", 0.02))

    payload = aggregator.to_dict()

    assert payload["stage_count"] == 3
    assert payload["status_counts"] == {"ok": 1, "degraded": 1, "failed": 1}
    assert abs(payload["total_elapsed_seconds"] - 3.62) < 1e-6
    assert payload["stages"][1]["fallback"] == "funasr"


def test_aggregate_run_diagnostics_folds_context_entries():
    from pathlib import Path

    context = RunContext(input_path=Path("in.wav"))
    context.add_diagnostic("preflight", {"status": "ok", "elapsed_seconds": 0.2})
    context.add_diagnostic("asr", {
        "status": "failed", "elapsed_seconds": 1.0, "category": "asr_execution",
    })

    payload = aggregate_run_diagnostics(context)

    assert payload["stage_count"] == 2
    assert payload["status_counts"] == {"ok": 1, "failed": 1}
    asr_stage = next(item for item in payload["stages"] if item["stage"] == "asr")
    assert asr_stage["category"] == "asr_execution"
    assert asr_stage["elapsed_seconds"] == 1.0


def test_stage_reports_are_json_serializable():
    from pathlib import Path

    context = RunContext(input_path=Path("in.wav"))
    context.add_diagnostic("preflight", {"status": "ok", "elapsed_seconds": 0.1})

    payload = aggregate_run_diagnostics(context)

    restored = json.loads(json.dumps(payload, ensure_ascii=False))
    assert restored["stage_count"] == 1
    assert restored["stages"][0]["status"] == "ok"
