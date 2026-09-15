from __future__ import annotations

import json
from pathlib import Path

from scripts.calibrate_quality_gate import calibrate


def test_calibration_reports_reference_metrics_and_unavailable_acoustic_gold(
    tmp_path: Path,
):
    summary_path = tmp_path / "summary.json"
    summary = {
        "manifest": "test/quality_manifest.yaml",
        "requested_asr_path": "global",
        "scenes": [
            {
                "scene": "demo",
                "success": True,
                "elapsed_sec": 10.0,
                "comparison": {
                    "statistics": {
                        "start": {"mae_ms": 100},
                        "end": {"mae_ms": 200},
                    },
                    "text_metrics": {"reference_content_coverage": 0.95},
                },
                "diagnostic": {"physical_violation_count": 0},
            }
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = calibrate(summary, summary_path)

    assert report["quality_gate"]["status"] == "calibrated"
    assert report["quality_gate"]["min_reference_content_coverage"] == 0.94
    assert report["acoustic_gate"]["status"] == "uncalibrated"
    assert "acoustic_thresholds_uncalibrated" in report["blocking_reasons"]
    assert report["publishable"] is False
