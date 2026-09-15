#!/usr/bin/env python3
"""Calibrate conservative quality thresholds from a real-material summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import quantiles
from typing import Any


def _finite(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    points = quantiles(values, n=100, method="inclusive")
    index = max(0, min(len(points) - 1, math.ceil(probability * 100) - 1))
    return points[index]


def _diagnostic_paths(summary: dict[str, Any], summary_path: Path):
    output_root = summary_path.parent
    for scene in summary.get("scenes", []):
        output = scene.get("output")
        if not output:
            continue
        path = Path(output)
        if not path.is_absolute():
            path = output_root.parent.parent / path
        diagnostic = path.parent / "diagnostic_report.json"
        if diagnostic.is_file():
            yield scene, diagnostic


def calibrate(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    scenes = [item for item in summary.get("scenes", []) if item.get("success")]
    comparisons = [item.get("comparison") for item in scenes if item.get("comparison")]
    coverage = _finite(
        [
            item.get("text_metrics", {}).get("reference_content_coverage")
            for item in comparisons
        ]
    )
    start_mae = _finite(
        [
            item.get("statistics", {}).get("start", {}).get("mae_ms")
            for item in comparisons
        ]
    )
    end_mae = _finite(
        [
            item.get("statistics", {}).get("end", {}).get("mae_ms")
            for item in comparisons
        ]
    )
    elapsed = _finite([item.get("elapsed_sec") for item in scenes])
    physical_violations = _finite(
        [
            item.get("diagnostic", {}).get("physical_violation_count", 0)
            for item in scenes
        ]
    )

    risk_scores: list[float] = []
    evidence_reports = 0
    for _scene, diagnostic_path in _diagnostic_paths(summary, summary_path):
        try:
            payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        risk = (
            payload.get("quality_diagnostics", {})
            .get("evidence_review", {})
            .get("risk", [])
        )
        if isinstance(risk, list):
            risk_scores.extend(
                _finite([item.get("score") for item in risk if isinstance(item, dict)])
            )
            evidence_reports += bool(risk)

    # The selected values describe the observed gold-set envelope. They are
    # intentionally rounded outward so a later run is not rejected by noise.
    min_coverage = min(coverage) if coverage else None
    max_start = max(start_mae) if start_mae else None
    max_end = max(end_mae) if end_mae else None
    p95_elapsed = _percentile(elapsed, 0.95)
    quality_gate = {
        "status": "calibrated" if comparisons else "uncalibrated",
        "gold_scene_count": len(comparisons),
        "min_reference_content_coverage": (
            round(max(0.0, min_coverage - 0.01), 3)
            if min_coverage is not None
            else None
        ),
        "max_start_mae_ms": round(max_start * 1.10, 1)
        if max_start is not None
        else None,
        "max_end_mae_ms": round(max_end * 1.10, 1) if max_end is not None else None,
        "max_physical_violation_count": 0,
        "p95_latency_seconds": round(p95_elapsed * 1.25, 3)
        if p95_elapsed is not None
        else None,
    }
    risk = {
        "status": "calibrated" if risk_scores else "uncalibrated",
        "evidence_report_count": evidence_reports,
        "sample_count": len(risk_scores),
        "observed_p95": round(_percentile(risk_scores, 0.95), 6)
        if risk_scores
        else None,
        "thresholds": {
            "medium": 0.25,
            "high": 0.50,
            "critical": 0.75,
        },
        "reason": None if risk_scores else "evidence_review_diagnostics_missing",
    }
    acoustic = {
        "status": "uncalibrated",
        "reason": "manifest_has_no_acoustic_word_boundaries",
    }
    return {
        "schema_version": "quality-calibration-v1",
        "source_summary": str(summary_path),
        "source_manifest": summary.get("manifest"),
        "source_asr_path": summary.get("requested_asr_path"),
        "quality_gate": quality_gate,
        "risk_thresholds": risk,
        "acoustic_gate": acoustic,
        "observed": {
            "reference_content_coverage": coverage,
            "start_mae_ms": start_mae,
            "end_mae_ms": end_mae,
            "physical_violation_count": physical_violations,
            "elapsed_seconds": elapsed,
        },
        "publishable": quality_gate["status"] == "calibrated"
        and risk["status"] == "calibrated",
        "blocking_reasons": [
            reason
            for reason in (
                "no_reference_gold_metrics" if not comparisons else None,
                "risk_thresholds_uncalibrated" if not risk_scores else None,
                "acoustic_thresholds_uncalibrated",
            )
            if reason
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        report = calibrate(summary, args.summary)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 3 if args.ci and not report["publishable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
