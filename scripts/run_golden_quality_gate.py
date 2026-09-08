#!/usr/bin/env python3
"""Run the offline golden-set quality gate on a normalized JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocal_subtitle.quality.golden_gate import GoldenQualityThresholds, evaluate_golden_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON file with a cases list")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-category", action="append", default=[])
    parser.add_argument("--ci", action="store_true")
    parser.add_argument(
        "--strict-reference",
        action="store_true",
        help="研发回归模式：在安全门禁通过后额外阻断人工参考质量不达标",
    )
    parser.add_argument(
        "--strict-min-overlap-seconds",
        type=float,
        default=0.01,
        help="strict match 的最小有效重叠秒数；不改变 legacy match",
    )
    parser.add_argument(
        "--strict-overlap-ratio",
        type=float,
        default=0.0,
        help="strict match 的 expected overlap ratio 下限",
    )
    args = parser.parse_args(argv)
    payload: dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases", [])
    report = evaluate_golden_set(
        cases,
        thresholds=GoldenQualityThresholds(),
        required_categories=args.required_category,
        metadata={
            "input_schema_version": payload.get("schema_version")
            if isinstance(payload, dict)
            else None,
            "generated_from": payload.get("generated_from")
            if isinstance(payload, dict)
            else None,
            **(
                payload.get("metadata", {})
                if isinstance(payload, dict)
                and isinstance(payload.get("metadata", {}), dict)
                else {}
            ),
        },
        gate_mode="strict-reference" if args.strict_reference else "safety",
        strict_min_overlap_seconds=args.strict_min_overlap_seconds,
        strict_min_overlap_ratio=args.strict_overlap_ratio,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 3 if args.ci and not report["publishable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
