#!/usr/bin/env python3
"""Evaluate predicted word boundaries against manually reviewed acoustic gold."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Mapping


SCHEMA_VERSION = "acoustic-gold-v1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / (
    "acoustic-gold-v1.schema.json"
)


def _finite_time(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def load_word_boundaries(path: Path) -> dict[str, dict[str, Any]]:
    """Load the shared gold/prediction schema and reject ambiguous data."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported or missing schema_version")
    words = payload.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError(f"{path}: words must be a non-empty list")

    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(words):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}: words[{index}] must be an object")
        word_id = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        speaker = str(item.get("speaker", "")).strip()
        if not word_id or word_id in result:
            raise ValueError(f"{path}: duplicate or empty word id {word_id!r}")
        if not text or not speaker:
            raise ValueError(f"{path}: word {word_id} requires text and speaker")
        onset = _finite_time(item.get("onset"), f"word {word_id} onset")
        offset = _finite_time(item.get("offset"), f"word {word_id} offset")
        if offset <= onset:
            raise ValueError(f"{path}: word {word_id} must satisfy onset < offset")
        result[word_id] = {
            "id": word_id,
            "text": text,
            "speaker": speaker,
            "onset": onset,
            "offset": offset,
        }
    return result


def schema_path() -> Path:
    """Return the repository schema used by the evaluator."""
    return SCHEMA_PATH


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_boundaries(
    gold: Mapping[str, Mapping[str, Any]],
    prediction: Mapping[str, Mapping[str, Any]],
    *,
    audible_error_ms: float = 80.0,
) -> dict[str, Any]:
    """Match stable word IDs and calculate acoustic boundary errors."""
    shared_ids = [word_id for word_id in gold if word_id in prediction]
    missing_ids = [word_id for word_id in gold if word_id not in prediction]
    extra_ids = [word_id for word_id in prediction if word_id not in gold]
    if not shared_ids:
        raise ValueError("gold and prediction contain no shared word ids")

    start_errors: list[float] = []
    end_errors: list[float] = []
    cut_head = 0
    cut_tail = 0
    speaker_errors = 0
    threshold_seconds = audible_error_ms / 1000.0
    for word_id in shared_ids:
        expected = gold[word_id]
        actual = prediction[word_id]
        start_delta = float(actual["onset"]) - float(expected["onset"])
        end_delta = float(actual["offset"]) - float(expected["offset"])
        start_errors.append(abs(start_delta) * 1000.0)
        end_errors.append(abs(end_delta) * 1000.0)
        cut_head += start_delta > threshold_seconds
        cut_tail += end_delta < -threshold_seconds
        speaker_errors += str(actual["speaker"]) != str(expected["speaker"])

    count = len(shared_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "gold_word_count": len(gold),
        "prediction_word_count": len(prediction),
        "matched_word_count": count,
        "missing_word_ids": missing_ids,
        "extra_word_ids": extra_ids,
        "start_median_ms": round(median(start_errors), 3),
        "start_p95_ms": round(_percentile(start_errors, 0.95), 3),
        "end_median_ms": round(median(end_errors), 3),
        "end_p95_ms": round(_percentile(end_errors, 0.95), 3),
        "boundary_median_ms": round(median(start_errors + end_errors), 3),
        "boundary_p95_ms": round(_percentile(start_errors + end_errors, 0.95), 3),
        "cut_head_over_threshold_rate": round(cut_head / count, 6),
        "cut_tail_over_threshold_rate": round(cut_tail / count, 6),
        "speaker_error_rate": round(speaker_errors / count, 6),
        "audible_error_threshold_ms": audible_error_ms,
        "gold_word_coverage": round(count / len(gold), 6),
        "prediction_word_coverage": round(count / len(prediction), 6),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--max-median-ms", type=float, default=80.0)
    parser.add_argument("--max-p95-ms", type=float, default=160.0)
    parser.add_argument("--max-cut-rate", type=float, default=0.005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_boundaries(
            load_word_boundaries(args.gold),
            load_word_boundaries(args.prediction),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"acoustic gold evaluation failed: {exc}", file=sys.stderr)
        return 2

    reasons: list[str] = []
    if report["missing_word_ids"]:
        reasons.append("missing_gold_words")
    if report["boundary_median_ms"] > args.max_median_ms:
        reasons.append("median_boundary_error")
    if report["boundary_p95_ms"] > args.max_p95_ms:
        reasons.append("p95_boundary_error")
    if max(
        report["cut_head_over_threshold_rate"],
        report["cut_tail_over_threshold_rate"],
    ) > args.max_cut_rate:
        reasons.append("audible_cut_rate")
    report["publishable"] = not reasons
    report["reasons"] = reasons

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 3 if args.ci and reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
