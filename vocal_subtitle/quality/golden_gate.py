"""Deterministic golden-set quality gate for offline subtitle production."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .matching import (
    DEFAULT_STRICT_MIN_OVERLAP_SECONDS,
    LEGACY_MATCH_POLICY_VERSION,
    STRICT_MATCH_POLICY_VERSION,
    classify_expected_match,
)
from .provenance import attribute_case_misses, attribution_counts


@dataclass(frozen=True)
class GoldenQualityThresholds:
    max_hallucination_retention_rate: float = 0.05
    max_real_speech_drop_rate: float = 0.05
    max_physical_violation_rate: float = 0.0
    max_cross_silence_rate: float = 0.0
    max_unresolved_rate: float = 0.30
    max_split_rate: float = 0.80
    max_drop_rate: float = 0.50
    max_trace_missing_rate: float = 0.0
    max_raw_bypass_count: int = 0
    # 高精度门禁(Task 7):词级时间覆盖率下限与字幕重叠率上限。
    # 默认不设限(coverage>=0 恒真、overlap<=1 恒真),保持基线行为;
    # 高精度门禁显式传 0.95 / 0.0 启用。
    min_word_time_coverage_rate: float = 0.0
    max_subtitle_overlap_rate: float = 1.0


def _matched(
    expected: Mapping[str, Any], predicted: Sequence[Mapping[str, Any]]
) -> bool:
    return classify_expected_match(expected, predicted)["matched"]


def _rate(count: float, denominator: float) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def _range_values(item: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(item.get("start"))
        end = float(item.get("end"))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _covered_duration(
    span: Mapping[str, Any],
    cues: Sequence[Mapping[str, Any]],
) -> float:
    span_range = _range_values(span)
    if span_range is None:
        return 0.0
    start, end = span_range
    ranges = []
    for cue in cues:
        cue_range = _range_values(cue)
        if cue_range is None or not str(cue.get("text", "")).strip():
            continue
        left = max(start, cue_range[0])
        right = min(end, cue_range[1])
        if right > left:
            ranges.append((left, right))
    covered = 0.0
    current_start = current_end = None
    for left, right in sorted(ranges):
        if current_start is None:
            current_start, current_end = left, right
        elif left <= current_end:
            current_end = max(current_end, right)
        else:
            covered += current_end - current_start
            current_start, current_end = left, right
    if current_start is not None:
        covered += current_end - current_start
    return covered


def audit_case_coverage(
    physical_speech_spans: Sequence[Mapping[str, Any]] | None,
    final_cues: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Audit final cue coverage against supplied physical speech evidence."""
    spans = [
        item for item in (physical_speech_spans or ()) if isinstance(item, Mapping)
    ]
    cues = [item for item in (final_cues or ()) if isinstance(item, Mapping)]
    valid_spans = [item for item in spans if _range_values(item) is not None]
    if not valid_spans:
        return {
            "status": "not_evaluable",
            "reason": "physical_speech_spans_missing",
            "physical_speech_span_count": 0,
            "final_cue_count": len(cues),
            "audible_blank_rate": None,
            "trailing_silence_ms": None,
        }

    total_duration = sum(
        _range_values(item)[1] - _range_values(item)[0] for item in valid_spans
    )
    covered_duration = sum(_covered_duration(item, cues) for item in valid_spans)
    last_speech_end = max(_range_values(item)[1] for item in valid_spans)
    cue_ends = [
        _range_values(item)[1]
        for item in cues
        if _range_values(item) is not None and str(item.get("text", "")).strip()
    ]
    last_cue_end = max(cue_ends, default=None)
    trailing_silence_ms = (
        round(max(0.0, last_cue_end - last_speech_end) * 1000.0, 3)
        if last_cue_end is not None
        else 0.0
    )
    blank_duration = max(0.0, total_duration - covered_duration)
    return {
        "status": "pass" if blank_duration == 0.0 else "fail",
        "physical_speech_span_count": len(valid_spans),
        "covered_physical_speech_span_count": sum(
            int(_covered_duration(item, cues) > 0.0) for item in valid_spans
        ),
        "final_cue_count": len(cues),
        "speech_duration_seconds": round(total_duration, 6),
        "blank_duration_seconds": round(blank_duration, 6),
        "audible_blank_rate": round(blank_duration / total_duration, 6)
        if total_duration
        else 0.0,
        "last_physical_speech_end": last_speech_end,
        "last_final_cue_end": last_cue_end,
        "trailing_silence_ms": trailing_silence_ms,
    }


def evaluate_golden_set(
    cases: Iterable[Mapping[str, Any]],
    *,
    thresholds: GoldenQualityThresholds | None = None,
    required_categories: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    gate_mode: str = "safety",
    strict_min_overlap_seconds: float = DEFAULT_STRICT_MIN_OVERLAP_SECONDS,
    strict_min_overlap_ratio: float = 0.0,
) -> dict[str, Any]:
    """Evaluate normalized cases with separate safety and reference gates.

    A case contains ``expected_events``, ``predicted_events``, optional
    ``decisions`` and ``diagnostics``, plus free-form ``categories``.  The
    format intentionally accepts serialized ``SubtitleEvent`` and decision
    dictionaries so reports can be generated by either the pipeline or tests.
    ``reference_role=advisory`` keeps human subtitle comparisons in the
    report without making speech recall a default release blocker.
    """
    if gate_mode not in {"safety", "strict-reference"}:
        raise ValueError("gate_mode must be 'safety' or 'strict-reference'")
    if strict_min_overlap_seconds < 0:
        raise ValueError("strict_min_overlap_seconds must be non-negative")
    if not 0.0 <= strict_min_overlap_ratio <= 1.0:
        raise ValueError("strict_min_overlap_ratio must be between 0 and 1")
    policy = thresholds or GoldenQualityThresholds()
    normalized_cases = [dict(case) for case in cases]
    hallucination_total = hallucination_retained = 0
    speech_total = speech_dropped = 0
    physical_violations = cross_silence = 0
    event_total = 0
    # Rates must use every decision as the denominator. Excluding keep and
    # replace makes an unavailable review path look worse than it is and
    # makes action rates incomparable across engine outputs.
    action_counts = {
        "keep": 0,
        "replace": 0,
        "unresolved": 0,
        "split": 0,
        "drop": 0,
    }
    trace_missing = 0
    raw_bypass = 0
    word_total = 0
    word_timed = 0
    overlapping_events = 0
    missing_diagnostics: dict[str, int] = {}
    categories: set[str] = set()
    case_summaries: list[dict[str, Any]] = []
    miss_attribution_counts: dict[str, int] = {}
    strict_miss_attribution_counts: dict[str, int] = {}
    strict_speech_dropped = 0
    strict_match_only_failure_count = 0
    detailed_miss_attribution_counts: dict[str, int] = {}
    advisory_case_count = 0
    strict_reference_case_count = 0
    coverage_case_counts = {"pass": 0, "fail": 0, "not_evaluable": 0}
    coverage_blank_duration = 0.0
    coverage_speech_duration = 0.0
    trailing_silence_values: list[float] = []
    engine_status_missing_case_count = 0
    engine_status_summary: dict[str, dict[str, int]] = {}
    global_evidence_summary = {
        "signal_count": 0,
        "considered_alternative_count": 0,
        "accepted_alternative_count": 0,
        "selected_global_count": 0,
        "rejected_count": 0,
    }

    for case in normalized_cases:
        expected = [
            item
            for item in case.get("expected_events", ())
            if isinstance(item, Mapping)
        ]
        predicted = [
            item
            for item in case.get("predicted_events", ())
            if isinstance(item, Mapping)
        ]
        reference_role = str(case.get("reference_role", "")).casefold()
        reference_status = str(case.get("reference_status", "")).casefold()
        has_reference = (
            reference_role not in {"none", "safety_only"}
            and reference_status != "no_manual_reference"
        )
        if has_reference:
            advisory_case_count += 1
            if reference_role == "strict":
                strict_reference_case_count += 1
        event_total += len(predicted)
        categories.update(
            str(item) for item in case.get("categories", ()) if str(item).strip()
        )
        case_misses: list[dict[str, Any]] = []
        strict_case_misses: list[dict[str, Any]] = []
        for item in expected:
            kind = str(item.get("kind", "speech")).casefold()
            if kind in {"non_speech", "non-speech", "hallucination", "noise"}:
                hallucination_total += 1
                hallucination_retained += int(_matched(item, predicted))
            elif has_reference:
                speech_total += 1
                match = classify_expected_match(item, predicted)
                strict_match = classify_expected_match(
                    item,
                    predicted,
                    min_overlap_seconds=strict_min_overlap_seconds,
                    min_overlap_ratio=strict_min_overlap_ratio,
                    match_policy=STRICT_MATCH_POLICY_VERSION,
                )
                if not strict_match["matched"]:
                    strict_speech_dropped += 1
                    strict_stage = str(strict_match["stage"])
                    strict_miss_attribution_counts[strict_stage] = (
                        strict_miss_attribution_counts.get(strict_stage, 0) + 1
                    )
                    strict_case_misses.append(strict_match)
                if match["matched"] and not strict_match["matched"]:
                    strict_match_only_failure_count += 1
                if not match["matched"]:
                    speech_dropped += 1
                    stage = str(match["stage"])
                    miss_attribution_counts[stage] = (
                        miss_attribution_counts.get(stage, 0) + 1
                    )
                    case_misses.append(
                        {
                            **match,
                            "legacy_match": match,
                            "strict_match": strict_match,
                        }
                    )
        detailed_case_misses = attribute_case_misses(
            expected,
            predicted,
            physical_spans=case.get("physical_speech_spans", ()) or (),
            candidates=case.get("candidate_trace", case.get("candidates", ())) or (),
            decisions=case.get("decisions", ()) or (),
            final_events=case.get("final_events", case.get("final_cues", predicted)),
        )
        for stage, count in attribution_counts(detailed_case_misses).items():
            detailed_miss_attribution_counts[stage] = (
                detailed_miss_attribution_counts.get(stage, 0) + count
            )
        diagnostic = case.get("diagnostics", {})
        if not isinstance(diagnostic, Mapping):
            diagnostic = {}
        # 词级时间覆盖率与字幕重叠率(高精度方案 Task 7):从预测事件
        # 的词表与相邻关系直接统计,词时间缺失或非法即计入未覆盖。
        case_previous_end: float | None = None
        for predicted_item in sorted(
            predicted,
            key=lambda entry: (
                float(entry.get("start", 0.0)),
                float(entry.get("end", 0.0)),
            ),
        ):
            for word in predicted_item.get("words", ()) or ():
                if not isinstance(word, Mapping):
                    continue
                word_total += 1
                word_start = word.get("start")
                word_end = word.get("end")
                if (
                    word_start is not None
                    and word_end is not None
                    and float(word_end) > float(word_start) >= 0.0
                ):
                    word_timed += 1
            item_start = float(predicted_item.get("start", 0.0))
            item_end = float(predicted_item.get("end", 0.0))
            # 完全没有词级数据的事件按一个未覆盖单元计入,防止
            # "整批无词事件"在覆盖率分母中被静默排除。
            case_word_count = sum(
                1
                for word in predicted_item.get("words", ()) or ()
                if isinstance(word, Mapping)
            )
            if case_word_count == 0:
                word_total += 1
            if case_previous_end is not None and item_start < case_previous_end - 1e-9:
                overlapping_events += 1
            case_previous_end = (
                item_end
                if case_previous_end is None
                else max(case_previous_end, item_end)
            )
        for key in (
            "physical_violation_count",
            "cross_silence_count",
            "raw_event_bypass_count",
        ):
            value = diagnostic.get(key)
            if value is None:
                missing_diagnostics[key] = missing_diagnostics.get(key, 0) + 1
        physical_violations += int(diagnostic.get("physical_violation_count", 0) or 0)
        cross_silence += int(
            diagnostic.get(
                "cross_silence_count", diagnostic.get("cross_silence_events", 0)
            )
            or 0
        )
        raw_bypass += int(
            diagnostic.get(
                "raw_event_bypass_count", diagnostic.get("global_event_bypass_count", 0)
            )
            or 0
        )
        global_diagnostic = diagnostic.get("global_evidence", {})
        if isinstance(global_diagnostic, Mapping):
            for key in global_evidence_summary:
                global_evidence_summary[key] += int(global_diagnostic.get(key, 0) or 0)
        for decision in case.get("decisions", ()):
            if not isinstance(decision, Mapping):
                continue
            action = str(decision.get("decision", "")).casefold()
            if action in action_counts:
                action_counts[action] += 1
            if not decision.get("revision_trace"):
                trace_missing += 1
        coverage = audit_case_coverage(
            case.get("physical_speech_spans"),
            case.get("final_cues", predicted),
        )
        coverage_status = str(coverage.get("status", "not_evaluable"))
        coverage_case_counts[coverage_status] = (
            coverage_case_counts.get(coverage_status, 0) + 1
        )
        if coverage_status != "not_evaluable":
            coverage_blank_duration += float(
                coverage.get("blank_duration_seconds", 0.0) or 0.0
            )
            coverage_speech_duration += float(
                coverage.get("speech_duration_seconds", 0.0) or 0.0
            )
            trailing_silence_values.append(
                float(coverage.get("trailing_silence_ms", 0.0) or 0.0)
            )

        case_engine_status = case.get("engine_status")
        if not isinstance(case_engine_status, Mapping) or not case_engine_status:
            engine_status_missing_case_count += 1
        else:
            for engine_name, entry in case_engine_status.items():
                if not isinstance(entry, Mapping):
                    continue
                aggregate = engine_status_summary.setdefault(
                    engine_name,
                    {
                        "case_count": 0,
                        "selected_count": 0,
                        "enabled_count": 0,
                        "available_count": 0,
                        "windows_processed": 0,
                        "windows_failed": 0,
                    },
                )
                aggregate["case_count"] += 1
                aggregate["selected_count"] += int(bool(entry.get("selected")))
                aggregate["enabled_count"] += int(bool(entry.get("enabled")))
                aggregate["available_count"] += int(bool(entry.get("available")))
                aggregate["windows_processed"] += int(
                    entry.get("windows_processed", 0) or 0
                )
                aggregate["windows_failed"] += int(entry.get("windows_failed", 0) or 0)

        if case_misses or strict_case_misses or detailed_case_misses:
            case_summaries.append(
                {
                    "id": case.get("id"),
                    "miss_count": len(case_misses),
                    "miss_attribution": case_misses,
                    "strict_miss_count": len(strict_case_misses),
                    "strict_miss_attribution": strict_case_misses,
                    "detailed_miss_count": len(detailed_case_misses),
                    "detailed_miss_attribution": detailed_case_misses,
                    "engine_status": dict(
                        case_engine_status
                        or {
                            "status": "unavailable",
                            "reason": "case_engine_status_missing",
                        }
                    )
                    if isinstance(case_engine_status, Mapping)
                    else {
                        "status": "unavailable",
                        "reason": "case_engine_status_missing",
                    },
                }
            )

    decision_total = sum(action_counts.values())
    missing_categories = sorted(set(required_categories) - categories)
    reference_case_count = advisory_case_count
    no_reference_case_count = len(normalized_cases) - reference_case_count
    metrics = {
        "case_count": len(normalized_cases),
        "reference_case_count": reference_case_count,
        "advisory_case_count": advisory_case_count,
        "strict_reference_case_count": strict_reference_case_count,
        "no_reference_case_count": no_reference_case_count,
        "event_count": event_total,
        "expected_speech_count": speech_total,
        "expected_non_speech_count": hallucination_total,
        "hallucination_retention_rate": _rate(
            hallucination_retained, hallucination_total
        ),
        "real_speech_drop_rate": _rate(speech_dropped, speech_total),
        "strict_speech_drop_rate": _rate(strict_speech_dropped, speech_total),
        "strict_match_only_failure_count": strict_match_only_failure_count,
        "legacy_match_policy": LEGACY_MATCH_POLICY_VERSION,
        "strict_match_policy": STRICT_MATCH_POLICY_VERSION,
        "strict_min_overlap_seconds": strict_min_overlap_seconds,
        "strict_min_overlap_ratio": strict_min_overlap_ratio,
        "physical_violation_rate": _rate(physical_violations, event_total),
        "cross_silence_rate": _rate(cross_silence, event_total),
        "word_time_coverage_rate": _rate(word_timed, word_total),
        "subtitle_overlap_rate": _rate(overlapping_events, event_total),
        "unresolved_rate": _rate(action_counts["unresolved"], decision_total),
        "split_rate": _rate(action_counts["split"], decision_total),
        "drop_rate": _rate(action_counts["drop"], decision_total),
        "trace_missing_rate": _rate(trace_missing, decision_total),
        "raw_bypass_count": raw_bypass,
        "missing_diagnostics": missing_diagnostics,
        "action_counts": action_counts,
        "miss_attribution_counts": miss_attribution_counts,
        "strict_miss_attribution_counts": strict_miss_attribution_counts,
        "detailed_miss_attribution_counts": detailed_miss_attribution_counts,
        "coverage": {
            "status": (
                "not_evaluable"
                if not trailing_silence_values
                else "fail"
                if coverage_case_counts.get("fail", 0)
                else "pass"
            ),
            "case_counts": coverage_case_counts,
            "evaluable_case_count": len(trailing_silence_values),
            "audible_blank_rate": _rate(
                round(coverage_blank_duration, 6),
                round(coverage_speech_duration, 6),
            )
            if coverage_speech_duration
            else None,
            "trailing_silence_ms": max(trailing_silence_values, default=None),
        },
        "engine_status": engine_status_summary,
        "engine_status_missing_case_count": engine_status_missing_case_count,
        "global_evidence": global_evidence_summary,
    }
    checks = {
        "hallucination_retention": metrics["hallucination_retention_rate"]
        <= policy.max_hallucination_retention_rate,
        "real_speech_drop": metrics["real_speech_drop_rate"]
        <= policy.max_real_speech_drop_rate,
        "physical_violation": metrics["physical_violation_rate"]
        <= policy.max_physical_violation_rate,
        "cross_silence": metrics["cross_silence_rate"] <= policy.max_cross_silence_rate,
        "unresolved": metrics["unresolved_rate"] <= policy.max_unresolved_rate,
        "split": metrics["split_rate"] <= policy.max_split_rate,
        "drop": metrics["drop_rate"] <= policy.max_drop_rate,
        "trace": metrics["trace_missing_rate"] <= policy.max_trace_missing_rate,
        "raw_bypass": metrics["raw_bypass_count"] <= policy.max_raw_bypass_count,
        "word_time_coverage": metrics["word_time_coverage_rate"]
        >= policy.min_word_time_coverage_rate,
        "subtitle_overlap": metrics["subtitle_overlap_rate"]
        <= policy.max_subtitle_overlap_rate,
        "diagnostics_complete": not missing_diagnostics,
        "required_categories": not missing_categories,
    }
    safety_checks = {
        name: value for name, value in checks.items() if name != "real_speech_drop"
    }
    safety_passed = all(safety_checks.values())
    reference_checks = {"real_speech_drop": checks["real_speech_drop"]}
    reference_applicable = speech_total > 0
    reference_passed = checks["real_speech_drop"] if reference_applicable else True
    reference_status_value = (
        "pass"
        if reference_applicable and reference_passed
        else "fail"
        if reference_applicable
        else "not_applicable"
    )
    selected_passed = safety_passed and (
        reference_passed if gate_mode == "strict-reference" else True
    )
    release_status = (
        "blocked"
        if not safety_passed
        else "reference_improvement_required"
        if not reference_passed
        else "pass"
    )
    reference_quality = {
        "status": reference_status_value,
        "passed": reference_passed,
        "applicable": reference_applicable,
        "checks": reference_checks,
        "metrics": {
            "reference_case_count": reference_case_count,
            "advisory_case_count": advisory_case_count,
            "strict_reference_case_count": strict_reference_case_count,
            "expected_speech_count": speech_total,
            "real_speech_drop_rate": metrics["real_speech_drop_rate"],
            "strict_speech_drop_rate": metrics["strict_speech_drop_rate"],
            "strict_match_policy": metrics["strict_match_policy"],
            "miss_attribution_counts": miss_attribution_counts,
            "strict_miss_attribution_counts": strict_miss_attribution_counts,
        },
        "thresholds": {
            "max_real_speech_drop_rate": policy.max_real_speech_drop_rate,
        },
    }
    safety_gate = {
        "status": "pass" if safety_passed else "fail",
        "passed": safety_passed,
        "checks": safety_checks,
        "metrics": {
            "case_count": len(normalized_cases),
            "event_count": event_total,
            "expected_non_speech_count": hallucination_total,
            "hallucination_retention_rate": metrics["hallucination_retention_rate"],
            "physical_violation_rate": metrics["physical_violation_rate"],
            "cross_silence_rate": metrics["cross_silence_rate"],
            "unresolved_rate": metrics["unresolved_rate"],
            "split_rate": metrics["split_rate"],
            "drop_rate": metrics["drop_rate"],
            "trace_missing_rate": metrics["trace_missing_rate"],
            "raw_bypass_count": metrics["raw_bypass_count"],
            "missing_diagnostics": missing_diagnostics,
        },
    }
    report_metadata = dict(metadata or {})
    report_metadata.setdefault(
        "manifest_status",
        "available" if report_metadata.get("manifest_sha256") else "unavailable",
    )
    report_metadata.setdefault(
        "engine_status_status",
        "complete" if engine_status_missing_case_count == 0 else "unavailable",
    )
    report_metadata.setdefault(
        "comparability_status",
        "comparable"
        if report_metadata.get("manifest_sha256")
        and report_metadata.get("input_schema_version")
        and report_metadata.get("legacy_match_policy")
        else "metadata_incomplete",
    )
    return {
        "schema_version": "golden-quality-v2",
        "metadata": report_metadata,
        "gate_mode": gate_mode,
        "status": "pass" if selected_passed else "fail",
        "publishable": selected_passed,
        "release_status": release_status,
        "thresholds": asdict(policy),
        "required_categories": list(required_categories),
        "observed_categories": sorted(categories),
        "missing_categories": missing_categories,
        "case_summaries": case_summaries,
        "metrics": metrics,
        "coverage_quality": metrics["coverage"],
        "engine_status": metrics["engine_status"],
        "checks": checks,
        "safety_gate": safety_gate,
        "reference_quality": reference_quality,
    }


# ---------------------------------------------------------------------------
# TTS 专项黄金集门禁（高精度方案 Task 9 / 优化方案 §10.3）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTSGoldenThresholds:
    """TTS 骨架优先场景的独立门禁阈值。

    未达标只阻止 TTS profile 默认化,不参与通用模式门禁,
    也不得通过放宽通用门禁来隐藏其他场景回归。
    """

    max_skeleton_start_delta_p90_ms: float = 50.0
    max_skeleton_end_delta_p90_ms: float = 50.0
    min_skeleton_coverage_rate: float = 0.98
    max_cross_skeleton_merge_count: int = 0
    max_micro_pause_split_rate: float = 0.2


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = ratio * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def evaluate_tts_golden_set(
    cases: Iterable[Mapping[str, Any]],
    *,
    thresholds: TTSGoldenThresholds | None = None,
) -> dict[str, Any]:
    """对 TTS 骨架优先场景输出独立的质量结论。

    每个用例的 ``diagnostics`` 需要携带骨架优先运行产出的
    ``skeleton_start_delta_ms`` / ``skeleton_end_delta_ms`` /
    ``skeleton_coverage_rate`` / ``cross_skeleton_merge_count`` /
    ``micro_pause_split_count`` 字段(见 AcousticValidator
    ``_apply_skeleton_priority`` 报告)。
    """
    policy = thresholds or TTSGoldenThresholds()
    normalized = [dict(case) for case in cases]
    start_deltas: list[float] = []
    end_deltas: list[float] = []
    coverage_values: list[float] = []
    cross_merges = 0
    micro_splits = 0
    event_total = 0

    for case in normalized:
        diagnostic = case.get("diagnostics", {})
        if not isinstance(diagnostic, Mapping):
            diagnostic = {}
        start_deltas.append(
            float(diagnostic.get("skeleton_start_delta_ms", 0.0) or 0.0)
        )
        end_deltas.append(float(diagnostic.get("skeleton_end_delta_ms", 0.0) or 0.0))
        coverage = diagnostic.get("skeleton_coverage_rate")
        if coverage is not None:
            coverage_values.append(float(coverage))
        cross_merges += int(diagnostic.get("cross_skeleton_merge_count", 0) or 0)
        micro_splits += int(diagnostic.get("micro_pause_split_count", 0) or 0)
        event_total += int(diagnostic.get("event_count", 0) or 0)

    micro_rate = _rate(micro_splits, event_total)
    coverage_rate = (
        sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
    )
    metrics = {
        "case_count": len(normalized),
        "skeleton_start_delta_p50_ms": round(_percentile(start_deltas, 0.50), 3),
        "skeleton_start_delta_p90_ms": round(_percentile(start_deltas, 0.90), 3),
        "skeleton_end_delta_p50_ms": round(_percentile(end_deltas, 0.50), 3),
        "skeleton_end_delta_p90_ms": round(_percentile(end_deltas, 0.90), 3),
        "skeleton_coverage_rate": round(coverage_rate, 6),
        "cross_skeleton_merge_count": cross_merges,
        "micro_pause_split_count": micro_splits,
        "micro_pause_split_rate": micro_rate,
    }
    checks = {
        "skeleton_start_delta": (
            metrics["skeleton_start_delta_p90_ms"]
            <= policy.max_skeleton_start_delta_p90_ms
        ),
        "skeleton_end_delta": (
            metrics["skeleton_end_delta_p90_ms"] <= policy.max_skeleton_end_delta_p90_ms
        ),
        "skeleton_coverage": coverage_rate >= policy.min_skeleton_coverage_rate,
        "cross_skeleton_merge": cross_merges <= policy.max_cross_skeleton_merge_count,
        "micro_pause_split": micro_rate <= policy.max_micro_pause_split_rate,
    }
    passed = all(checks.values())
    return {
        "gate": "tts_skeleton_priority",
        "status": "pass" if passed else "fail",
        "publishable": passed,
        "metrics": metrics,
        "checks": checks,
        "thresholds": {
            "max_skeleton_start_delta_p90_ms": policy.max_skeleton_start_delta_p90_ms,
            "max_skeleton_end_delta_p90_ms": policy.max_skeleton_end_delta_p90_ms,
            "min_skeleton_coverage_rate": policy.min_skeleton_coverage_rate,
            "max_cross_skeleton_merge_count": policy.max_cross_skeleton_merge_count,
            "max_micro_pause_split_rate": policy.max_micro_pause_split_rate,
        },
    }


__all__ = [
    "DEFAULT_STRICT_MIN_OVERLAP_SECONDS",
    "GoldenQualityThresholds",
    "LEGACY_MATCH_POLICY_VERSION",
    "STRICT_MATCH_POLICY_VERSION",
    "TTSGoldenThresholds",
    "audit_case_coverage",
    "classify_expected_match",
    "evaluate_golden_set",
    "evaluate_tts_golden_set",
]
