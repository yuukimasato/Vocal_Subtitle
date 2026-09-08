#!/usr/bin/env python3
"""Run the local offline production chain and emit golden-gate input JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODELS = ("tiny", "small", "medium", "large-v3")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_timeline import parse_subtitle
from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.quality.golden_gate import (
    LEGACY_MATCH_POLICY_VERSION,
    STRICT_MATCH_POLICY_VERSION,
    classify_expected_match,
)
from vocal_subtitle.quality.provenance import attribute_case_misses, attribution_counts


def _parse_reference(path: Path | None, reference_format: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if reference_format == "normalized_json" or path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        events = payload if isinstance(payload, list) else payload.get("events", [])
        result = [dict(item) for item in events if isinstance(item, dict)]
    else:
        result = [
            {"start": item.start, "end": item.end, "text": item.text, "kind": "speech"}
            for item in parse_subtitle(path)
        ]
    for index, item in enumerate(result, start=1):
        item.setdefault("id", f"expected:event:{index:06d}")
    return result


def _config(args: argparse.Namespace, *, language: str | None = None):
    config = ConfigLoader().load_profile(args.profile)
    normalized_language = str(language or "").casefold()
    config.asr.engine = args.engine
    config.asr.model = args.model
    config.asr.device = args.device
    # A mixed-language fixture must not be sent to Whisper with a forced
    # single-language hint.  The training fixture contains English speech
    # despite its historical Chinese scene label.
    config.asr.language = (
        language
        if normalized_language in {"zh", "en", "ja", "ko", "fr", "de", "es", "ru"}
        else None
    )
    config.asr.language_mode = (
        "mixed" if normalized_language in {"mixed", "en/zh"} else "single"
    )
    config.asr.global_asr.enabled = True
    config.asr.global_asr.routing = "segmented"
    production_mode = getattr(args, "production_mode", "authoritative")
    config.asr.global_asr.evidence_enabled = production_mode != "baseline"
    config.asr.whisper_cpp_bin = args.whisper_cpp_bin
    config.asr.whisper_cpp_model_path = args.whisper_cpp_model_path
    qwen_model_path = getattr(args, "qwen_model_path", None)
    config.asr.qwen_model_path = qwen_model_path
    config.evidence_review.qwen_model_path = qwen_model_path
    config.asr.engine_pair.enabled = not args.disable_engine_pair
    config.asr.engine_pair.primary = args.engine
    config.asr.engine_pair.secondary = args.secondary_engine
    config.asr.engine_pair.policy = args.review_policy
    config.evidence_review.enabled = production_mode != "baseline"
    config.evidence_review.authoritative_mode = production_mode == "authoritative"
    config.evidence_review.shadow_mode = production_mode == "shadow"
    config.evidence_review.context_reasr_enabled = not args.disable_context_reasr
    config.evidence_review.qwen_enabled = False
    config.evidence_review.forced_aligner_enabled = False
    config.evidence_review.sed_enabled = False
    config.evidence_review.semantic_review_enabled = False
    config.diarization.enabled = False
    config.speaker_role.enabled = False
    config.feedback.enabled = False
    return config


def _categories(scene: dict[str, Any], duration: float) -> list[str]:
    categories = list(scene.get("tags", []) or [])
    category = str(scene.get("category", "")).strip()
    if category:
        categories.append(category)
    language = str(scene.get("language", "")).casefold()
    if language == "zh":
        categories.append("chinese")
    elif language in {"en", "ja", "mixed", "en/zh"}:
        categories.append("english_or_mixed")
    if int(scene.get("speaker_count", 0) or 0) > 1:
        categories.append("multi_speaker")
    if duration >= 180.0:
        categories.append("long_audio")
    return sorted(set(categories))


def _span_payloads(spans: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in spans or ():
        if isinstance(item, dict):
            payloads.append(dict(item))
            continue
        if not all(hasattr(item, name) for name in ("start", "end")):
            continue
        payloads.append({
            "id": getattr(item, "id", None),
            "start": float(item.start),
            "end": float(item.end),
            "source": getattr(item, "source", None),
            "physical_clip_id": getattr(item, "physical_clip_id", None),
        })
    return payloads


def _diagnostics(stats: Any) -> dict[str, Any]:
    review = (stats.quality_diagnostics or {}).get("evidence_review", {})
    global_diagnostics = (stats.global_diagnostics or {}).get("evidence", {})
    projection = review.get("physical_projection") or {}
    pair = review.get("engine_pair") or {}
    route = review.get("route") or {}
    return {
        "physical_violation_count": projection.get("physical_violation_count"),
        "cross_silence_count": projection.get("cross_silence_count"),
        "raw_event_bypass_count": projection.get("raw_event_bypass_count"),
        "decision_trace_missing_count": projection.get("decision_trace_missing_count"),
        "production_path": stats.production_path,
        "review_status": stats.review_status,
        "decision_count": stats.decision_count,
        "fallback_reason": stats.fallback_reason,
        "projection": {
            "mode": projection.get("mode"),
            "decision_count": projection.get("decision_count"),
            "event_count": projection.get("event_count"),
            "rejected_word_count": len(projection.get("rejected_word_ids", []) or []),
            "coverage_complete": (projection.get("coverage") or {}).get("complete"),
        },
        "duration_seconds": stats.duration_seconds,
        "elapsed_seconds": stats.total_time,
        "selected_engine": stats.selected_engine,
        "primary_engine": pair.get("primary") or route.get("selected_engine") or stats.selected_engine,
        "primary_model": route.get("selected_model"),
        "secondary_engine": pair.get("secondary"),
        "review_policy": pair.get("policy") or review.get("review_policy"),
        "review": review.get("review") or {},
        "optional_engines": review.get("optional_engines") or {},
        "route_version": route.get("route_version") or stats.asr_route_version,
        "pair_route_version": pair.get("route_version"),
        "quality_status": stats.quality_status,
        "global_attempted": stats.global_attempted,
        "global_evidence": global_diagnostics,
        "physical_speech_spans": _span_payloads(
            (stats.quality_diagnostics or {}).get("physical_speech_spans", ())
        ),
        "coverage_audit": projection.get("coverage") or {},
    }


def _engine_status(diagnostics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Combine selected engines with the execution diagnostics for one case."""
    pair = diagnostics.get("engine_pair") or {}
    primary = diagnostics.get("primary_engine") or diagnostics.get("selected_engine")
    secondary = pair.get("secondary") or diagnostics.get("secondary_engine")
    review = diagnostics.get("review") or {}
    optional = diagnostics.get("optional_engines") or {}
    result: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        *,
        lifecycle: str,
        selected: bool,
        enabled: bool,
        status: str,
        reason: str | None = None,
        windows: list[dict[str, Any]] | None = None,
        model_path: str | None = None,
    ) -> None:
        window_items = windows or []
        failed = sum(1 for item in window_items if item.get("status") not in {"ok", "cache_hit"})
        result[name] = {
            "lifecycle": lifecycle,
            "enabled": enabled,
            "selected": selected,
            "available": status not in {"disabled", "unavailable", "model_missing", "port_missing"},
            "status": status,
            "model_path": model_path,
            "windows_processed": len(window_items),
            "windows_failed": failed,
            "reason": reason,
        }

    if primary:
        add(
            str(primary),
            lifecycle="primary",
            selected=True,
            enabled=True,
            status="completed" if diagnostics.get("production_path") not in {"segmented_fallback", "failed"} else "execution_failed",
            reason=diagnostics.get("fallback_reason"),
        )

    if secondary:
        secondary_key = str(secondary)
        secondary_diag = optional.get(secondary_key) or optional.get("qwen") or {}
        secondary_windows = secondary_diag.get("windows") or []
        secondary_status = str(secondary_diag.get("status", "unavailable"))
        if secondary_status == "disabled":
            normalized_status = "selected_but_disabled"
        elif secondary_status in {"failed", "degraded"} or any(
            item.get("status") == "failed" for item in secondary_windows
        ):
            normalized_status = "execution_failed"
        elif secondary_status in {"ok", "ready", "completed", "cache_hit"}:
            normalized_status = "completed"
        else:
            normalized_status = secondary_status
        add(
            secondary_key,
            lifecycle="secondary",
            selected=True,
            enabled=bool(secondary_diag.get("enabled", True)),
            status=normalized_status,
            reason=secondary_diag.get("reason"),
            windows=secondary_windows,
            model_path=secondary_diag.get("model_path"),
        )

    context_windows = review.get("windows") or []
    add(
        "context_reasr",
        lifecycle="review",
        selected=bool(context_windows),
        enabled=bool(context_windows) or review.get("status") not in {"disabled", "skipped"},
        status=(
            "completed" if review.get("status") in {"ok", "completed"} else
            "execution_failed" if review.get("status") in {"failed", "degraded"} else
            "port_missing" if review.get("reason") == "context_reasr_port_missing" else
            "residual_risk_gate" if review.get("reason") == "no_review_windows" else
            str(review.get("status", "unavailable"))
        ),
        reason=review.get("reason"),
        windows=context_windows,
    )
    return result


def _miss_attribution(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach gate-compatible, stage-level attribution to each miss."""
    decisions = diagnostics.get("decisions") or ()
    candidate_trace = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        candidate_ids = list(decision.get("candidate_ids", ()) or ())
        for candidate_id in candidate_ids:
            candidate_trace.append({
                "id": candidate_id,
                "start": decision.get("start"),
                "end": decision.get("end"),
                "text": decision.get("final_text", ""),
                "trace_context": {
                    "candidate_id": candidate_id,
                    "decision_id": decision.get("decision_id"),
                    "physical_span_ids": (
                        decision.get("trace_context", {}) or {}
                    ).get("physical_span_ids", []),
                },
            })
    return attribute_case_misses(
        expected,
        predicted,
        physical_spans=diagnostics.get("physical_speech_spans", []),
        candidates=candidate_trace,
        decisions=decisions,
        final_events=predicted,
    )


def _candidate_trace(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose candidate-shaped records from decision provenance."""
    result = []
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        for candidate_id in decision.get("candidate_ids", ()) or ():
            trace_context = decision.get("trace_context") or {}
            result.append({
                "id": candidate_id,
                "source": trace_context.get("source_id", "unknown"),
                "start": decision.get("start"),
                "end": decision.get("end"),
                "text": decision.get("final_text", ""),
                "decision_id": decision.get("decision_id"),
                "candidate_role": decision.get("candidate_role", "primary"),
                "trace_context": trace_context,
            })
    return result


def _physical_span_ids(event: Any) -> list[str]:
    """Flatten physical span payloads into stable id strings per the trace contract.

    ``SubtitleEvent.physical_spans`` holds span payload dicts, while
    ``trace_context.physical_span_ids`` must contain hashable id strings.
    """
    ids: list[str] = []
    for span in getattr(event, "physical_spans", ()) or ():
        if isinstance(span, dict):
            clip_id = span.get("physical_clip_id")
            if clip_id:
                ids.append(str(clip_id))
                continue
            ids.extend(
                str(evidence_id)
                for evidence_id in span.get("evidence_ids", ()) or ()
                if evidence_id
            )
        elif isinstance(span, str) and span:
            ids.append(span)
    return list(dict.fromkeys(ids))


def run_scene(scene: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    audio = REPO_ROOT / str(scene["audio"])
    ground_truth_value = scene.get("ground_truth")
    ground_truth = REPO_ROOT / str(ground_truth_value) if ground_truth_value else None
    reference_role = str(
        scene.get("reference_role", "advisory" if ground_truth else "none")
    ).casefold()
    gate_profile = str(scene.get("gate_profile", "safety")).casefold()
    subtitle_root = args.output.parent / "subtitles"
    if getattr(args, "model_matrix", False):
        subtitle_root /= args.model
    output = subtitle_root / f"{scene['name']}.ass"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = Pipeline(_config(args, language=str(scene.get("language", "")).casefold())).run(
        audio,
        output,
        output_format="ass",
        skip_separation=True,
    )
    stats = result["stats"]
    review = (stats.quality_diagnostics or {}).get("evidence_review", {})
    predicted = [
        {
            "id": f"final:event:{index:06d}",
            "start": float(item.start),
            "end": float(item.end),
            "text": item.text,
            "trace_context": {
                "final_event_ids": [f"final:event:{index:06d}"],
                "physical_span_ids": _physical_span_ids(item),
            },
        }
        for index, item in enumerate(result.get("events", []), start=1)
        if str(item.text).strip()
    ]
    diagnostics = {
        **_diagnostics(stats),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "decisions": review.get("decisions", []),
    }
    expected_events = _parse_reference(ground_truth, scene.get("ground_truth_format"))
    engine_status = _engine_status(diagnostics)
    candidate_trace = _candidate_trace(diagnostics.get("decisions", []))
    miss_attribution = _miss_attribution(
        expected_events,
        predicted,
        diagnostics,
    )
    return {
        "id": scene["name"],
        "engine": args.engine,
        "model": args.model,
        "secondary_engine": args.secondary_engine,
        "review_policy": args.review_policy,
        "production_mode": getattr(args, "production_mode", "authoritative"),
        "audio": str(audio.relative_to(REPO_ROOT)),
        "ground_truth": str(ground_truth.relative_to(REPO_ROOT)) if ground_truth else None,
        "reference_role": reference_role,
        "gate_profile": gate_profile,
        "reference_status": (
            "manual_reference" if ground_truth and reference_role != "none"
            else "no_manual_reference"
        ),
        "categories": _categories(scene, stats.duration_seconds),
        "expected_events": expected_events,
        "predicted_events": predicted,
        "final_cues": predicted,
        "physical_speech_spans": diagnostics.get("physical_speech_spans", []),
        "coverage_audit": diagnostics.get("coverage_audit", {}),
        "engine_status": engine_status,
        "decisions": review.get("decisions", []),
        "candidate_trace": candidate_trace,
        "miss_attribution": miss_attribution,
        "diagnostics": {
            **diagnostics,
            "decisions": review.get("decisions", []),
            "miss_attribution_counts": attribution_counts(miss_attribution),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("test/quality_manifest.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--engine",
        choices=("faster-whisper", "whisper-cpp", "funasr", "qwen"),
        default="faster-whisper",
        help="主引擎；正式生产建议使用 faster-whisper/funasr/qwen",
    )
    parser.add_argument(
        "--model",
        choices=PRODUCTION_MODELS,
        default="large-v3",
        help="单模型运行时使用的 ASR 模型",
    )
    parser.add_argument(
        "--model-matrix",
        action="store_true",
        help="依次运行 tiny/small/medium/large-v3 四个 faster-whisper 模型",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--qwen-model-path",
        default=os.environ.get("QWEN_ASR_MODEL_PATH"),
        help="Qwen review model path; omitted uses the shared local cache",
    )
    parser.add_argument(
        "--secondary-engine",
        choices=("auto", "faster-whisper", "funasr", "qwen", "whisper-cpp"),
        default="auto",
        help="异质副引擎，默认按语言选择",
    )
    parser.add_argument(
        "--review-policy",
        choices=("risk_only", "full_quality"),
        default="risk_only",
    )
    parser.add_argument(
        "--production-mode",
        choices=("baseline", "shadow", "authoritative"),
        default="authoritative",
        help="baseline=segmented only; shadow=review diagnostics; authoritative=projected output",
    )
    parser.add_argument(
        "--disable-engine-pair",
        action="store_true",
        help="仅用于单引擎 smoke；生产验收默认启用配对",
    )
    parser.add_argument(
        "--disable-context-reasr",
        action="store_true",
        help="仅用于单引擎 smoke；生产验收默认启用 Context Re-ASR",
    )
    parser.add_argument(
        "--whisper-cpp-bin",
        default=str(REPO_ROOT / "cache/whisper_cpp/build/bin/whisper-cli"),
    )
    parser.add_argument(
        "--whisper-cpp-model-path",
        default=str(REPO_ROOT / "cache/whisper_cpp/models/ggml-tiny.bin"),
    )
    args = parser.parse_args(argv)
    args.manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    args.output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise SystemExit("manifest must contain a non-empty scenes list")
    cases = []
    models = PRODUCTION_MODELS if args.model_matrix else (args.model,)
    if args.model_matrix and args.engine != "faster-whisper":
        raise SystemExit("--model-matrix 只适用于 faster-whisper")
    for model in models:
        args.model = model
        for scene in scenes:
            item = run_scene(scene, args)
            cases.append(item)
            print(json.dumps({
                "id": item["id"],
                "engine": item["engine"],
                "model": item["model"],
                "production_mode": item["production_mode"],
                "reference_status": item["reference_status"],
                "production_path": item["diagnostics"]["production_path"],
                "review_status": item["diagnostics"]["review_status"],
                "decision_count": item["diagnostics"]["decision_count"],
                "elapsed_seconds": item["diagnostics"]["elapsed_seconds"],
            }, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    reference_case_count = sum(
        1 for case in cases
        if str(case.get("reference_role", "")).casefold() not in {"none", "safety_only"}
        and case.get("reference_status") != "no_manual_reference"
    )
    expected_speech_count = sum(
        1 for case in cases
        if str(case.get("reference_role", "")).casefold() not in {"none", "safety_only"}
        and case.get("reference_status") != "no_manual_reference"
        for event in case.get("expected_events", ())
        if str(event.get("kind", "speech")).casefold() not in {"non_speech", "non-speech", "noise", "hallucination"}
    )
    expected_non_speech_count = sum(
        1 for case in cases
        for event in case.get("expected_events", ())
        if str(event.get("kind", "speech")).casefold() in {"non_speech", "non-speech", "noise", "hallucination"}
    )
    args.output.write_text(json.dumps({
        "schema_version": "golden-quality-input-v3",
        "generated_from": str(args.manifest.relative_to(REPO_ROOT)),
        "metadata": {
            "primary_engine": args.engine,
            "primary_model": args.model if not args.model_matrix else None,
            "models": list(models),
            "secondary_engine": args.secondary_engine,
            "review_policy": args.review_policy,
            "production_mode": args.production_mode,
            "engine_pair_enabled": not args.disable_engine_pair,
            "context_reasr_enabled": not args.disable_context_reasr,
            "qwen_model_path": args.qwen_model_path,
            "whisper_cpp_model_path": args.whisper_cpp_model_path,
            "route_version": "asr-route-v1",
            "pair_route_version": "asr-pair-v1",
            "manifest_path": str(args.manifest.relative_to(REPO_ROOT)),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "manifest_version": payload.get("schema_version", "quality-manifest-v1"),
            "manifest_case_count": len(scenes),
            "case_ids": [str(scene.get("name", "")) for scene in scenes],
            "reference_case_count": reference_case_count,
            "expected_speech_count": expected_speech_count,
            "expected_non_speech_count": expected_non_speech_count,
            "reference_role_rule": "advisory|strict|none",
            "legacy_match_policy": LEGACY_MATCH_POLICY_VERSION,
            "strict_match_policy": STRICT_MATCH_POLICY_VERSION,
            "strict_min_overlap_seconds": 0.01,
            "strict_min_overlap_ratio": 0.0,
        },
        "engine": args.engine,
        "model": args.model if not args.model_matrix else None,
        "models": list(models),
        "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
