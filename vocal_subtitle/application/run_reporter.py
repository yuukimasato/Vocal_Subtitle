"""Run report orchestration for offline pipeline runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..application.pipeline_result import PipelineStats

logger = logging.getLogger(__name__)


def _runtime_engine_status(stats: PipelineStats, evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize selected engines and review-window execution for run reports."""
    result: dict[str, dict[str, Any]] = {}
    primary = stats.selected_engine or stats.final_engine or stats.requested_engine
    if primary:
        result[primary] = {
            "lifecycle": "primary",
            "enabled": True,
            "selected": True,
            "available": stats.production_path not in {"segmented_fallback", "failed"},
            "status": "completed" if stats.production_path != "segmented_fallback" else "execution_failed",
            "windows_processed": 0,
            "windows_failed": int(stats.production_path == "segmented_fallback"),
            "reason": stats.fallback_reason or None,
        }
    for name, payload in (evidence.get("optional_engines") or {}).items():
        if not isinstance(payload, dict):
            continue
        windows = payload.get("windows") or []
        failed = sum(1 for item in windows if item.get("status") not in {"ok", "cache_hit"})
        status = str(payload.get("status", "unavailable"))
        result[str(name)] = {
            "lifecycle": "secondary" if name in {"qwen", "secondary"} else "review",
            "enabled": bool(payload.get("enabled", False)),
            "selected": name in {"qwen", "secondary"} or bool(windows),
            "available": status in {"ok", "ready", "completed", "cache_hit"},
            "status": status,
            "windows_processed": len(windows),
            "windows_failed": failed,
            "reason": payload.get("reason"),
        }
    return result


def generate_run_report(
    pipeline: Any,
    input_path: Path,
    stats: PipelineStats,
    task_id: Optional[str] = None,
    sample_rate: int = 0,
    final_subtitle_path: Optional[Path] = None,
) -> None:
    """Build and persist the unified run report."""
    try:
        from ..reporting import (
            EngineAvailabilityChecker,
            RunReportBuilder,
            build_capability_maturity,
            build_noise_shadow,
            inspect_feedback_profile,
        )
        from ..utils.session_manager import create_run_id, create_task_id

        report_builder = getattr(pipeline, "_report_builder", None)
        if report_builder is None:
            effective_task_id = task_id or create_task_id(input_path)
            effective_run_id = create_run_id(effective_task_id)
            stats.run_id = effective_run_id
            stats.task_id = effective_task_id
            report_builder = RunReportBuilder(
                run_id=effective_run_id,
                task_id=effective_task_id,
            )
            try:
                pipeline._services.attach_degradation_logger(
                    report_builder.degradation_logger
                )
            except Exception:
                pass

        report_builder.set_input(
            input_path,
            stats.duration_seconds,
            sample_rate=sample_rate,
        )

        try:
            checker = EngineAvailabilityChecker(pipeline.config)
            report_builder.set_engine_snapshot(checker.check_all())
        except Exception as exc:
            logger.warning("Engine availability check failed: %s", exc)

        report_builder.set_pipeline_path(
            mode=pipeline.config.mode,
            production_path=stats.production_path or "quality_first",
            route_version=stats.asr_route_version,
            quality_gate_version=stats.quality_gate_version,
            review_policy_version=getattr(
                pipeline.config, "review_policy_version", "review-policy-v1"
            ),
            cover_policy=getattr(
                getattr(pipeline.config, "evidence_review", None),
                "cover_policy",
                getattr(getattr(pipeline.config, "evidence_review", None), "review_policy_version", ""),
            ),
            engine_policy=getattr(
                getattr(pipeline.config, "evidence_review", None),
                "engine_policy",
                getattr(getattr(pipeline.config, "evidence_review", None), "review_policy_version", ""),
            ),
            risk_policy_version=getattr(
                pipeline.config, "risk_policy_version", "risk-policy-v1"
            ),
            decision_policy_version=getattr(
                pipeline.config, "decision_policy_version", "decision-policy-v1"
            ),
            evidence_schema_version=getattr(
                pipeline.config, "evidence_schema_version", "evidence-v1"
            ),
            golden_quality_gate_version=getattr(
                pipeline.config,
                "golden_quality_gate_version",
                "golden-quality-v1",
            ),
        )

        for stage_name, elapsed in stats.stage_timings.items():
            report_builder.set_stage(stage_name, duration=elapsed)
        report_builder.set_stage(
            "export",
            status="completed",
            extra={"formats": ["srt", "vtt", "ass"]},
        )

        evidence = stats.quality_diagnostics.get("evidence_review") or {}
        quality_diagnostics = stats.quality_diagnostics or {}
        report_builder.capability_maturity = build_capability_maturity(
            evidence=evidence,
            quality=quality_diagnostics,
            config=pipeline.config,
        )
        noise_profile = quality_diagnostics.get("noise_profile") or quality_diagnostics.get(
            "physical_noise_profile", {}
        )
        report_builder.noise_shadow = build_noise_shadow(
            noise_floor_db=(noise_profile.get("noise_floor_db") if isinstance(noise_profile, dict) else None),
            current_vad_threshold=getattr(getattr(pipeline.config, "vad", None), "threshold", None),
            current_skeleton_noise_db=getattr(
                getattr(pipeline.config, "acoustic_validation", None),
                "skeleton_noise_db",
                None,
            ),
            diagnostics=noise_profile if isinstance(noise_profile, dict) else {},
        )
        report_builder.feedback_profile = inspect_feedback_profile(pipeline.config)
        report_builder.set_engine_status(_runtime_engine_status(stats, evidence))
        if evidence:
            projection = evidence.get("physical_projection") or {}
            decision_status = (
                "completed"
                if evidence.get("status") in (None, "completed", "shadow_observed")
                else "degraded"
            )
            report_builder.set_stage(
                "decision",
                status=decision_status,
                decision_count=evidence.get("decision_count", stats.decision_count),
                actions=evidence.get(
                    "actions",
                    {"keep": 0, "replace": 0, "split": 0, "drop": 0, "unresolved": 0},
                ),
            )
            report_builder.set_stage(
                "projection",
                status=decision_status,
                event_count=projection.get("event_count", 0),
                physical_violation_count=projection.get("physical_violation_count", 0),
                cross_silence_count=projection.get("cross_silence_count", 0),
                raw_event_bypass_count=projection.get("raw_event_bypass_count", 0),
            )
        else:
            report_builder.set_stage(
                "decision", status="skipped", decision_count=stats.decision_count
            )
            report_builder.set_stage("projection", status="skipped")

        report_builder.quality_info.status = stats.quality_status
        if stats.quality_diagnostics:
            report_builder.quality_info.coverage_audit = stats.quality_diagnostics.get(
                "coverage_audit", {}
            )
            report_builder.quality_info.acoustic_report = stats.quality_diagnostics.get(
                "acoustic_report", {}
            )

        report_builder.degradation.overall_mode = (
            "degraded" if stats.fallback_reason else "full"
        )
        report_builder.degradation.fallback_category = stats.fallback_category
        report_builder.degradation.fallback_reason = stats.fallback_reason

        if stats.fallback_reason:
            report_builder.record_degradation(
                stage="pipeline",
                from_path=stats.production_path or "quality_first",
                to_path="degraded",
                reason=stats.fallback_reason,
                category=stats.fallback_category or "",
            )

        report_builder.output_info.subtitle_count = stats.subtitle_count
        report_builder.output_info.speaker_count = stats.speaker_count
        report_builder.output_info.detected_language = stats.detected_language
        report_builder.output_info.language_probability = stats.language_probability
        report_builder.output_info.total_duration_seconds = stats.duration_seconds
        report_builder.output_info.files = {
            "srt": str(final_subtitle_path) if final_subtitle_path else "",
        }
        report_builder.stage_timings = dict(stats.stage_timings)

        config_snapshot = report_builder.build_config_snapshot(pipeline.config)
        report = report_builder.build(config_snapshot=config_snapshot)
        report_builder.persist(report, config=pipeline.config)
        logger.info("Run report persisted: %s", report_builder.report_dir)
    except Exception as exc:
        logger.warning("Failed to generate run report: %s", exc)
