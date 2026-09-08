"""Application service for feedback alignment and adaptive learning."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Type

from ..config import ConfigLoader
from ..feedback import (
    DiffAnalyzer,
    FewShotBuilder,
    ParamLearner,
    SubtitleAligner,
    UserProfileManager,
)
from ..feedback.aligner import AlignmentError, parse_subtitle_file
from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


def _pipeline_class() -> Type[Pipeline]:
    api = sys.modules.get("vocal_subtitle.webui.api")
    return getattr(api, "Pipeline", None) or Pipeline


class FeedbackLearningService:
    """Run feedback learning without exposing pipeline internals to routes."""

    def learn(
        self,
        audio_path: Path,
        reference_path: Path,
        *,
        profile: str = "default",
        feedback_profile: str = "user_default",
        run_pipeline_first: bool = True,
        dry_run: bool = False,
        consent: str = "anonymous",
    ) -> Dict[str, Any]:
        config = ConfigLoader().load_profile(profile)
        feedback_cfg = config.feedback
        manual_events = parse_subtitle_file(reference_path)
        if not manual_events:
            return {"status": "error", "message": "未从修订文件中解析到字幕事件"}

        auto_events = []
        if run_pipeline_first:
            result = _pipeline_class()(config).run(
                input_path=audio_path,
                skip_separation=True,
            )
            auto_events = result.get("events", [])
        if not auto_events and run_pipeline_first:
            return {"status": "error", "message": "管道未生成字幕事件"}

        aligner = SubtitleAligner(
            min_iou=feedback_cfg.alignment_min_iou,
            min_coverage=feedback_cfg.alignment_min_coverage,
            text_weight=feedback_cfg.alignment_text_weight,
            semantic_weight=feedback_cfg.alignment_semantic_weight,
            semantic_enabled=feedback_cfg.alignment_semantic_enabled,
        )
        try:
            pairs = aligner.align(auto_events, manual_events)
        except AlignmentError as exc:
            return {
                "status": "error",
                "message": f"对齐失败: {exc}",
                "alignment_coverage": round(exc.coverage, 3) if exc.coverage else 0,
                "total_pairs": exc.n_matched,
                "auto_event_count": exc.n_auto,
                "manual_event_count": exc.n_manual,
                "time_shifts_count": 0,
                "merge_actions_count": 0,
                "text_edits_count": 0,
            }
        except Exception as exc:
            return {"status": "error", "message": f"对齐失败: {exc}"}

        diff_report = DiffAnalyzer(
            param_isolation_enabled=feedback_cfg.param_isolation_enabled,
        ).analyze(pairs)
        response: Dict[str, Any] = {
            "status": "ok",
            "alignment_coverage": round(diff_report.alignment_coverage, 3),
            "total_pairs": diff_report.total_pairs,
            "time_shifts_count": len(diff_report.time_shifts),
            "merge_actions_count": len(diff_report.merge_actions),
            "text_edits_count": len(diff_report.text_edits),
            "param_adjustments": {
                path: {
                    "direction": adjustment.direction,
                    "confidence": round(adjustment.confidence, 3),
                    "learn_weight": round(adjustment.learn_weight, 3),
                    "reason": adjustment.reason,
                    "param_tier": adjustment.param_tier,
                }
                for path, adjustment in diff_report.attribution.items()
            },
            "structural_revision": diff_report.structural_revision,
            "message": "",
        }

        if not dry_run and diff_report.attribution:
            profile_mgr = UserProfileManager(feedback_cfg)
            current_profile = profile_mgr.load(feedback_profile)
            learner = ParamLearner(profile_mgr)
            learner.learn_from_diff(
                diff_report=diff_report,
                current_config_overrides=current_profile.get("overrides", {}),
                profile_name=feedback_profile,
            )
            if feedback_cfg.few_shot_enabled:
                few_shot = FewShotBuilder(max_examples=feedback_cfg.few_shot_max_examples)
                few_shot.load_cache(feedback_profile)
                few_shot.build_merge_examples(diff_report.merge_actions)
                if diff_report.text_edits:
                    few_shot.build_format_examples(diff_report.text_edits)
                few_shot.save_cache(feedback_profile)
            response["message"] = f"已学习 {len(diff_report.attribution)} 个参数调整"
        elif dry_run:
            response["message"] = "[dry-run] 未实际更新配置"
        else:
            response["message"] = "无需调整参数"

        # ---- D2 自动入库 ----
        _ingest_feedback_d2_sample(
            auto_events=auto_events,
            manual_events=manual_events,
            alignment_coverage=diff_report.alignment_coverage,
            consent=consent,
            diff_report=diff_report,
        )

        return response


def _ingest_feedback_d2_sample(
    auto_events: list,
    manual_events: list,
    alignment_coverage: float,
    consent: str,
    diff_report,
) -> None:
    """将反馈学习结果自动入库到 D2 候选反馈集。非致命操作。"""
    try:
        from ..feedback.sample_manager import FeedbackSampleManager

        def _events_to_srt_text(events) -> str:
            lines = []
            for i, evt in enumerate(events, 1):
                start = getattr(evt, "start", 0)
                end = getattr(evt, "end", 0)
                text = getattr(evt, "text", "")
                lines.append(f"{i}\n{start:.3f} --> {end:.3f}\n{text}\n")
            return "\n".join(lines)

        auto_text = _events_to_srt_text(auto_events)
        human_text = _events_to_srt_text(manual_events)

        edit_types = {}
        if diff_report:
            if diff_report.text_edits:
                edit_types["text_correction"] = len(diff_report.text_edits)
            if diff_report.time_shifts:
                edit_types["time_adjustment"] = len(diff_report.time_shifts)
            if diff_report.merge_actions:
                edit_types["format_preference"] = len(diff_report.merge_actions)

        mgr = FeedbackSampleManager()
        sample = mgr.ingest(
            auto_subtitle=auto_text,
            human_revision=human_text,
            alignment={
                "method": "dtw",
                "coverage_ratio": alignment_coverage,
                "confidence": getattr(diff_report, "confidence", 0.8) if diff_report else 0.8,
            },
            consent_level=consent,
            language="unknown",
            scene="",
            audio_duration=0.0,
            audio_condition="",
            speaker_count=0,
            original_config={},
            edit_types=edit_types,
        )
        if sample:
            logger.info("D2 sample ingested via WebUI: %s", sample.sample_id)
    except Exception as e:
        logger.warning("D2 sample ingestion failed (non-fatal): %s", e)


__all__ = ["FeedbackLearningService"]
