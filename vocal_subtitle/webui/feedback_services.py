"""Application service for feedback alignment and adaptive learning."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Type

from ..config import ConfigLoader
from ..feedback import (
    DiffAnalyzer,
    FewShotBuilder,
    ParamLearner,
    SubtitleAligner,
    UserProfileManager,
)
from ..feedback.aligner import AlignmentError, parse_subtitle_file
from ..mapping.time_mapper import SubtitleEvent
from ..pipeline import Pipeline
from .runtime_state import state

logger = logging.getLogger(__name__)

# D20：整文件对齐率 <50% 时警告但放行（V2 external-correction 上传学习）
LOW_COVERAGE_WARN_THRESHOLD = 0.5


def _pipeline_class() -> Type[Pipeline]:
    api = sys.modules.get("vocal_subtitle.webui.api")
    return getattr(api, "Pipeline", None) or Pipeline


class TaskBaselineError(Exception):
    """task_id 引用的任务或会话产物不可用（D25）。

    status_code 供 HTTP 层映射：404=任务记录不存在，410=任务结果/会话产物已清理。
    """

    def __init__(self, message: str, status_code: int = 410):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TaskBaseline:
    """从任务历史解析出的可复用管线基线（已存事件 + 统计）"""

    task_id: str
    events: List[SubtitleEvent]
    stats: Dict[str, Any]
    session_dir: Path
    run_id: str = ""


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA256（音频上传分支的 audio_ref 依据）；文件缺失返回空串"""
    if path is None or not Path(path).exists():
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_sha256_from_session(session_dir: Optional[Path]) -> str:
    """从任务会话目录解析输入音频完整 sha256（metadata.json → 目录名前缀回退）"""
    if session_dir is None:
        return ""
    try:
        meta = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        sha = str(meta.get("input_sha256") or "")
        if sha:
            return sha
    except (OSError, ValueError):
        pass
    # metadata.json 缺失时无法给出完整 64 位 sha256；返回空串而非截断的目录名，
    # 避免下游（dataset-v1 audio_ref）误把 16 位前缀当完整哈希
    return ""


def _words_from_payload(raw: Any) -> list:
    """任务历史中 words 为 {"word","start","end","confidence"} 字典序列化，转回属性形态。"""
    if not isinstance(raw, list):
        return []
    converted = []
    for item in raw:
        if isinstance(item, dict):
            converted.append(SimpleNamespace(
                word=str(item.get("word", "")),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                confidence=item.get("confidence"),
            ))
        else:
            converted.append(item)
    return converted


def _events_from_task_result(payload_events: list) -> List[SubtitleEvent]:
    """任务历史 result_json 的序列化事件 → 对齐器可用的 SubtitleEvent。

    走 SubtitleEvent.from_dict 全量反序列化：词级时间戳、说话人溯源、物理时间轴等
    可选字段不再被静默丢弃（工单 01 基线复用的完整性要求）；个别残缺事件回退最小构造。
    """
    events: List[SubtitleEvent] = []
    for position, item in enumerate(payload_events, 1):
        try:
            event = SubtitleEvent.from_dict(item)
            event.words = _words_from_payload(item.get("words") or [])
        except (KeyError, TypeError, ValueError):
            try:
                event = SubtitleEvent(
                    index=int(item.get("index", position)),
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    text=str(item.get("text", "")),
                    original_text=item.get("original_text") or None,
                    speaker_id=item.get("speaker_id"),
                    speaker_label=item.get("speaker_label") or None,
                )
            except (TypeError, ValueError):
                logger.warning("跳过任务历史中无法解析的事件 #%d", position)
                continue
        events.append(event)
    return events


def _stats_to_dict(stats: Any) -> Dict[str, Any]:
    """Pipeline.run 返回的 stats（对象或字典）→ 字典"""
    if isinstance(stats, dict):
        return stats
    to_dict = getattr(stats, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def _metadata_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """从任务 stats 补全 D2 入库元数据（修复 F5 的 unknown/0/空）"""
    return {
        "language": str(stats.get("detected_language") or "unknown"),
        "speaker_count": int(
            stats.get("speaker_count") or stats.get("canonical_speaker_count") or 0
        ),
        "audio_duration": float(stats.get("duration_seconds") or 0.0),
    }


class FeedbackLearningService:
    """Run feedback learning without exposing pipeline internals to routes."""

    def __init__(self, task_history=None):
        # 任务历史默认取运行时状态；测试可注入替身
        self._task_history = task_history

    def _history(self):
        return self._task_history if self._task_history is not None else state.task_history

    def resolve_task_baseline(self, task_id: str) -> TaskBaseline:
        """解析 task_id 引用任务已存的管线输出作为对照基线（D25，不重跑管线）。

        Raises:
            TaskBaselineError: 任务记录不存在（404），
                或字幕事件/会话产物已清理（410，提示改走音频上传）。
        """
        record = self._history().get(task_id)
        if record is None:
            raise TaskBaselineError(
                f"任务 {task_id} 不存在或已从历史清理，无法复用其字幕事件；请改用音频上传重跑学习",
                status_code=404,
            )
        try:
            result = json.loads(record.get("result_json") or "")
        except (TypeError, ValueError):
            result = None
        events = _events_from_task_result((result or {}).get("events") or [])
        if not events:
            raise TaskBaselineError(
                f"任务 {task_id} 没有可复用的字幕事件（结果为空或已清理）；请改用音频上传重跑学习",
                status_code=410,
            )
        artifacts = (result or {}).get("artifacts") or {}
        input_path = str((result or {}).get("input_path") or artifacts.get("input") or "")
        session_dir = Path(input_path).parent if input_path else None
        if session_dir is None or not session_dir.exists():
            raise TaskBaselineError(
                f"任务 {task_id} 的会话产物已清理，无法引用学习；请改用音频上传重跑学习",
                status_code=410,
            )
        stats = (result or {}).get("stats")
        return TaskBaseline(
            task_id=task_id,
            events=events,
            stats=stats if isinstance(stats, dict) else {},
            session_dir=session_dir,
            run_id=str(record.get("run_id") or ""),
        )

    def learn(
        self,
        audio_path: Optional[Path],
        reference_path: Path,
        *,
        task_id: str = "",
        scenario: str = "",
        profile: str = "default",
        feedback_profile: str = "user_default",
        run_pipeline_first: bool = True,
        dry_run: bool = False,
        consent: str = "anonymous",
    ) -> Dict[str, Any]:
        """执行一次学习。

        基线三源分支（定案 §4）：带 task_id 时复用任务历史已存 events（秒级）；
        否则按 run_pipeline_first 用上传音频重跑管线（旧客户端路径，行为不变）。
        """
        config = ConfigLoader().load_profile(profile)
        feedback_cfg = config.feedback
        manual_events = parse_subtitle_file(reference_path)
        if not manual_events:
            return {"status": "error", "message": "未从修订文件中解析到字幕事件"}

        baseline_source = "pipeline_rerun" if run_pipeline_first else "none"
        task_stats: Dict[str, Any] = {}
        auto_events: List[SubtitleEvent] = []
        if task_id:
            baseline = self.resolve_task_baseline(task_id)
            auto_events = baseline.events
            task_stats = baseline.stats
            baseline_source = "task_history"
        elif run_pipeline_first and audio_path is not None:
            result = _pipeline_class()(config).run(
                input_path=audio_path,
                skip_separation=True,
            )
            auto_events = result.get("events", [])
            task_stats = _stats_to_dict(result.get("stats"))
        if not auto_events and run_pipeline_first and not task_id:
            return {"status": "error", "message": "管道未生成字幕事件"}

        aligner = SubtitleAligner(
            min_iou=feedback_cfg.alignment_min_iou,
            min_coverage=feedback_cfg.alignment_min_coverage,
            text_weight=feedback_cfg.alignment_text_weight,
            semantic_weight=feedback_cfg.alignment_semantic_weight,
            semantic_enabled=feedback_cfg.alignment_semantic_enabled,
        )
        try:
            # D20"低覆盖率警告但放行"仅限外部修正上传场景（Aegisub 重度重断句是常态）；
            # 其余场景维持默认 raise——传错文件/片源不符的学习应明确失败而非静默学入。
            # 未匹配行由对齐器标 INSERT/DELETE（重构行），不参与时间轴维度学习
            pairs = aligner.align(
                auto_events, manual_events,
                on_low_coverage="warn" if scenario == "external-correction" else "raise",
            )
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
            "baseline_source": baseline_source,
        }
        # D20：重构行（未匹配的人工字幕行，DELETE）与自动版多出行（INSERT）计数；
        # 两者均不参与时间轴维度学习，仅供报告呈现
        reconstructed_lines = sum(1 for p in pairs if p.match_type == "DELETE")
        inserted_lines = sum(1 for p in pairs if p.match_type == "INSERT")
        response["reconstructed_lines"] = reconstructed_lines
        response["inserted_lines"] = inserted_lines
        if diff_report.alignment_coverage < LOW_COVERAGE_WARN_THRESHOLD:
            response["coverage_warning"] = (
                f"对齐覆盖率 {diff_report.alignment_coverage:.0%} 低于 50%："
                f"修正字幕与来源任务差异过大（可能重度重断句或并非同一片源），"
                f"{reconstructed_lines} 行未匹配字幕已标为重构行、不参与时间轴维度学习；"
                f"仍允许继续学习，但请核对来源任务是否选对"
            )
        else:
            response["coverage_warning"] = ""
        if task_id:
            response["task_id"] = task_id

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

        # ---- D2 自动入库（仅确认学习；dry_run 预览不应有入库副作用） ----
        if not dry_run:
            if task_id:
                provenance = {
                    "task_id": task_id,
                    "audio_sha256": _audio_sha256_from_session(baseline.session_dir),
                    "run_id": baseline.run_id,
                }
            else:
                provenance = {
                    "task_id": "",
                    "audio_sha256": _sha256_file(audio_path) if audio_path is not None else "",
                    "run_id": str(task_stats.get("run_id") or ""),
                }
            _ingest_feedback_d2_sample(
                auto_events=auto_events,
                manual_events=manual_events,
                alignment_coverage=diff_report.alignment_coverage,
                consent=consent,
                diff_report=diff_report,
                scenario=scenario,
                metadata=_metadata_from_stats(task_stats),
                provenance=provenance,
            )

        return response


def _ingest_feedback_d2_sample(
    auto_events: list,
    manual_events: list,
    alignment_coverage: float,
    consent: str,
    diff_report,
    *,
    scenario: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, str]] = None,
) -> None:
    """将反馈学习结果自动入库到 D2 候选反馈集。非致命操作。

    scenario 为 D27 场景标签（写入样本 scene 字段）；
    metadata 从任务 stats 补全 language/speaker_count/audio_duration；
    provenance 携带 task_id/audio_sha256/run_id（dataset-v1 音频引用依据）。
    """
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

        metadata = metadata or {}
        provenance = provenance or {}
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
            language=str(metadata.get("language") or "unknown"),
            scene=str(scenario or ""),
            audio_duration=float(metadata.get("audio_duration") or 0.0),
            audio_condition="",
            speaker_count=int(metadata.get("speaker_count") or 0),
            original_config={},
            edit_types=edit_types,
            task_id=str(provenance.get("task_id") or ""),
            audio_sha256=str(provenance.get("audio_sha256") or ""),
            run_id=str(provenance.get("run_id") or ""),
        )
        if sample:
            logger.info("D2 sample ingested via WebUI: %s", sample.sample_id)
    except Exception as e:
        logger.warning("D2 sample ingestion failed (non-fatal): %s", e)


__all__ = ["FeedbackLearningService", "TaskBaseline", "TaskBaselineError"]
