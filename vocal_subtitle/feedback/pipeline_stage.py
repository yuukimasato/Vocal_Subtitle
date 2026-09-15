"""Feedback learning adapter for the application pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from ..mapping.time_mapper import SubtitleEvent

logger = logging.getLogger(__name__)


class PipelineFeedbackMixin:
    def _run_feedback_learning(
        self,
        auto_events: list[SubtitleEvent],
        reference_path: Path,
        audio_path: str,
    ) -> dict | None:
        """离线反馈学习：对齐 → 分析 → 更新配置 → 构建 Few-shot

        此过程不影响当前管道的输出，异常被静默捕获。

        Args:
            auto_events: 自动生成的字幕事件
            reference_path: 用户修订的字幕文件 (.srt/.ass)
            audio_path: 音频文件路径

        Returns:
            学习报告 dict，失败时返回 None
        """
        try:
            from ..feedback import (
                AudioFingerprinter,
                DiffAnalyzer,
                FewShotBuilder,
                ParamLearner,
                SubtitleAligner,
                UserProfileManager,
            )
            from ..feedback.aligner import parse_subtitle_file
            from ..feedback.conflict_detector import ConflictDetector
            from ..feedback.health_scorer import (
                compute_health_score_from_pairs,
                should_auto_rollback,
            )

            logger.info("Feedback learning started: reference=%s", reference_path)

            # Step 1: 解析用户修订字幕
            manual_events = parse_subtitle_file(reference_path)
            if not manual_events:
                logger.warning("Feedback: no events parsed from reference file")
                return None

            # Step 2: 对齐
            feedback_cfg = self.config.feedback
            aligner = SubtitleAligner(
                min_iou=feedback_cfg.alignment_min_iou,
                min_coverage=feedback_cfg.alignment_min_coverage,
                text_weight=feedback_cfg.alignment_text_weight,
                semantic_weight=feedback_cfg.alignment_semantic_weight,
                semantic_enabled=feedback_cfg.alignment_semantic_enabled,
            )
            pairs = aligner.align(auto_events, manual_events)

            # Step 2.5: 健康度评分（调整前）
            health_before, health_detail = compute_health_score_from_pairs(pairs)

            # Step 3: 差异分析
            analyzer = DiffAnalyzer(
                param_isolation_enabled=feedback_cfg.param_isolation_enabled,
            )
            diff_report = analyzer.analyze(pairs)

            # Step 3.5: 震荡检测
            profile_mgr = UserProfileManager(feedback_cfg)
            profile = profile_mgr.load(feedback_cfg.active_profile)
            history = profile.get("history", [])
            locked_params = profile.get("locked_params", [])

            detector = ConflictDetector(
                window=feedback_cfg.oscillation_detection_window
            )
            conflicts = detector.detect_all_oscillations(history)

            # 过滤掉已锁定的参数调整
            if locked_params:
                filtered_attr = {}
                for param_path, adj in diff_report.attribution.items():
                    if param_path in locked_params:
                        logger.info(
                            "Feedback: skipping locked param '%s'",
                            param_path,
                        )
                        continue
                    filtered_attr[param_path] = adj
                diff_report.attribution = filtered_attr

            if conflicts:
                logger.warning(
                    "Feedback: %d parameter oscillations detected",
                    len(conflicts),
                )
                for cr in conflicts:
                    logger.warning(
                        "  %s: %d flips (recommend: %s)",
                        cr.param_path,
                        cr.oscillation_count,
                        cr.recommended_action,
                    )

            # Step 4: 参数学习
            current_overrides = profile.get("overrides", {})

            learner = ParamLearner(profile_mgr)
            updated_overrides = learner.learn_from_diff(
                diff_report=diff_report,
                current_config_overrides=current_overrides,
                profile_name=feedback_cfg.active_profile,
            )

            # Step 4.5: 自动回滚检查
            health_after = health_before  # 同一对齐对上的评分
            if feedback_cfg.auto_rollback_on_quality_drop and health_before > 0:
                should_rollback, rollback_reason = should_auto_rollback(
                    health_before,
                    health_after,
                    drop_threshold=feedback_cfg.quality_drop_threshold,
                )
                if should_rollback:
                    logger.warning(
                        "Feedback: auto-rollback triggered — %s",
                        rollback_reason,
                    )
                    try:
                        profile_mgr.rollback(feedback_cfg.active_profile)
                        logger.info(
                            "Feedback: rolled back profile '%s'",
                            feedback_cfg.active_profile,
                        )
                    except Exception as rb_err:
                        logger.error("Feedback: rollback failed: %s", rb_err)

            # Step 5: Few-shot 构建
            if feedback_cfg.few_shot_enabled:
                few_shot = FewShotBuilder(
                    max_examples=feedback_cfg.few_shot_max_examples
                )
                few_shot.load_cache(feedback_cfg.active_profile)
                few_shot.build_merge_examples(diff_report.merge_actions)
                if diff_report.text_edits:
                    few_shot.build_format_examples(diff_report.text_edits)
                few_shot.save_cache(feedback_cfg.active_profile)

            # Step 6: 音频指纹提取与存储
            if feedback_cfg.fingerprint_enabled and diff_report.attribution:
                try:
                    fingerprinter = AudioFingerprinter(
                        distance_method=feedback_cfg.fingerprint_distance_method,
                        knn_k=feedback_cfg.fingerprint_knn_k,
                        min_absolute_similarity=feedback_cfg.fingerprint_min_absolute_similarity,
                        relative_margin=feedback_cfg.fingerprint_relative_margin,
                    )
                    fp = fingerprinter.extract(Path(audio_path))
                    if fp is not None:
                        audio_hash = AudioFingerprinter.compute_audio_hash(
                            Path(audio_path)
                        )
                        fingerprinter.store(
                            profile_id=feedback_cfg.active_profile,
                            fingerprint=fp,
                            audio_hash=audio_hash,
                            config_snapshot=updated_overrides,
                        )
                        fingerprinter.record_feedback(
                            profile_id=feedback_cfg.active_profile,
                            audio_hash=audio_hash,
                            alignment_coverage=diff_report.alignment_coverage,
                            diff_summary="; ".join(
                                adj.reason for adj in diff_report.attribution.values()
                            ),
                            adjustments={
                                k: [adj.observed_value, adj.confidence]
                                for k, adj in diff_report.attribution.items()
                            },
                            health_before=health_before,
                            health_after=health_after,
                            health_detail=health_detail,
                        )
                        logger.info(
                            "Feedback: audio fingerprint stored — %s",
                            fp.audio_signature,
                        )
                except Exception as fp_err:
                    logger.warning(
                        "Feedback: fingerprint extraction failed (non-fatal): %s",
                        fp_err,
                    )

            # 构建学习报告
            report = {
                "alignment_coverage": round(diff_report.alignment_coverage, 3),
                "total_pairs": diff_report.total_pairs,
                "time_shifts_count": len(diff_report.time_shifts),
                "merge_actions_count": len(diff_report.merge_actions),
                "text_edits_count": len(diff_report.text_edits),
                "health_score": round(health_before, 2),
                "health_score_detail": health_detail,
                "param_adjustments": {
                    path: {
                        "direction": adj.direction,
                        "confidence": round(adj.confidence, 3),
                        "reason": adj.reason,
                    }
                    for path, adj in diff_report.attribution.items()
                },
                "structural_revision": diff_report.structural_revision,
                "oscillations_detected": len(conflicts),
            }

            logger.info(
                "Feedback learning complete: coverage=%.1f%%, health=%.1f, adjustments=%d, "
                "shifts=%d, merges=%d, edits=%d, conflicts=%d",
                diff_report.alignment_coverage * 100,
                health_before,
                len(diff_report.attribution),
                len(diff_report.time_shifts),
                len(diff_report.merge_actions),
                len(diff_report.text_edits),
                len(conflicts),
            )

            # ---- D2 自动入库 ----
            self._ingest_feedback_sample(
                auto_events=auto_events,
                manual_events=manual_events,
                alignment_coverage=diff_report.alignment_coverage,
                alignment_confidence=getattr(diff_report, "confidence", 0.8),
                consent_level=getattr(
                    self.config.feedback, "consent_level", "anonymous"
                ),
                language=getattr(self, "_resolved_language", "unknown") or "unknown",
                diff_report=diff_report,
            )

            return report

        except Exception as e:
            logger.warning("Feedback learning failed (non-fatal): %s", e)
            return None

    def _ingest_feedback_sample(
        self,
        auto_events: list[SubtitleEvent],
        manual_events: list,
        alignment_coverage: float,
        alignment_confidence: float,
        consent_level: str,
        language: str,
        diff_report,
    ) -> None:
        """将反馈学习结果自动入库到 D2 候选反馈集。

        非致命操作：入库失败不影响管道主流程。
        """
        try:
            from ..feedback.sample_manager import FeedbackSampleManager

            # 生成文本表示（SRT 格式用于哈希）
            def _events_to_srt_text(events) -> str:
                lines = []
                for i, evt in enumerate(events, 1):
                    start_ts = getattr(evt, "start", 0)
                    end_ts = getattr(evt, "end", 0)
                    text = getattr(evt, "text", "")
                    lines.append(f"{i}\n{start_ts:.3f} --> {end_ts:.3f}\n{text}\n")
                return "\n".join(lines)

            auto_text = _events_to_srt_text(auto_events)
            human_text = _events_to_srt_text(manual_events)

            # 分类编辑类型
            edit_types = {}
            if diff_report:
                if diff_report.text_edits:
                    edit_types["text_correction"] = len(diff_report.text_edits)
                if diff_report.time_shifts:
                    edit_types["time_adjustment"] = len(diff_report.time_shifts)
                if diff_report.merge_actions:
                    edit_types["format_preference"] = len(diff_report.merge_actions)
                if diff_report.structural_revision:
                    edit_types["structural_rewrite"] = 1

            mgr = FeedbackSampleManager()
            sample = mgr.ingest(
                auto_subtitle=auto_text,
                human_revision=human_text,
                alignment={
                    "method": "dtw",
                    "coverage_ratio": alignment_coverage,
                    "confidence": alignment_confidence,
                },
                consent_level=consent_level,
                language=language,
                scene="",
                audio_duration=0.0,
                audio_condition="",
                speaker_count=0,
                original_config={},
                edit_types=edit_types,
            )
            if sample:
                logger.info("D2 sample ingested: %s", sample.sample_id)
        except Exception as e:
            logger.warning("D2 sample ingestion failed (non-fatal): %s", e)
