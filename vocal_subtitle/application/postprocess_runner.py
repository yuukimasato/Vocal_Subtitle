"""Post-processing stage composition and final acoustic validation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..application.pipeline_result import PipelineStats
from ..mapping.time_mapper import SubtitleEvent

logger = logging.getLogger(__name__)


class PipelinePostprocessMixin:
    def _run_llm_merge(
        self,
        events: list[SubtitleEvent],
        audio: np.ndarray | None,
        sample_rate: int,
        stats: PipelineStats,
    ) -> list[SubtitleEvent]:
        """LLM 语义合并（方案五）—— 改变事件边界，必须在声学校验之前执行"""
        try:
            from ..merging.llm_merge_engine import (
                LLMMergeEngine,
            )
            from ..merging.llm_merge_engine import (
                MergeDecisionConfig as LLMMergeDecisionConfig,
            )

            self._progress.start_stage(
                "llm_merge",
                description="LLM 语义合并",
                total_items=1,
            )
            llm_merge_config = LLMMergeDecisionConfig(
                fast_merge_max_gap=self.config.merge_decision.fast_merge_max_gap,
                llm_decision_min_gap=self.config.merge_decision.llm_decision_min_gap,
                llm_decision_max_gap=self.config.merge_decision.llm_decision_max_gap,
                hard_split_min_gap=self.config.merge_decision.hard_split_min_gap,
                max_combined_duration=self.config.merge_decision.max_combined_duration,
                min_fragment_duration=self.config.merge_decision.min_fragment_duration,
                llm_tier=self.config.merge_decision.llm_tier,
                llm_model=self.config.merge_decision.llm_model,
                llm_base_url=self.config.merge_decision.llm_base_url,
                llm_api_key=self.config.merge_decision.llm_api_key,
                llm_temperature=self.config.merge_decision.llm_temperature,
                llm_timeout=self.config.merge_decision.llm_timeout,
                llm_fallback_to_rules=self.config.merge_decision.llm_fallback_to_rules,
            )
            merge_engine = LLMMergeEngine(llm_merge_config)

            fragments = []
            for i, evt in enumerate(events):
                gap = None
                gap_is_silent = None
                if i < len(events) - 1:
                    gap = events[i + 1].start - evt.end
                    gap_is_silent = gap > 0.05
                fragments.append(
                    {
                        "id": i + 1,
                        "start": evt.start,
                        "end": evt.end,
                        "text": evt.text,
                        # Use the stable numeric identity as the merge key. The
                        # display label may be changed by role labeling and must
                        # not decide whether acoustic speaker boundaries merge.
                        "speaker": (
                            str(evt.speaker_id)
                            if evt.speaker_id is not None
                            else "unknown"
                        ),
                        "physical_bin_id": getattr(evt, "physical_bin_id", None),
                        "physical_spans": list(
                            getattr(evt, "physical_spans", []) or []
                        ),
                        "gap_to_next_sec": round(gap, 3) if gap is not None else None,
                        "gap_is_silent": gap_is_silent,
                    }
                )

            merged_fragments = merge_engine.merge(
                fragments,
                audio=audio,
                sample_rate=sample_rate,
            )

            if merged_fragments and len(merged_fragments) < len(events):
                import copy

                new_events = []
                for frag in merged_fragments:
                    frag_id = frag.get("id", 0)
                    orig_idx = frag_id - 1 if frag_id > 0 else 0
                    if orig_idx < len(events):
                        member_ids = frag.get("_merged_ids", [frag_id])
                        members = [
                            events[item_id - 1]
                            for item_id in member_ids
                            if isinstance(item_id, int) and 0 < item_id <= len(events)
                        ] or [events[orig_idx]]
                        base = copy.deepcopy(members[0])
                        base.index = len(new_events) + 1
                        base.start = frag.get("start", base.start)
                        base.end = frag.get("end", base.end)
                        base.text = frag.get("text", base.text)
                        base.original_text = base.text
                        base.words = [
                            word for member in members for word in member.words
                        ]
                        base.source_word_ids = list(
                            dict.fromkeys(
                                word_id
                                for member in members
                                for word_id in member.source_word_ids
                            )
                        )
                        physical_starts = [
                            member.physical_start
                            for member in members
                            if member.physical_start is not None
                        ]
                        physical_ends = [
                            member.physical_end
                            for member in members
                            if member.physical_end is not None
                        ]
                        if physical_starts:
                            base.physical_start = min(physical_starts)
                        if physical_ends:
                            base.physical_end = max(physical_ends)
                        new_events.append(base)
                if new_events:
                    logger.info(
                        "LLM merge: %d → %d events",
                        len(events),
                        len(new_events),
                    )
                    events = new_events
                    stats.subtitle_count = len(events)

            self._progress.update_stage(
                1, extra={"detail": f"LLM 合并: {len(events)} 条字幕"}
            )
            stats.stage_timings["llm_merge"] = self._progress.finish_stage()
        except Exception as e:
            logger.warning(
                "LLM merge failed, continuing with unmerged events: %s",
                e,
            )
        return events

    def _post_process_events(
        self,
        events: list[SubtitleEvent],
        vocals_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        stats: PipelineStats,
        ffmpeg_unified_result: dict | None = None,
    ) -> list[SubtitleEvent]:
        """后处理管线（三种路径共用）

        执行顺序经过精心设计：
        0. 事件级说话人聚类 — 全局事件集合声学聚类（替代段级 diarization）
        1. 帧级无缝衔接 — 消除相邻字幕的帧级间隙（方案六）
        2. LLM 语义合并 — 改变事件边界（方案五，必须在声学校验之前）
        3. 声学标尺校验 — 对最终边界做物理骨架吸附和诊断（方案七，最终关卡）

        Args:
            events: 字幕事件列表
            vocals_path: 人声音频路径
            audio: 音频数组
            sample_rate: 采样率
            stats: 管道统计（写入阶段耗时和诊断报告）
            ffmpeg_unified_result: 复用的 ffmpeg 骨架结果（避免重复调用）

        Returns:
            后处理完成的事件列表
        """
        # ---- -1. 静音幻听碎片吸收（2026-09-13 诊断定案） ----
        # ASR 词时间戳在换人/换句边界偏早，会把下一句首音节配上"骑在
        # 静音区"的时间（实例：「得了吧」拆成静音上的「得」+ 提前截止
        # 的「了吧」）。必须在说话人归属之前吸收，否则碎片按错误边界
        # 归属说话人、合并引擎又因 unknown/跨说话人拒绝合并。
        if self.config.acoustic_validation.enabled and audio is not None:
            try:
                from ..merging.fragment_absorber import (
                    absorb_silent_fragments,
                    reanchor_word_timestamps,
                    resolve_speech_skeleton,
                )

                _skeleton = resolve_speech_skeleton(
                    vocals_path,
                    ffmpeg_unified_result,
                    self.config.acoustic_validation,
                    audio,
                    sample_rate,
                )
                if _skeleton:
                    _before = len(events)
                    events = absorb_silent_fragments(events, _skeleton)
                    if len(events) < _before:
                        stats.subtitle_count = len(events)
                        stats.quality_diagnostics["silent_fragments_absorbed"] = (
                            _before - len(events)
                        )
                # 词级时间戳能量重锚定：finalize 的显示拆行以词起点为准，
                # 句内停顿处偏早的词起点会把静音吞进下一行行首。
                reanchored = reanchor_word_timestamps(
                    events,
                    audio,
                    sample_rate,
                )
                if reanchored:
                    stats.quality_diagnostics["words_re_anchored"] = reanchored
            except Exception as e:
                logger.warning("Silent fragment absorption failed: %s", e)

        # ---- 0. 两条主线说话人融合 ----
        # 在单块/多块/骨架三种路径的事件拼接完成后统一处理，
        # 让完整音频的全局 turns 跨越所有宏观块和骨架段。
        #
        # [层1] early_turns 生效时（分离后已成功运行全局 pass）：
        # 事件标签直接来自前置全局 turns，事件级聚类退役（D5），
        # stats 的 speaker_count/diarization_backend 等字段用早前 pass
        # 的结果填充，保持下游字段契约；early_turns 关闭、全局 pass
        # 失败或标签注入异常时，完整回退到后处理事件级聚类（现状行为）。
        if self.config.diarization.enabled and events:
            early_state = getattr(self, "_early_turns_state", None)
            early_done = False
            if early_state is not None and early_state.active:
                try:
                    from ..diarization.early_turns import assign_event_speakers

                    diar_cfg = self.config.diarization
                    word_split_cfg = bool(
                        getattr(diar_cfg, "word_split_on_turn", False)
                    )
                    # 单说话人短路：归一后 turns ≤1 个说话人时跳过切分与
                    # 多说话人路径（TTS/口播素材零额外开销）。
                    single = (
                        bool(getattr(diar_cfg, "single_speaker_shortcut", True))
                        and early_state.single_speaker
                    )
                    events, early_diag = assign_event_speakers(
                        events,
                        early_state.turns,
                        word_split=word_split_cfg and not single,
                        min_part_duration=getattr(
                            diar_cfg,
                            "min_local_segment_seconds",
                            0.25,
                        ),
                        language=self._resolved_language_or_config(),
                        model_ref=early_state.model_ref,
                    )
                    stats.speaker_count = early_diag.get("speaker_count", 0)
                    stats.diarization_backend = early_state.backend
                    stats.diarization_status = "ok"
                    stats.diarization_silhouette = None
                    stats.local_speaker_split_count = early_diag.get(
                        "local_split_count", 0
                    )
                    stats.speaker_conflict_count = early_diag.get("conflict_count", 0)
                    stats.unknown_speaker_count = early_diag.get("unknown_count", 0)
                    stats.quality_diagnostics.update(
                        {
                            "embedding_model": "",
                            "embedding_status": "skipped",
                            "embedding_silhouette": None,
                            "global_model": early_state.model_ref,
                            "global_status": early_state.diagnostics.get(
                                "global_status", "ok"
                            ),
                            "global_turn_count": len(early_state.turns),
                            "local_split_count": early_diag.get("local_split_count", 0),
                            "fallback_split_count": early_diag.get(
                                "fallback_split_count", 0
                            ),
                            "overlapped_count": early_diag.get("overlapped_count", 0),
                            "conflict_count": 0,
                            "unknown_count": early_diag.get("unknown_count", 0),
                            "expected_speakers": early_state.diagnostics.get(
                                "expected_speakers"
                            ),
                            "early_turns_status": "ok",
                            "early_turn_word_split": word_split_cfg and not single,
                            "single_speaker_shortcut": single,
                            # D5:事件级聚类退役为校验诊断，不再是标签来源
                            "event_clustering": "retired_by_early_turns",
                        }
                    )
                    early_done = True
                except Exception as e:
                    logger.warning(
                        "Early turns labeling failed; falling back to "
                        "event-level fusion: %s",
                        e,
                    )
                    stats.quality_diagnostics["early_turns_fallback_reason"] = str(e)
            if not early_done:
                try:
                    from ..diarization.speaker_fusion import run_speaker_fusion

                    fusion = run_speaker_fusion(
                        events,
                        audio,
                        sample_rate,
                        self.config,
                        embedding_engine=self._get_embedding_engine(),
                        language=self._resolved_language_or_config(),
                    )
                    events = fusion.events
                    stats.speaker_count = fusion.speaker_count
                    stats.diarization_backend = fusion.backend
                    stats.diarization_status = fusion.status
                    stats.diarization_silhouette = fusion.diagnostics.get(
                        "embedding_silhouette"
                    )
                    stats.local_speaker_split_count = fusion.local_split_count
                    stats.speaker_conflict_count = fusion.conflict_count
                    stats.unknown_speaker_count = fusion.unknown_count
                    stats.quality_diagnostics.update(fusion.diagnostics)
                except Exception as e:
                    logger.warning(
                        "Speaker fusion failed; preserving unknown speakers: %s", e
                    )
                    stats.diarization_backend = "unknown"
                    stats.diarization_status = "failed"
                    stats.quality_diagnostics["speaker_fusion_error"] = str(e)

            # 事件级角色标注
            if self.config.speaker_role.enabled:
                try:
                    events = self._run_event_role_labeling(events)
                except Exception as e:
                    logger.warning("Event-level role labeling failed: %s", e)

        # ---- 1. 帧级无缝衔接（方案六） ----
        try:
            from ..merging.llm_merge_engine import apply_frame_seamless_stitching

            stitch_gap = self.config.subtitle.max_stitch_gap
            events = apply_frame_seamless_stitching(events, max_stitch_gap=stitch_gap)
        except Exception as e:
            logger.warning("Frame seamless stitching failed: %s", e)

        # ---- 2. LLM 语义合并（方案五） ----
        # 必须在声学校验之前：合并改变事件边界
        physical_bin_ids = {
            getattr(event, "physical_bin_id", None)
            for event in events
            if getattr(event, "physical_bin_id", None) is not None
        }
        skeleton_constrained = bool((ffmpeg_unified_result or {}).get("skeleton"))
        if (
            self.config.merge_decision.llm_tier != "rule_only"
            and len(events) > 1
            and not skeleton_constrained
        ):
            events = self._run_llm_merge(events, audio, sample_rate, stats)
        elif len(events) > 1 and (skeleton_constrained or physical_bin_ids):
            stats.quality_diagnostics["semantic_merge"] = {
                "status": "skipped",
                "reason": (
                    "acoustic_skeleton_boundary"
                    if skeleton_constrained
                    else "physical_bin_boundary"
                ),
                "bin_count": len(physical_bin_ids),
            }

        # ---- 3. 声学标尺校验（方案七） ----
        # 对最终边界做物理骨架吸附和诊断（最终关卡）
        if self.config.acoustic_validation.enabled:
            try:
                from ..acoustic import AcousticValidator

                self._progress.start_stage(
                    "acoustic",
                    description="声学校验",
                    total_items=1,
                )
                validator = AcousticValidator(self.config.acoustic_validation)
                events, validation_report = validator.validate(
                    events,
                    audio_path=vocals_path,
                    audio=audio,
                    sample_rate=sample_rate,
                    ffmpeg_unified_result=ffmpeg_unified_result,
                )
                health = validation_report.get("health_score")
                if health is not None:
                    logger.info(
                        "Acoustic validation health: %.1f%%",
                        health,
                    )
                    self._progress.update_stage(
                        1, extra={"detail": f"声学健康度: {health:.1f}%"}
                    )
                stats.diagnostic_report = validation_report
                stats.stage_timings["acoustic"] = self._progress.finish_stage()
            except Exception as e:
                logger.warning("Acoustic validation failed: %s", e)

        # ---- 最终去重：兜底检查所有后处理阶段可能引入的重复字幕 ----
        # 各处理阶段（帧级衔接、LLM 合并、声学校验）可能修改事件边界，
        # 重新引入重叠重复。此处做全量扫描确保输出无重复。
        try:
            from ..mapping.time_mapper import TimeMapper

            events = TimeMapper._deduplicate_overlapping(events)
            stats.subtitle_count = len(events)
        except Exception as e:
            logger.warning("Final dedup in post_process failed: %s", e)

        return events
