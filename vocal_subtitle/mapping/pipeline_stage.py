"""Mapping, export and optional language-model post-processing stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..asr.base import TranscriptionSegment
from ..application.pipeline_result import PipelineStats
from ..mapping.time_mapper import SubtitleEvent, TimeMapper
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


class PipelineMappingMixin:
    def _export_subtitles_multi_format(
        self,
        builder,
        events: List[SubtitleEvent],
        default_output_path: Path,
        output_format: str,
        session_dir: Optional[Path] = None,
        label: str = "asr",
    ) -> Dict[str, str]:
        """导出字幕为多格式（SRT / VTT / ASS）

        当 session_dir 提供时，使用标准化文件名输出所有三种格式；
        否则仅在 default_output_path 输出单一格式。

        Args:
            builder: SubtitleBuilder 实例
            events: 字幕事件列表
            default_output_path: 默认输出路径（无 session_dir 时使用）
            output_format: 默认输出格式
            session_dir: 可选的会话目录
            label: 标签 — "asr" 或 "llm"

        Returns:
            {format: path} 字典，如 {"srt": "/path/to/ASR-generated.srt", ...}
        """
        from ..utils.session_manager import ASR_FORMAT_KEYS, LLM_FORMAT_KEYS, OUTPUT_NAMES

        fmt_keys = ASR_FORMAT_KEYS if label == "asr" else LLM_FORMAT_KEYS
        result: Dict[str, str] = {}

        if session_dir:
            # 输出所有三种格式到会话目录
            for fmt in ("srt", "vtt", "ass"):
                name_key = fmt_keys[fmt]
                out_path = session_dir / OUTPUT_NAMES[name_key]
                builder.build(events, out_path, fmt=fmt)
                result[fmt] = str(out_path)
                logger.debug(
                    "Exported %s/%s: %s (%d events)",
                    label, fmt, out_path, len(events),
                )
        else:
            # 仅输出请求的格式到默认路径
            builder.build(events, default_output_path, fmt=output_format)
            result[output_format] = str(default_output_path)

        return result

    def _finalize_events(
        self,
        events: List[SubtitleEvent],
        stats: PipelineStats,
        audio_duration: Optional[float],
    ) -> List[SubtitleEvent]:
        """Finalize once before stats, API responses, cache, and export."""
        from ..mapping.finalize import finalize_subtitle_events
        from ..mapping.finalize import FinalizeConfig
        sub_cfg = getattr(self.config, "subtitle", None)

        # 合并塌缩补偿：word_split_on_turn 关闭时，横跨多个说话人 turn 的
        # 合并事件由 assign_event_speakers 整体继承主说话人（配置语义），
        # finalizer 随后按行长/时长拆出的 cue 全部继承同一标签。这里记录
        # 此类事件的范围，finalization 之后按各 cue 的主导 turn 重新归属。
        multi_spans, early_turns = self._multi_speaker_spans(events)

        result = finalize_subtitle_events(
            events,
            config=FinalizeConfig(
                min_duration=getattr(sub_cfg, "min_duration", 0.8),
                max_duration=getattr(sub_cfg, "max_duration", 5.0),
                max_chars_cjk=getattr(sub_cfg, "max_chars_cjk", 20),
                max_chars_latin=getattr(sub_cfg, "max_chars_latin", 42),
                max_lines=getattr(sub_cfg, "max_lines", 2),
            ),
            audio_duration=audio_duration if audio_duration and audio_duration > 0 else None,
        )
        if multi_spans:
            relabeled_count, distinct_speakers = self._relabel_multi_speaker_cues(
                result.events, multi_spans, early_turns,
            )
            if relabeled_count and distinct_speakers > stats.speaker_count:
                stats.speaker_count = distinct_speakers
        stats.subtitle_count = result.subtitle_count
        stats.quality_diagnostics["finalization"] = result.diagnostics
        return result.events

    def _multi_speaker_spans(self, events: list[SubtitleEvent]):
        """找出横跨多个说话人 turn 的事件范围（early_turns 生效时）。

        Returns:
            ([(start, end), ...], EarlyTurnsState) — 无可补偿范围时首项为空。
        """
        diar_cfg = getattr(self.config, "diarization", None)
        if diar_cfg is None or not getattr(diar_cfg, "enabled", True):
            return [], None
        state = getattr(self, "_early_turns_state", None)
        if state is None or not state.active or state.single_speaker:
            return [], state
        spans = []
        for event in events:
            relevant = [
                turn for turn in state.turns
                if turn.end > event.start and turn.start < event.end
            ]
            if len({turn.speaker_id for turn in relevant}) > 1:
                spans.append((float(event.start), float(event.end)))
        return spans, state

    def _relabel_multi_speaker_cues(
        self,
        cues: list[SubtitleEvent],
        spans,
        state,
    ) -> tuple:
        """按主导 turn 重新归属落在多说话人范围内的 cue（原地修改）。

        Returns:
            (relabeled_count, distinct_speaker_count)
        """
        from ..diarization.early_turns import dominant_speaker_at
        from ..diarization.speaker_fusion import _speaker_label

        language = self._resolved_language_or_config()
        relabeled = 0
        for cue in cues:
            inside = any(
                span_start - 1e-6 <= float(cue.start)
                and float(cue.end) <= span_end + 1e-6
                for span_start, span_end in spans
            )
            if not inside:
                continue
            speaker_id = dominant_speaker_at(
                state.turns, float(cue.start), float(cue.end),
            )
            if speaker_id is None or speaker_id == cue.speaker_id:
                continue
            cue.speaker_id = speaker_id
            cue.speaker_label = _speaker_label(language, speaker_id)
            cue.speaker_source = "global"
            relabeled += 1
        if relabeled:
            distinct = len({
                cue.speaker_id for cue in cues if cue.speaker_id is not None
            })
            logger.info(
                "Merged-event speaker compensation: %d cues relabeled",
                relabeled,
            )
            return relabeled, distinct
        return 0, 0

    def _run_mapping(
        self,
        asr_results: List[List[TranscriptionSegment]],
        segments: List[SpeechSegment],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        speaker_ids: Optional[List[int]] = None,
        role_names: Optional[Dict[int, str]] = None,
    ) -> List[SubtitleEvent]:
        """Stage 5: 时间轴映射"""
        sub_cfg = self.config.subtitle

        mapper = TimeMapper(
            seamless_threshold=sub_cfg.gap_handling.seamless_threshold,
            natural_pause_max=sub_cfg.gap_handling.natural_pause_max,
        )

        events = mapper.map(
            asr_results, segments,
            speaker_ids=speaker_ids,
            audio=audio, sample_rate=sample_rate,
        )

        # 应用 LLM 角色名称到事件
        if role_names and speaker_ids:
            for event in events:
                spk_id = event.speaker_id
                if spk_id is not None and spk_id in role_names:
                    event.speaker_label = role_names[spk_id]

        # 当有 diarization 结果但无 role_names 时，生成通用标签
        # e.g. speaker_id=0 → "说话人A" (zh) / "Speaker A" (en) / "話者A" (ja)
        if speaker_ids and not role_names:
            lang = self._resolved_language_or_config()
            for event in events:
                spk_id = event.speaker_id
                if spk_id is not None and event.speaker_label is None:
                    event.speaker_label = self._make_speaker_label(lang, spk_id)

        return events

    def _run_llm_optimize(
        self, events: List[SubtitleEvent]
    ) -> List[SubtitleEvent]:
        """可选的 LLM 后处理（无 API 配置时自动跳过）"""
        llm_cfg = self.config.llm_optimize

        # 无 API 配置时优雅跳过（避免无意义的网络错误）
        import os
        has_api_config = bool(
            llm_cfg.base_url
            or llm_cfg.api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        if not has_api_config:
            logger.info(
                "LLM optimization skipped: no API credentials configured. "
                "Set llm_optimize.base_url + api_key in config, "
                "or DEEPSEEK_API_KEY / OPENAI_API_KEY env variable."
            )
            return events

        try:
            from llm_subtitle_optimizer import SubtitleOptimizer

            # 构建字幕字典 {index: text} 和元数据 {index: metadata}
            subtitle_dict = {}
            event_metadata = {}
            lang = self._resolved_language_or_config()
            sorted_events = sorted(events, key=lambda e: e.index)
            for i, e in enumerate(sorted_events):
                idx_str = str(e.index)
                subtitle_dict[idx_str] = e.text

                # 构建条目元数据（说话人、时间、相邻信息）
                meta = {}
                if e.speaker_label:
                    meta["speaker"] = e.speaker_label
                elif e.speaker_id is not None:
                    meta["speaker"] = self._make_speaker_label(lang, e.speaker_id)
                meta["start"] = round(e.start, 2)
                meta["end"] = round(e.end, 2)
                # 相邻条目信息
                if i > 0:
                    prev = sorted_events[i - 1]
                    meta["gap_to_prev"] = round(e.start - prev.end, 3)
                    prev_label = prev.speaker_label
                    if not prev_label and prev.speaker_id is not None:
                        prev_label = self._make_speaker_label(lang, prev.speaker_id)
                    meta["prev_speaker"] = prev_label
                if i < len(sorted_events) - 1:
                    nxt = sorted_events[i + 1]
                    meta["gap_to_next"] = round(nxt.start - e.end, 3)
                    next_label = nxt.speaker_label
                    if not next_label and nxt.speaker_id is not None:
                        next_label = self._make_speaker_label(lang, nxt.speaker_id)
                    meta["next_speaker"] = next_label
                event_metadata[idx_str] = meta

            # 进度回调：每批优化完成后向 UI 推送绝对进度。
            # 注意回调在 worker 线程中执行，抛异常会导致该批次被
            # optimizer 静默回退原文，因此内部自行兜底。
            import threading

            progress = getattr(self, "_progress", None)
            total_chunks = max(
                1, -(-len(subtitle_dict) // max(1, llm_cfg.batch_num))
            )
            lock = threading.Lock()
            done = {"chunks": 0}

            def _on_chunk_optimized(_batch_result):
                if progress is None:
                    return
                with lock:
                    done["chunks"] += 1
                    current = done["chunks"]
                try:
                    progress.report_progress(
                        current, total_chunks,
                        {"detail": f"LLM 优化批次 {current}/{total_chunks}"},
                    )
                except Exception:
                    logger.debug("LLM progress report failed", exc_info=True)

            optimizer = self._build_safe_optimizer(
                llm_cfg, update_callback=_on_chunk_optimized,
            )

            optimized = optimizer.optimize(subtitle_dict, event_metadata)

            # 更新事件文本（保留原始文本用于前端对比）
            for event in events:
                idx_str = str(event.index)
                if idx_str in optimized:
                    event.original_text = event.text  # 保存 LLM 优化前的原始 ASR 文本
                    event.text = optimized[idx_str]   # 应用 LLM 优化后的文本

            # ★ ASR 锚定去重（方案五+）：用 ASR 原文作为 ground truth，
            # 检测 LLM 是否将其他条目的内容追加到了当前条目。
            # 覆盖同一说话人和不同说话人两种情况，以 ASR 为权威判断
            # "每个短语属于哪个条目"。
            #
            # 算法：对于被 LLM 修改过的条目 i，检查其 LLM 文本是否包含
            # 相邻条目 j 的 ASR 文本（≥4字）。如果 i 的 ASR 原文中不包含
            # 该短语，说明是 LLM 添加的 → 从 i 中移除。
            import re as _re
            # 保存 LLM 原始输出用于比对（避免循环中修改干扰检测）
            _llm_texts = {e.index: e.text for e in events}
            for i, cur in enumerate(events):
                if not cur.original_text:
                    continue  # LLM 未修改此条目，跳过

                cur_asr = cur.original_text
                cur_llm = _llm_texts[cur.index]

                # 检查所有其他条目（不限窗口，LLM 可能远距离搬运文本）
                for j in range(len(events)):
                    if j == i:
                        continue
                    other = events[j]
                    other_asr = (other.original_text or other.text).strip()
                    if len(other_asr) < 3:
                        continue

                    # 核心判断：other_asr 出现在 cur_llm 中，但不在 cur_asr 中
                    if other_asr in cur_llm and other_asr not in cur_asr:
                        logger.warning(
                            "ASR-anchored dedup: entry %d absorbed text from "
                            "entry %d — removing %r",
                            cur.index, other.index, other_asr,
                        )
                        events[i].text = events[i].text.replace(
                            other_asr, ""
                        )
                        # 清理替换产生的残留（多余标点、空格）
                        events[i].text = _re.sub(
                            r'([.。！!？?，,；;、])\s*\1+', r'\1',
                            events[i].text,
                        )
                        events[i].text = _re.sub(r'\s{2,}', ' ', events[i].text)
                        events[i].text = events[i].text.strip().rstrip(".,，。;；").strip()

            # 过滤掉被上一句吸收后清空的冗余事件
            # （LLM 将下一句内容追加到当前句末尾时，下一句会被清空）
            removed = [e for e in events if not e.text.strip()]
            if removed:
                events = [e for e in events if e.text.strip()]
                logger.info(
                    "LLM optimize: removed %d absorbed event(s): %s",
                    len(removed),
                    [e.index for e in removed],
                )

            # ★ LLM 优化后去重：LLM 可能将短事件文本合并到前面的长事件中
            # （如 "你看了没?" → "我的天. 总算上来了. 你看了没?"），
            # 但短事件自身文本未清空，空文本过滤无法捕获。
            # 此处对 LLM 优化后的文本做时间重叠 + 子串检测，移除冗余事件。
            try:
                from ..mapping.time_mapper import TimeMapper
                before = len(events)
                events = TimeMapper._deduplicate_overlapping(events)
                if len(events) < before:
                    logger.info(
                        "LLM optimize post-dedup: %d → %d events (%d removed)",
                        before, len(events), before - len(events),
                    )
            except Exception as e:
                logger.warning("LLM optimize post-dedup failed: %s", e)

            logger.info("LLM optimization complete: %d events", len(events))

        except ImportError:
            logger.info("LLM optimizer not available (dependencies not installed), skipping")
        except Exception as e:
            error_msg = str(e)
            if any(kw in error_msg.lower() for kw in (
                "api_key", "401", "authentication", "unauthorized",
            )):
                logger.info(
                    "LLM optimization skipped: API authentication failed. "
                    "Check your llm_optimize.api_key config."
                )
            else:
                logger.warning("LLM optimization failed: %s", e)

        return events

    # ------------------------------------------------------------------
    # 块处理管线（单块 + 多块共用）
    # ------------------------------------------------------------------
