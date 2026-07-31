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

        result = finalize_subtitle_events(
            events,
            config=FinalizeConfig(
                max_duration=getattr(sub_cfg, "max_duration", 5.0),
                max_chars_cjk=getattr(sub_cfg, "max_chars_cjk", 20),
                max_chars_latin=getattr(sub_cfg, "max_chars_latin", 42),
                max_lines=getattr(sub_cfg, "max_lines", 2),
            ),
            audio_duration=audio_duration if audio_duration and audio_duration > 0 else None,
        )
        stats.subtitle_count = result.subtitle_count
        stats.quality_diagnostics["finalization"] = result.diagnostics
        return result.events

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

            # Use the SubtitleOptimizer wrapper with enhanced validation
            optimizer = self._build_safe_optimizer(llm_cfg)

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
