"""时间轴映射核心模块

将 ASR 分段识别结果映射回全局时间轴。

核心公式:
    全局时间 = 片段偏移 + 段内 ASR 时间戳

    当词级时间戳可用时，使用首/尾词的时间戳替代段级边界，
    提升边界精度。

段间间隙处理策略:
    gap ≤ 0.2s 且间隙为真实静音 → 无缝衔接
    gap ≤ 0.2s 但间隙有能量（语音）→ 保留自然停顿
    0.2 < gap ≤ 1.0s → 保留自然停顿
    gap > 1.0s   → 段落间隔
"""

import difflib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..asr.base import TranscriptionSegment
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


def _merge_distinct_texts(first: str, second: str) -> str:
    """合并两条语义不同但时间重叠的文本。

    当同一音频区域被不同 ASR 窗口产生两条不同文本时，
    按时间顺序拼接，同时去重尾部/头部的重叠部分。

    Args:
        first: 时间较早的事件文本
        second: 时间较晚（或同时结束）的事件文本

    Returns:
        合并后的文本
    """
    first = first.strip()
    second = second.strip()
    if not first:
        return second
    if not second:
        return first

    # 检查尾部/头部重叠并去重
    # e.g. "应该不会" + "不会. 那是地平线" → "应该不会. 那是地平线"
    for overlap_len in range(min(len(first), len(second)), 1, -1):
        if first[-overlap_len:] == second[:overlap_len]:
            return first + second[overlap_len:]

    # 无重叠，直接拼接
    return first + " " + second


@dataclass
class SubtitleEvent:
    """字幕事件（单条字幕）

    Fields:
        index: 序列编号
        start: 显示开始时间（秒）
        end: 显示结束时间（秒）
        text: 显示文本
        words: 词级时间戳列表
        original_text: LLM 优化前的原始文本
        speaker_id: 说话人编号
        speaker_label: 说话人标签

        # 双时间轴字段（物理证据 vs 显示时间）
        physical_start: 冻结的人声证据起始时间（秒），语义阶段后不可变
        physical_end: 冻结的人声证据结束时间（秒），语义阶段后不可变
        physical_spans: 声学证据跨度列表 (SpeechEvidenceSpan)
        source_word_ids: 来源词 ID 列表
        logical_sentence_id: 逻辑句 ID
        alignment_warning: 对齐警告信息

        # 物理归属
        physical_region_id: 物理区域 ID
        physical_bin_id: 物理字幕仓 ID
        physical_bin_start: 物理字幕仓起始时间
        physical_bin_end: 物理字幕仓结束时间
        time_source: 时间来源标识
        hard_split_before: 前方是否硬拆分

        # 说话人溯源
        speaker_status: 说话人状态 (confirmed/unknown/inferred)
        speaker_source: 说话人来源
        speaker_repair_reason: 说话人修复原因
        asr_text: ASR 原始文本（覆盖前）

        # 重叠对白
        genuine_overlap: 是否真实重叠
        overlap_group_id: 重叠组 ID
        overlap_tracks: 重叠轨道列表

        # 修订追溯
        revision_trace: 修订记录列表
    """

    index: int
    start: float  # 显示时间（秒）
    end: float  # 显示时间（秒）
    text: str
    words: List = field(default_factory=list)
    original_text: Optional[str] = None
    speaker_id: Optional[int] = None
    speaker_label: Optional[str] = None

    # 双时间轴
    physical_start: Optional[float] = None
    physical_end: Optional[float] = None
    physical_spans: List = field(default_factory=list)
    source_word_ids: List[str] = field(default_factory=list)
    logical_sentence_id: Optional[str] = None
    alignment_warning: Optional[str] = None

    # 物理归属
    physical_region_id: Optional[str] = None
    physical_bin_id: Optional[str] = None
    physical_bin_start: Optional[float] = None
    physical_bin_end: Optional[float] = None
    time_source: str = ""
    hard_split_before: bool = False

    # 说话人溯源
    speaker_status: str = ""
    speaker_source: str = ""
    speaker_repair_reason: str = ""
    asr_text: Optional[str] = None

    # 重叠对白
    genuine_overlap: bool = False
    overlap_group_id: Optional[str] = None
    overlap_tracks: List = field(default_factory=list)

    # 修订追溯
    revision_trace: List = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        """序列化为字典，复制可变集合字段避免共享引用。"""
        import copy
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": copy.deepcopy(self.words),
            "original_text": self.original_text,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "physical_start": self.physical_start,
            "physical_end": self.physical_end,
            "physical_spans": copy.deepcopy(self.physical_spans),
            "source_word_ids": list(self.source_word_ids),
            "logical_sentence_id": self.logical_sentence_id,
            "alignment_warning": self.alignment_warning,
            "physical_region_id": self.physical_region_id,
            "physical_bin_id": self.physical_bin_id,
            "physical_bin_start": self.physical_bin_start,
            "physical_bin_end": self.physical_bin_end,
            "time_source": self.time_source,
            "hard_split_before": self.hard_split_before,
            "speaker_status": self.speaker_status,
            "speaker_source": self.speaker_source,
            "speaker_repair_reason": self.speaker_repair_reason,
            "asr_text": self.asr_text,
            "genuine_overlap": self.genuine_overlap,
            "overlap_group_id": self.overlap_group_id,
            "overlap_tracks": copy.deepcopy(self.overlap_tracks),
            "revision_trace": copy.deepcopy(self.revision_trace),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SubtitleEvent":
        """从字典反序列化，处理缺失字段的默认值。"""
        return cls(
            index=payload["index"],
            start=payload["start"],
            end=payload["end"],
            text=payload["text"],
            words=payload.get("words", []),
            original_text=payload.get("original_text"),
            speaker_id=payload.get("speaker_id"),
            speaker_label=payload.get("speaker_label"),
            physical_start=payload.get("physical_start"),
            physical_end=payload.get("physical_end"),
            physical_spans=payload.get("physical_spans", []),
            source_word_ids=payload.get("source_word_ids", []),
            logical_sentence_id=payload.get("logical_sentence_id"),
            alignment_warning=payload.get("alignment_warning"),
            physical_region_id=payload.get("physical_region_id"),
            physical_bin_id=payload.get("physical_bin_id"),
            physical_bin_start=payload.get("physical_bin_start"),
            physical_bin_end=payload.get("physical_bin_end"),
            time_source=payload.get("time_source", ""),
            hard_split_before=payload.get("hard_split_before", False),
            speaker_status=payload.get("speaker_status", ""),
            speaker_source=payload.get("speaker_source", ""),
            speaker_repair_reason=payload.get("speaker_repair_reason", ""),
            asr_text=payload.get("asr_text"),
            genuine_overlap=payload.get("genuine_overlap", False),
            overlap_group_id=payload.get("overlap_group_id"),
            overlap_tracks=payload.get("overlap_tracks", []),
            revision_trace=payload.get("revision_trace", []),
        )

    def __repr__(self) -> str:
        spk = f", speaker={self.speaker_label}" if self.speaker_label else ""
        return (
            f"SubtitleEvent(idx={self.index}, "
            f"start={self.start:.3f}, end={self.end:.3f}, "
            f"text='{self.text[:40]}...'{spk})"
        )


class TimeMapper:
    """时间轴映射器

    将分段识别的 ASR 结果映射到原始音频的全局时间轴上。

    使用示例:
        mapper = TimeMapper()
        events = mapper.map(
            asr_results=[...],          # 各段的 ASR 结果列表
            segment_offsets=[0.0, 3.2, 8.1, ...],  # 各段起始偏移
        )
    """

    def __init__(
        self,
        seamless_threshold: float = 0.2,
        natural_pause_max: float = 1.0,
    ):
        """
        Args:
            seamless_threshold: 无缝衔接阈值（秒），gap ≤ 此值时无缝
            natural_pause_max: 自然停顿最大间隔（秒）
        """
        self.seamless_threshold = seamless_threshold
        self.natural_pause_max = natural_pause_max

    def map(
        self,
        asr_segments_list: List[List[TranscriptionSegment]],
        speech_segments: List[SpeechSegment],
        speaker_ids: Optional[List[int]] = None,
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[SubtitleEvent]:
        """将分段 ASR 结果映射到全局时间轴

        Args:
            asr_segments_list: 每个语音片段的 ASR 识别结果列表
                e.g. [[TranscriptionSegment, ...], [...], ...]
            speech_segments: 对应的语音片段（含全局起始时间）
                e.g. [SpeechSegment(start=0.0, end=3.2), ...]
            speaker_ids: 每个片段对应的说话人编号（可选）
                e.g. [0, 1, 0, 0, 1, ...]
            audio: 原始音频数据（可选，用于间隙静音验证）
            sample_rate: 采样率

        Returns:
            全局时间轴上的字幕事件列表
        """
        if len(asr_segments_list) != len(speech_segments):
            raise ValueError(
                f"Mismatch: {len(asr_segments_list)} ASR result groups "
                f"vs {len(speech_segments)} speech segments"
            )

        all_events: List[SubtitleEvent] = []
        event_index = 0

        for seg_idx, (asr_segments, speech_seg) in enumerate(
            zip(asr_segments_list, speech_segments)
        ):
            offset = speech_seg.start  # 该片段在原始音频中的起始偏移
            spk_id = speaker_ids[seg_idx] if speaker_ids else None

            asr_count = len(asr_segments)

            for asr_idx, asr_seg in enumerate(asr_segments):
                # ---- 核心公式: 全局时间 = 片段偏移 + 段内 ASR 时间戳 ----
                # 优先使用词级时间戳（更精确的起止点）
                if asr_seg.words and len(asr_seg.words) > 0:
                    # 用首词的 start 作为全局起始（比段级 start 更精确）
                    global_start = offset + asr_seg.words[0].start
                    # 用末词的 end 作为全局结束
                    global_end = offset + asr_seg.words[-1].end
                else:
                    # 回退到段级时间戳
                    global_start = offset + asr_seg.start
                    global_end = offset + asr_seg.end

                # 末尾段：end 锚定到声学边界（Stage 2/2.5 VAD+ffmpeg）
                # ASR 时间戳对结束位置可能严重高估或低估，不可依赖
                if asr_idx == asr_count - 1:
                    global_end = speech_seg.end

                # 边界检查：不超出语音段范围（含小容差）
                _clamp_ceiling = max(
                    speech_seg.end, global_end - 0.05
                ) + self.seamless_threshold
                global_end = min(global_end, _clamp_ceiling)

                if global_end <= global_start:
                    continue

                event_index += 1
                all_events.append(
                    SubtitleEvent(
                        index=event_index,
                        start=global_start,
                        end=global_end,
                        text=asr_seg.text.strip(),
                        words=asr_seg.words,
                        speaker_id=spk_id,
                    )
                )

        # 处理段间间隙（含能量验证）
        all_events = self._merge_gaps(all_events, audio, sample_rate)

        # 重叠/重复事件检测与去重
        all_events = self._deduplicate_overlapping(all_events)

        logger.info(
            "Mapped %d segments → %d subtitle events",
            sum(len(s) for s in asr_segments_list),
            len(all_events),
        )
        return all_events

    def _merge_gaps(
        self,
        events: List[SubtitleEvent],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[SubtitleEvent]:
        """处理段间间隙 — 利用精确的 start 时间反向确定 end 时间

        核心原理:
        - 每个字幕的开始时间是精确的（VAD onset + ASR首词 + 声学骨架 三重保险）
        - 从下一个事件的精确 start 向回扫描 gap 区域，找到语音→静音的转折点
        - 将该转折点设为当前事件的结束时间

        ★ 说话人安全检查：当 prev 和 curr 属于不同说话人时，不扩展 prev.end。
        间隙中的语音能量可能是下一个说话人开始发言，而非当前说话人的语尾拖音。

        算法:
        1. 从 curr.start 向 prev.end 方向，逐帧（5ms）扫描 RMS 能量
        2. 找到第一帧 RMS > 静音阈值×2 的位置 → 语音在此结束
        3. 将该位置 + 30ms 尾随余量设为 prev.end
        4. 如果整个 gap 都是静音 → 保留一小段静音尾随（100ms），不延伸到 curr.start
        5. 如果 gap 中检测到语音能量 → 可能存在漏检，保留 gap 不处理
        """
        if len(events) <= 1:
            return events

        # 估算全局静音 RMS 阈值（作为基准参考）
        silence_rms = None
        if audio is not None:
            from ..utils.audio_utils import AudioUtils
            silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)

        result = [events[0]]
        refined_count = 0

        for i in range(1, len(events)):
            prev = result[-1]
            curr = events[i]
            gap = curr.start - prev.end

            if gap <= 0:
                # 重叠：不处理，交给 _deduplicate_overlapping
                result.append(curr)
                continue

            # 只处理中等大小的间隙（过大的间隙是段落间隔，保留）
            if gap > self.natural_pause_max:
                result.append(curr)
                continue

            # ★ 不同说话人：间隙是说话人切换的自然停顿，不扩展 prev.end
            prev_spk = getattr(prev, "speaker_id", None)
            curr_spk = getattr(curr, "speaker_id", None)
            if prev_spk is not None and curr_spk is not None and prev_spk != curr_spk:
                result.append(curr)
                continue

            if audio is None or silence_rms is None:
                # 无音频数据 → 回退：仅当 gap ≤ seamless_threshold 时直接衔接
                if gap <= self.seamless_threshold:
                    prev.end = curr.start - 0.001
                    if prev.end <= prev.start:
                        prev.end = curr.start
                result.append(curr)
                continue

            # ── 反向扫描：从 curr.start 向 prev.end 逐帧检测语音能量 ──
            from ..utils.audio_utils import AudioUtils

            speech_end = self._find_speech_end_backward(
                audio, sample_rate,
                search_start=prev.end,
                search_end=curr.start,
                silence_rms=silence_rms,
            )

            if speech_end is not None:
                # 找到了语音→静音的精确转折点
                # +30ms 尾随余量（语尾自然衰减，不突兀截断）
                refined_end = speech_end + 0.03
                # 不超出 curr.start
                refined_end = min(refined_end, curr.start - 0.01)
                if refined_end > prev.end:
                    prev.end = refined_end
                    refined_count += 1
            else:
                # gap 中未检测到语音能量 → 整个 gap 都是静音
                # 给当前字幕添加合理的尾随余量（方便阅读），但不延伸到下一句
                tail_margin = min(0.15, gap * 0.5)
                candidate = prev.end + tail_margin
                if candidate > prev.end:
                    prev.end = candidate
                    refined_count += 1

            result.append(curr)

        if refined_count > 0:
            logger.info(
                "Gap refine: adjusted %d/%d event end times "
                "via backward energy scan",
                refined_count, len(events),
            )

        return result

    @staticmethod
    def _find_speech_end_backward(
        audio: np.ndarray,
        sample_rate: int,
        search_start: float,
        search_end: float,
        silence_rms: float,
        frame_ms: int = 5,
        energy_threshold_ratio: float = 2.0,
    ) -> Optional[float]:
        """从 search_end 向 search_start 反向扫描，找语音→静音的转折点。

        原理：
        - 从下一个字幕的精确 start 位置向回扫描
        - 逐 5ms 帧计算 RMS 能量
        - 找到第一帧 RMS > silence_rms × threshold_ratio 的位置
        - 该位置即为真实语音结束点

        Args:
            audio: 音频数组
            sample_rate: 采样率
            search_start: 扫描起点（当前 ASR end，偏早）
            search_end: 扫描终点（下一个字幕 start，精确）
            silence_rms: 静音 RMS 阈值
            frame_ms: 每帧时长（ms），默认 5ms
            energy_threshold_ratio: 语音判定 = silence_rms × ratio

        Returns:
            语音结束时间（秒），如果整个区间都是静音则返回 None
        """
        frame_samples = int(frame_ms / 1000.0 * sample_rate)
        if frame_samples < 1:
            return None

        hop_sec = frame_ms / 1000.0
        energy_threshold = silence_rms * energy_threshold_ratio
        total_samples = len(audio)

        # 从 search_end 向 search_start 逐帧扫描
        t = search_end
        while t > search_start + hop_sec:
            t -= hop_sec
            frame_start = int(t * sample_rate)
            frame_end = frame_start + frame_samples
            if frame_end > total_samples:
                frame_end = total_samples
            if frame_end <= frame_start:
                continue

            frame = audio[frame_start:frame_end]
            rms = float(np.sqrt(np.mean(frame ** 2)))

            if rms > energy_threshold:
                # 找到语音能量 — 这是语音结束的位置
                return t + hop_sec  # 返回帧的结束时间

        return None  # 整个区间都是静音

    @staticmethod
    def _deduplicate_overlapping(
        events: List[SubtitleEvent],
    ) -> List[SubtitleEvent]:
        """检测并移除时间上重叠且文本高度相似的重复事件。

        相邻 VAD 段的重叠区域可能导致 ASR 对同一语音做重复识别，
        产生两条 start/end 重叠且文本相似的字幕。

        去重策略：
        - overlap_ratio > 50%（两条字幕时间重叠超一半）
        - text_similarity > 80%（用 difflib.SequenceMatcher 比较）
        - 保留时间覆盖更完整的事件（更早 start 且更晚 end）
        - 文本次优事件更宽泛覆盖

        Args:
            events: 按 start 排序的字幕事件列表

        Returns:
            去重后的事件列表（保留顺序和编号不变）
        """
        if len(events) <= 1:
            return events

        # 预先组织文本（strip + 统一空格）
        texts = [" ".join(e.text.split()) for e in events]

        to_remove: set = set()
        n = len(events)

        for i in range(n):
            if i in to_remove:
                continue
            a = events[i]

            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                b = events[j]

                # 时间重叠检查
                overlap_start = max(a.start, b.start)
                overlap_end = min(a.end, b.end)
                overlap_dur = overlap_end - overlap_start

                if overlap_dur <= 0:
                    # 无重叠（可能 b.start >= a.end）→ 后续事件也不会重叠
                    if b.start >= a.end:
                        break
                    continue

                # 计算重叠比例
                a_dur = a.end - a.start
                b_dur = b.end - b.start
                overlap_ratio = overlap_dur / min(a_dur, b_dur)

                if overlap_ratio < 0.5:
                    continue

                # ★ 不同说话人 → 不视为重复（两个说话人可能说了相似的话）
                spk_a = getattr(a, "speaker_id", None)
                spk_b = getattr(b, "speaker_id", None)
                if spk_a is not None and spk_b is not None and spk_a != spk_b:
                    continue

                # 文本相似度检查
                text_sim = difflib.SequenceMatcher(
                    None, texts[i], texts[j],
                ).ratio()

                # ★ 子串包含检查：如果一条文本完全包含在另一条中，
                #    即使整体相似度 < 0.8 也应视为重复（防止长文本吞短文本的漏检）
                text_contained = (
                    texts[i] in texts[j] or texts[j] in texts[i]
                )

                if text_sim < 0.8 and not text_contained:
                    # ★ 文本不相似但时间包含 + 同说话人：
                    #   可能是不同 ASR 窗口对同一音频区域的不同识别结果。
                    #   应将文本合并到时间覆盖更大的事件中，而非保留两个。
                    a_text_len = len(texts[i])
                    b_text_len = len(texts[j])
                    if (a.start <= b.start and a.end >= b.end
                            and a_text_len > 0 and b_text_len > 0):
                        merged = _merge_distinct_texts(texts[i], texts[j])
                        events[i].text = merged
                        to_remove.add(j)
                        logger.info(
                            "Merged overlapping events: #%d (%.1f-%.1f, '%s') "
                            "+ #%d (%.1f-%.1f, '%s') → '%s'",
                            a.index, a.start, a.end, texts[i],
                            b.index, b.start, b.end, texts[j],
                            merged,
                        )
                    elif (b.start <= a.start and b.end >= a.end
                            and a_text_len > 0 and b_text_len > 0):
                        merged = _merge_distinct_texts(texts[j], texts[i])
                        events[j].text = merged
                        to_remove.add(i)
                        logger.info(
                            "Merged overlapping events: #%d (%.1f-%.1f, '%s') "
                            "+ #%d (%.1f-%.1f, '%s') → '%s'",
                            b.index, b.start, b.end, texts[j],
                            a.index, a.start, a.end, texts[i],
                            merged,
                        )
                        break
                    continue

                # 确认重复：保留覆盖更完整的事件
                # 优先看时间覆盖（更早 start + 更晚 end），
                # 其次看文本长度（更长的文本通常包含更多上下文）
                a_span = a.end - a.start
                b_span = b.end - b.start
                a_text_len = len(texts[i])
                b_text_len = len(texts[j])

                # 如果一个是另一个的完整超集（时间+文本），保留超集
                if a.start <= b.start and a.end >= b.end and a_text_len >= b_text_len:
                    to_remove.add(j)
                    logger.info(
                        "Deduplicated overlapping events: #%d (%.1f-%.1f) "
                        "subsumes #%d (%.1f-%.1f), similarity=%.0f%%",
                        a.index, a.start, a.end,
                        b.index, b.start, b.end,
                        text_sim * 100,
                    )
                elif b.start <= a.start and b.end >= a.end and b_text_len >= a_text_len:
                    to_remove.add(i)
                    logger.info(
                        "Deduplicated overlapping events: #%d (%.1f-%.1f) "
                        "subsumes #%d (%.1f-%.1f), similarity=%.0f%%",
                        b.index, b.start, b.end,
                        a.index, a.start, a.end,
                        text_sim * 100,
                    )
                    break  # a 已移除，不再与后续比较
                else:
                    # 难分优劣：保留文本更长的（更完整的信息）
                    if b_text_len > a_text_len:
                        to_remove.add(i)
                        logger.info(
                            "Deduplicated overlapping events: #%d dropped in "
                            "favor of #%d (longer text), similarity=%.0f%%",
                            a.index, b.index, text_sim * 100,
                        )
                        break
                    else:
                        to_remove.add(j)
                        logger.info(
                            "Deduplicated overlapping events: #%d dropped in "
                            "favor of #%d (longer text), similarity=%.0f%%",
                            b.index, a.index, text_sim * 100,
                        )

        if not to_remove:
            return events

        # 重建事件列表，重新编号
        kept = [e for i, e in enumerate(events) if i not in to_remove]
        for idx, e in enumerate(kept):
            e.index = idx + 1

        logger.info(
            "Overlap dedup: %d → %d events (%d duplicates removed)",
            len(events), len(kept), len(to_remove),
        )
        return kept

    @staticmethod
    def map_single_segment(
        asr_results: List[TranscriptionSegment],
        segment_offset: float,
    ) -> List[SubtitleEvent]:
        """映射单个语音片段的 ASR 结果

        Args:
            asr_results: 单个语音片段的 ASR 识别结果
            segment_offset: 该片段在原始音频中的起始时间

        Returns:
            字幕事件列表
        """
        events = []
        for i, asr_seg in enumerate(asr_results):
            events.append(
                SubtitleEvent(
                    index=i + 1,
                    start=segment_offset + asr_seg.start,
                    end=segment_offset + asr_seg.end,
                    text=asr_seg.text.strip(),
                    words=asr_seg.words,
                )
            )
        return events
