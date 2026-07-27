"""字幕构建与格式化模块

将 SubtitleEvent 列表构建为 SRT / VTT / ASS 格式字幕文件。

规则:
- 基于时长和字符数自动拆分/合并字幕行
- 中文每行最多 20 字，英文每行最多 42 字符
- 单条字幕最多 2 行
"""
import copy
import math

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .time_mapper import SubtitleEvent

logger = logging.getLogger(__name__)


@dataclass
class SubtitleRule:
    """字幕构建规则"""

    min_duration: float = 0.8  # 最小字幕时长（秒）
    max_duration: float = 5.0  # 最大字幕时长（秒）
    max_chars_cjk: int = 20  # 中文每行最多字数
    max_chars_latin: int = 42  # 英文每行最多字符数
    max_lines: int = 2  # 单条字幕最多行数
    speaker_label_format: str = "bracket"  # bracket | prefix | none


class SubtitleBuilder:
    """字幕构建器

    将字幕事件列表构建为标准格式字幕文件，支持 SRT / VTT / ASS。

    使用示例:
        builder = SubtitleBuilder(rule=SubtitleRule(max_chars_cjk=20))
        builder.build(events, Path("output.srt"), format="srt")
    """

    # CJK 字符范围
    _CJK_PATTERN = re.compile(
        r"[一-鿿぀-ゟ゠-ヿ가-힯]"
    )

    # 句子结束标点（自然断句边界）
    _SENTENCE_END = re.compile(r"[.!?！？。]+")

    # 常见缩写（不应断句）
    _ABBREVIATIONS = frozenset({
        "dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr",
        "etc", "vs", "inc", "ltd", "co", "dept", "est",
        "approx", "esp", "eg", "ie",
    })

    # 弱分隔符（逗号、分号、空格等 — 仅在强制拆分时使用）
    _WEAK_SPLIT = re.compile(r"[,，;；\s—…]+")

    def __init__(self, rule: Optional[SubtitleRule] = None):
        self.rule = rule or SubtitleRule()

    def build(
        self,
        events: List[SubtitleEvent],
        output_path: Path,
        fmt: str = "srt",
    ) -> Path:
        """构建字幕文件

        Args:
            events: 字幕事件列表
            output_path: 输出文件路径
            fmt: 输出格式 (srt / vtt / ass)

        Returns:
            输出文件路径
        """
        # Events reaching the builder have already passed the finalizer.  The
        # builder may format text for a target container, but it must not alter
        # logical cue count, timing, numbering, or caller-owned objects.
        wrapped = self._apply_line_wrapping(copy.deepcopy(list(events)))

        # 使用 pysubs2 导出
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs = self._to_ssa(wrapped, fmt=fmt)
        subs.save(str(output_path), format_=fmt)

        logger.info(
            "Subtitle saved: %s (%d events, format=%s)",
            output_path,
            len(wrapped),
            fmt,
        )
        return output_path

    def build_to_string(
        self,
        events: List[SubtitleEvent],
        fmt: str = "srt",
    ) -> str:
        """构建字幕并返回字符串

        Args:
            events: 字幕事件列表
            fmt: 输出格式

        Returns:
            字幕文件内容字符串
        """
        wrapped = self._apply_line_wrapping(copy.deepcopy(list(events)))
        subs = self._to_ssa(wrapped, fmt=fmt)
        return subs.to_string(fmt)

    def _merge_short_events(
        self, events: List[SubtitleEvent]
    ) -> List[SubtitleEvent]:
        """合并过短的相邻字幕（时长 < min_duration 且中间无大间隙）"""
        if not events:
            return []

        merged = [events[0]]
        rule = self.rule

        for event in events[1:]:
            prev = merged[-1]
            gap = event.start - prev.end

            # 如果前一条太短且间隙小，合并
            if prev.duration < rule.min_duration and gap < 0.3:
                # ★ 不同说话人 → 不合并
                if _same_speaker(prev, event):
                    prev.end = event.end
                    prev.text = prev.text + " " + event.text
                    if event.words:
                        prev.words.extend(event.words)
                    continue

            # 如果当前太短，尝试与前一条合并
            if event.duration < rule.min_duration and gap < 0.3:
                if _same_speaker(prev, event):
                    prev.end = event.end
                    prev.text = prev.text + " " + event.text
                    if event.words:
                        prev.words.extend(event.words)
                    continue

            merged.append(event)

        # 重新编号
        for i, e in enumerate(merged):
            e.index = i + 1

        return merged

    def _split_long_events(
        self, events: List[SubtitleEvent]
    ) -> List[SubtitleEvent]:
        """拆分超长字幕行（时长 > max_duration 或字符数超标）"""
        rule = self.rule
        result = []

        for event in events:
            char_count = self._count_display_chars(event.text)

            if event.duration <= rule.max_duration and char_count <= self._max_chars_for_text(event.text):
                result.append(event)
                continue

            # 需要拆分
            sub_events = self._split_event(event)
            result.extend(sub_events)

        # 重新编号
        for i, e in enumerate(result):
            e.index = i + 1

        return result

    # ------------------------------------------------------------------
    # 自然断句拆分
    # ------------------------------------------------------------------

    def _split_event(
        self, event: SubtitleEvent
    ) -> List[SubtitleEvent]:
        """按说话人自然短句拆分过长的字幕事件。

        策略:
        1. 优先在句子结束标点 (.!?！？。) 处断开 —— 对应说话人的自然停顿
        2. 仅在单句仍超过 max_duration 时，才降级到逗号/空格处强制拆分
        3. 利用 word 时间戳精确定时；无 word 数据时按字符比例估算
        """
        text = event.text
        total_duration = event.duration
        total_chars = max(1, self._count_display_chars(text))

        # ---- 第一步：拆分为原子 utterance（每个是一条自然句子） ----
        utterances = self._split_into_utterances(text)
        if len(utterances) <= 1:
            # 无句子边界可拆分 → 强制在逗号/空格处断开
            return self._split_event_forced(event)

        # ---- 第二步：计算每个 utterance 的时间范围 ----
        timed = self._time_utterances(utterances, event, total_duration, total_chars)

        # ---- 第三步：按句子边界分组，尊重 max_duration ----
        return self._group_utterances(timed, event)

    # ------------------------------------------------------------------
    # 第一步辅助：按句子结束标点拆分文本
    # ------------------------------------------------------------------

    @classmethod
    def _split_into_utterances(cls, text: str) -> List[str]:
        """在句子结束标点处拆分文本，返回句子列表。

        保留标点：每个 utterance 末尾包含其结束标点。
        支持英文（标点后有空格）和中文（标点后无空格）。
        自动过滤常见缩写（Dr., Mr., etc.）的误拆分。

        例： "Hello world. How are you?" → ["Hello world.", "How are you?"]
        例： "第一句话。第二句话。" → ["第一句话。", "第二句话。"]
        """
        # 在 sentence-end 标点处拆分
        raw_parts = re.split(r"(?<=[.!?！？。])\s*", text)
        raw_parts = [p.strip() for p in raw_parts if p.strip()]
        if len(raw_parts) <= 1:
            return raw_parts

        # 合并缩写误拆分：如果前一段以缩写结尾（如 "Dr."），
        # 则将其与后一段合并
        merged = []
        for part in raw_parts:
            # 检查前一段是否以缩写结尾
            if merged:
                prev = merged[-1]
                # 提取最后一个"词"（不含标点）
                last_word = re.sub(r"[.!?！？。]+$", "", prev.split()[-1]) if prev.split() else ""
                if last_word.lower() in cls._ABBREVIATIONS:
                    merged[-1] = prev + " " + part
                    continue
            merged.append(part)

        return merged

    # ------------------------------------------------------------------
    # 第二步辅助：为每个 utterance 计算时间范围
    # ------------------------------------------------------------------

    @staticmethod
    def _time_utterances(
        utterances: List[str],
        event: SubtitleEvent,
        total_duration: float,
        total_chars: int,
    ) -> List[dict]:
        """为每个 utterance 计算 start/end 时间。

        优先使用 word 时间戳精确定时，无数据时按字符比例估算。
        返回 [{"text", "start", "end", "words"}, ...]，时间均为全局秒数。
        """
        result = []
        char_offset = 0  # 显示字符偏移

        for utt_text in utterances:
            utt_chars = SubtitleBuilder._count_display_chars(utt_text)

            # 按字符比例估算时间（fallback）
            start_prop = char_offset / total_chars
            end_prop = (char_offset + utt_chars) / total_chars
            est_start = event.start + start_prop * total_duration
            est_end = event.start + end_prop * total_duration

            # 尝试用 word 时间戳精确定时
            utt_words = SubtitleBuilder._match_words_to_text(
                utt_text, event.words, est_start - event.start, est_end - event.start
            )

            if utt_words:
                # 使用第一个和最后一个词的时间
                precise_start = event.start + utt_words[0].start
                precise_end = event.start + utt_words[-1].end
            else:
                precise_start = max(event.start, est_start)
                precise_end = min(event.end, est_end)

            result.append({
                "text": utt_text,
                "start": precise_start,
                "end": precise_end,
                "words": utt_words,
                "chars": utt_chars,
            })
            char_offset += utt_chars

        return result

    # ------------------------------------------------------------------
    # 第三步辅助：按 max_duration 分组 utterance
    # ------------------------------------------------------------------

    def _group_utterances(
        self, timed: List[dict], event: SubtitleEvent
    ) -> List[SubtitleEvent]:
        """将时间化的 utterance 分组为字幕事件。

        核心原则：每个自然句子独立为一条字幕，仅在以下情况合并：
        - 单句太短（< min_duration）且与下一句合并后不超过 max_duration
        - 单句超过 max_duration 时在弱分隔符处强制拆分
        """
        rule = self.rule
        sub_events = []
        i = 0

        while i < len(timed):
            utt = timed[i]
            duration = utt["end"] - utt["start"]

            # ---- 情况 1：单句太长 → 强制拆分 ----
            if duration > rule.max_duration:
                forced = self._split_event_forced(
                    SubtitleEvent(
                        index=0,
                        start=utt["start"],
                        end=utt["end"],
                        text=utt["text"],
                        words=utt["words"],
                    )
                )
                sub_events.extend(forced)
                i += 1
                continue

            # ---- 情况 2：单句太短 → 尝试与后续句子合并 ----
            if duration < rule.min_duration and i + 1 < len(timed):
                merged_texts = [utt["text"]]
                merged_words = list(utt["words"])
                merged_start = utt["start"]
                merged_end = utt["end"]
                j = i + 1

                while j < len(timed):
                    next_utt = timed[j]
                    combined_duration = next_utt["end"] - merged_start
                    if combined_duration > rule.max_duration:
                        break
                    merged_texts.append(next_utt["text"])
                    merged_words.extend(next_utt["words"])
                    merged_end = next_utt["end"]
                    j += 1
                    if merged_end - merged_start >= rule.min_duration:
                        break  # 合并到满足最小时长为止

                sub_events.append(self._make_sub_event(
                    merged_texts, merged_words, merged_start, merged_end, event,
                ))
                i = j
                continue

            # ---- 情况 3：正常单句 → 独立字幕 ----
            sub_events.append(self._make_sub_event(
                [utt["text"]], utt["words"], utt["start"], utt["end"], event,
            ))
            i += 1

        return sub_events

    @staticmethod
    def _make_sub_event(
        texts: List[str],
        words: list,
        start: float,
        end: float,
        _original: SubtitleEvent,
    ) -> SubtitleEvent:
        """从累积的句子列表构造单个 SubtitleEvent，保留说话人信息"""
        return SubtitleEvent(
            index=0,
            start=start,
            end=end,
            text=" ".join(texts),
            words=words,
            original_text=_original.original_text,
            speaker_id=_original.speaker_id,
            speaker_label=_original.speaker_label,
        )

    # ------------------------------------------------------------------
    # 降级路径：强制拆分（在逗号/空格处），用于无句子边界或单句过长
    # ------------------------------------------------------------------

    def _split_event_forced(
        self, event: SubtitleEvent
    ) -> List[SubtitleEvent]:
        """在弱分隔符处按字符数阈值强制拆分（降级路径）。"""
        text = event.text
        parts = re.split(f"({self._WEAK_SPLIT.pattern})", text)
        total_duration = event.duration
        total_chars = max(1, self._count_display_chars(text))
        target_chars = self._max_chars_for_text(text) // 2

        sub_events = []
        current_text = ""
        char_pos = 0
        seg_start_char = 0

        for part in parts:
            if not part:
                continue

            is_delim = bool(re.match(f"^{self._WEAK_SPLIT.pattern}$", part))

            if is_delim:
                if not current_text.strip():
                    if sub_events:
                        sub_events[-1].text = sub_events[-1].text.rstrip() + part
                        delim_chars = self._count_display_chars(part)
                        new_char_pos = char_pos + delim_chars
                        sub_events[-1].end = (
                            event.start
                            + (new_char_pos / total_chars) * total_duration
                        )
                        seg_start_char = new_char_pos
                        char_pos = new_char_pos
                    else:
                        char_pos += self._count_display_chars(part)
                        seg_start_char = char_pos
                    continue
                current_text += part
                continue

            part_chars = self._count_display_chars(part)
            new_char_pos = char_pos + part_chars
            current_text += part
            current_seg_chars = new_char_pos - seg_start_char

            if current_seg_chars >= target_chars or new_char_pos >= total_chars:
                start_prop = seg_start_char / total_chars
                end_prop = new_char_pos / total_chars
                sub_start = event.start + start_prop * total_duration
                sub_end = event.start + end_prop * total_duration

                sub_words = [
                    w for w in event.words
                    if w.start >= sub_start - event.start
                    and w.end <= sub_end - event.start + 0.001
                ]

                sub_text = current_text.strip()
                sub_events.append(
                    SubtitleEvent(
                        index=0,
                        start=max(event.start, sub_start),
                        end=min(event.end, sub_end),
                        text=sub_text,
                        words=sub_words,
                        original_text=event.original_text,
                        speaker_id=event.speaker_id,
                        speaker_label=event.speaker_label,
                    )
                )
                current_text = ""
                seg_start_char = new_char_pos

            char_pos = new_char_pos

        # 尾部残留：太短则合并到前一条，避免孤立碎片
        if current_text.strip():
            stripped = current_text.strip()
            tail_chars = self._count_display_chars(stripped)
            # 尾部片段占比 < 25% 或 ≤ 4 个显示字符 → 合并到前一条
            if sub_events and (
                tail_chars <= 4
                or (tail_chars / max(1, total_chars)) < 0.25
            ):
                sub_events[-1].end = event.end
                sub_events[-1].text = (
                    sub_events[-1].text.rstrip() + " " + stripped
                )
            else:
                prev_end = sub_events[-1].end if sub_events else event.start
                sub_events.append(
                    SubtitleEvent(
                        index=0,
                        start=prev_end,
                        end=event.end,
                        text=stripped,
                        original_text=event.original_text,
                        speaker_id=event.speaker_id,
                        speaker_label=event.speaker_label,
                    )
                )

        return sub_events if sub_events else [event]

    # ------------------------------------------------------------------
    # 词匹配辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _match_words_to_text(
        text: str, words: list, t_start: float, t_end: float
    ) -> list:
        """从 WordTimestamp 列表中匹配属于指定文本的词。

        采用简单级联匹配：将 text 拆分为词，在 words 中按顺序查找。
        匹配成功返回子列表（同一次序），失败返回空列表。

        Args:
            text: 要匹配的文本
            words: WordTimestamp 列表（segment-relative 时间）
            t_start: 文本的预估起始时间（segment-relative）
            t_end: 文本的预估结束时间（segment-relative）

        Returns:
            匹配到的 WordTimestamp 子列表
        """
        if not words:
            return []

        # 在预估时间窗口内筛选候选词
        margin = (t_end - t_start) * 0.5
        candidates = [
            w for w in words
            if w.start >= t_start - margin and w.end <= t_end + margin
        ]
        if not candidates:
            return []

        # 拆分文本为词（去除标点，小写）
        text_words = re.findall(r"[\w']+", text.lower())
        if not text_words:
            return []

        # 在候选词中贪心匹配文本词序列
        matched = []
        cand_idx = 0
        for tw in text_words:
            found = None
            for j in range(cand_idx, len(candidates)):
                cw = candidates[j]
                # 清理词文本（去除标点、小写）
                cw_clean = cw.word.strip(".,!?;:，。！？；：\"'").lower()
                if cw_clean == tw or tw in cw_clean or cw_clean in tw:
                    found = cw
                    cand_idx = j + 1
                    break
            if found is not None:
                matched.append(found)
            # 允许跳过少量不匹配的词（LLM 优化可能增删词）

        return matched

    def _apply_line_wrapping(
        self, events: List[SubtitleEvent]
    ) -> List[SubtitleEvent]:
        """应用自动换行规则"""
        rule = self.rule
        for event in events:
            text = event.text
            if "\n" in text:
                continue  # 已有换行

            max_chars = self._max_chars_for_text(text)
            if self._count_display_chars(text) > max_chars:
                # 在合适位置插入换行
                event.text = self._insert_line_break(text, max_chars)

        return events

    def _max_chars_for_text(self, text: str) -> int:
        """根据文本语言返回最大字符数"""
        cjk_count = len(self._CJK_PATTERN.findall(text))
        total = max(1, len(text.replace(" ", "")))

        if cjk_count / total > 0.5:
            return self.rule.max_chars_cjk * self.rule.max_lines
        else:
            return self.rule.max_chars_latin * self.rule.max_lines

    @staticmethod
    def _count_display_chars(text: str) -> int:
        """计算显示字符数（不含空格和换行）"""
        return len(text.replace(" ", "").replace("\n", ""))

    def _insert_line_break(self, text: str, max_chars: int) -> str:
        """在合适位置插入换行符"""
        mid = len(text) // 2
        # 在中点附近寻找标点或空格
        for offset in range(min(10, mid)):
            for direction in [1, -1]:
                pos = mid + offset * direction
                if 0 < pos < len(text) and text[pos] in "，,。.；;、 ":
                    return text[: pos + 1] + "\n" + text[pos + 1 :]

        # 没找到合适的断点，在中间强制换行
        return text[:mid] + "\n" + text[mid:]

    def _to_ssa(self, events: List[SubtitleEvent], fmt: str = "srt"):
        """转换为 pysubs2 内部格式

        Args:
            events: 字幕事件列表
            fmt: 输出格式。ASS 格式使用 Name 字段存放说话人标签；
                 SRT/VTT 格式将说话人前缀嵌入文本。
        """
        import pysubs2

        subs = pysubs2.SSAFile()

        for event in events:
            # 确定 ASS Name 字段的值：
            # 优先使用 speaker_label（LLM 角色标注结果）
            # 其次使用 speaker_id 生成通用标签（如 "说话人A"）
            # 都没有则为空
            name = ""
            text = event.text

            if fmt == "ass":
                label = event.speaker_label
                if not label and event.speaker_id is not None:
                    # 从 speaker_id 生成通用标签：0→"说话人A", 1→"说话人B", ...
                    spk_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    sid = event.speaker_id
                    if sid < 26:
                        label = f"说话人{spk_labels[sid]}"
                    else:
                        label = f"说话人{sid}"
                if label:
                    name = label
            else:
                # SRT/VTT 格式：说话人作为文本前缀嵌入
                text = self._format_speaker_label(event)

            sub_event = pysubs2.SSAEvent(
                start=max(1, int(math.floor(event.start * 1000 + 0.5))),  # SRT uses ms, never 0
                end=max(1, int(math.floor(event.end * 1000 + 0.5))),
                text=text,
                name=name,
            )
            subs.append(sub_event)

        return subs

    def _format_speaker_label(self, event: SubtitleEvent) -> str:
        """根据配置格式化说话人标签

        bracket: [主持人] 大家好
        prefix:  主持人: 大家好
        none:    大家好
        """
        if not event.speaker_label:
            return event.text

        fmt = self.rule.speaker_label_format
        if fmt == "prefix":
            return f"{event.speaker_label}: {event.text}"
        elif fmt == "bracket":
            return f"[{event.speaker_label}] {event.text}"
        else:
            return event.text

    @staticmethod
    def merge_formats(
        srt_path: Path,
        vtt_path: Optional[Path] = None,
        ass_path: Optional[Path] = None,
    ) -> dict:
        """同时输出多种格式

        Args:
            srt_path: SRT 输出路径
            vtt_path: VTT 输出路径（可选）
            ass_path: ASS 输出路径（可选）

        Returns:
            输出路径字典
        """
        results = {"srt": str(srt_path)}

        import pysubs2

        subs = pysubs2.load(str(srt_path), encoding="utf-8")

        if vtt_path:
            vtt_path.parent.mkdir(parents=True, exist_ok=True)
            subs.save(str(vtt_path), format_="vtt")
            results["vtt"] = str(vtt_path)

        if ass_path:
            ass_path.parent.mkdir(parents=True, exist_ok=True)
            subs.save(str(ass_path), format_="ass")
            results["ass"] = str(ass_path)

        return results


def _same_speaker(a, b) -> bool:
    """检查两个 SubtitleEvent 是否来自同一说话人。

    策略（从保守到严格）：
    - 双方都有 speaker_id → 严格比较，相同才允许合并
    - 仅一方有 speaker_id → 不合并（保守：防止不同说话人内容被混在一起）
    - 双方都无 speaker_id → 允许合并（无法判断，默认安全）
    """
    sid_a = getattr(a, "speaker_id", None)
    sid_b = getattr(b, "speaker_id", None)
    if sid_a is not None and sid_b is not None:
        return sid_a == sid_b
    if sid_a is not None or sid_b is not None:
        # 一方有说话人信息、另一方没有 → 可能是不同的说话人
        # 保守处理：不合并，避免内容错配
        return False
    # 双方都无 speaker 信息 → 允许合并
    return True
