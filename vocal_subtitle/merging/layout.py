"""Frame seamless stitching, layout suggestions, and auto line break logic."""

import re
from typing import Dict, List

from .merge_constraints import _physical_owner_compatible_for_events


# LLM 合并 Prompt 中的排版规则补充（可选注入）
SUBTITLE_LAYOUT_RULES = """
Subtitle Layout Rules (apply to merged output):
8. If the merged subtitle text exceeds 40 characters (or 20 CJK characters),
   suggest a line break point. The break should occur at:
   - A natural phrase boundary (after comma, before conjunction)
   - Between subject and predicate for long clauses
   - NEVER in the middle of a word or proper noun
9. Output format for line break suggestion: add "line_break_after_word_index"
   to indicate where the first line ends.

Example:
Input: "So I have you down for a non-smoking king room tomorrow, is that right?"
Output: {
  "merge_groups": [{"ids": [3,4,5], "reason": "..."}],
  "layout_suggestions": [{
    "group_id": 0,
    "line1": "So I have you down for a non-smoking king room",
    "line2": "tomorrow, is that right?",
    "break_after_word_index": 12
  }]
}
"""


def apply_frame_seamless_stitching(
    events: List,
    max_stitch_gap: float = 0.12,
) -> List:
    """帧级无缝衔接

    对非句尾字幕（不以 .!?。！？ 结尾），
    将其结束时间延伸至下一句开始时间，消除字幕闪烁。

    ★ 说话人安全检查：仅在同一说话人（或说话人信息不可用）时衔接。
    不同说话人的字幕即使间隙很小也不衔接，避免 A 的字幕覆盖 B 的语音。
    """
    sentence_endings = {".", "!", "?", "。", "！", "？", "…", "——"}

    for i in range(len(events) - 1):
        curr = events[i]
        nxt = events[i + 1]
        gap = nxt.start - curr.end

        if 0 < gap <= max_stitch_gap:
            # ★ 不同说话人 → 不衔接（保留间隙作为说话人切换的视觉提示）
            curr_spk = getattr(curr, "speaker_id", None)
            next_spk = getattr(nxt, "speaker_id", None)
            if curr_spk is not None and next_spk is not None and curr_spk != next_spk:
                continue

            # ★ 不同物理片段 → 不衔接（不同录音源的边界不可跨）
            if not _physical_owner_compatible_for_events(curr, nxt):
                continue

            # 当前字幕的文本
            text = getattr(curr, "text", "")
            text_ends_with_terminal = (
                text.rstrip()[-1] in sentence_endings
                if text.rstrip() else False
            )

            if not text_ends_with_terminal:
                curr.end = nxt.start  # 无缝衔接

    return events


def apply_layout_suggestions(
    events: List,
    layout_suggestions: List[Dict],
) -> List:
    """将 LLM 的断行建议应用到字幕事件

    对于 ASS 格式：使用 \\N 作为换行标记
    对于 SRT 格式：使用 \\n
    """
    for suggestion in layout_suggestions:
        group_id = suggestion.get("group_id", -1)
        if 0 <= group_id < len(events):
            line1 = suggestion.get("line1", "")
            line2 = suggestion.get("line2", "")
            if line1 and line2:
                events[group_id].text = f"{line1}\\N{line2}"
                if hasattr(events[group_id], "_layout_applied"):
                    events[group_id]._layout_applied = True
                else:
                    setattr(events[group_id], "_layout_applied", True)

    return events


def auto_line_break_fallback(
    text: str,
    max_chars_per_line: int = 20,
) -> str:
    """纯规则的自动断行（LLM 不可用时的降级方案）

    在以下位置寻找断行点（按优先级）：
    1. 逗号/分号/破折号后
    2. 介词/连词前（and, but, or, 在, 给, 为）
    3. 中点字符数找空格（最后手段）
    """
    if len(text) <= max_chars_per_line:
        return text

    best_break = None   # (position, distance, priority)
    best_priority = 3   # 1=标点, 2=连词, 3=无(降级到中点)

    for expand in [1.0, 1.5, 2.0, 2.5]:
        search_limit = int(max_chars_per_line * expand)

        # 寻找自然断点（优先级 1：标点符号后）
        for char in [",", "，", ";", "；", "—", "…", "、"]:
            idx = text.rfind(char, 0, search_limit)
            if idx > max_chars_per_line * 0.3:
                dist = abs((idx + 1) - max_chars_per_line)
                if best_break is None or best_priority > 1 or dist < best_break[1]:
                    best_break = (idx + 1, dist, 1)
                    best_priority = 1

        # 寻找自然断点（优先级 2：连词/介词前）
        if best_priority > 1:
            for word in [
                " and ", " but ", " or ", " to ", " for ", " with ",
                " 在", " 给", " 为", " 和", " 而且", " 但是",
                " 所以", " 然后", " 因为",
            ]:
                idx = text.rfind(word, 0, search_limit + len(word))
                if idx > max_chars_per_line * 0.3:
                    dist = abs(idx - max_chars_per_line)
                    if best_break is None or best_priority > 1 or dist < best_break[1]:
                        best_break = (idx, dist, 2)
                        best_priority = 2

        # 找到优先级 1（标点）断点 → 不再扩大搜索
        if best_priority == 1:
            break

    if best_break is not None:
        break_point = best_break[0]
        return text[:break_point].rstrip() + "\\N" + text[break_point:].lstrip()

    # 优先级 3（最后手段）：中点附近找空格
    mid = len(text) // 2
    for offset in range(0, len(text) // 4):
        for direction in [1, -1]:
            idx = mid + offset * direction
            if 0 <= idx < len(text) and text[idx] == " ":
                return text[:idx] + "\\N" + text[idx + 1:]

    # 完全无法断行：在中点强制断行
    return text[:mid] + "\\N" + text[mid:]


def auto_layout_events(
    events: List,
    max_chars_cjk: int = 20,
    max_chars_latin: int = 40,
) -> List:
    """对字幕事件列表自动应用断行规则"""
    for event in events:
        text = getattr(event, "text", "")
        if not text:
            continue

        # 已有换行标记 → 跳过
        if "\\N" in text or "\\n" in text:
            continue

        # 估算字符类型
        cjk_count = sum(1 for c in text if "一" <= c <= "鿿"
                        or "぀" <= c <= "ゟ"
                        or "가" <= c <= "힯")
        latin_count = sum(1 for c in text if c.isascii() and c.isalpha())

        # 根据主导语言选择阈值
        if cjk_count > latin_count:
            threshold = max_chars_cjk
        else:
            threshold = max_chars_latin

        if len(text) > threshold:
            event.text = auto_line_break_fallback(text, threshold)

    return events
