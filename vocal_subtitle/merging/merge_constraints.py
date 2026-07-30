"""Merge constraints — hard rules for physical owner, speaker, gap checking."""

import re
from typing import Dict, List

# 这些模式表示"下一句开启了新的语义段落"，应阻止合并
_SECTION_START_PATTERNS: List[re.Pattern] = [
    # 编号列表（英文）
    re.compile(r'^\d+[\.\)]\s'),
    # 编号列表（英文文字）
    re.compile(r'^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)[,.\s]'),
    # 编号列表（中文）
    re.compile(r'^[一二三四五六七八九十][、，.]'),
    # 段落标题关键词
    re.compile(r'^(Summary\s+(and|&)\s+review)', re.IGNORECASE),
    re.compile(r'^(Example[:]?)', re.IGNORECASE),
    re.compile(r'^(Effective\s+Communication)', re.IGNORECASE),
    re.compile(r'^(Phone\s+Etiquette|Rapid\s+Response)', re.IGNORECASE),
    re.compile(r'^(Answer|Listen|Hang\s+up|Identify)', re.IGNORECASE),
    re.compile(r'^(End\s+with\s+courtesy)', re.IGNORECASE),
]

# 当前文本末尾是段落分隔符 → 不向后合并
_SECTION_END_MARKERS: List[re.Pattern] = [
    re.compile(r'(^|\s)(and|with)\s+courtesy[.]?\s*$', re.IGNORECASE),
]


def _detect_semantic_boundary(current_text: str, next_text: str) -> bool:
    """检测两个相邻片段间是否有语义边界。

    Returns:
        True 如果检测到边界（不应合并），False 如果无边界（可合并）。
    """
    # 检查下一段是否是段落开头（编号、标题等）
    next_stripped = next_text.strip()
    for pattern in _SECTION_START_PATTERNS:
        if pattern.match(next_stripped):
            return True

    # 检查当前段是否是段落结尾标记
    current_stripped = current_text.strip()
    for pattern in _SECTION_END_MARKERS:
        if pattern.search(current_stripped):
            return True

    return False


def _physical_owner_compatible(left: dict, right: dict) -> bool:
    """Two fragments belong to the same physical clip and can be merged."""
    left_spans = left.get("physical_spans", []) or []
    right_spans = right.get("physical_spans", []) or []
    if not left_spans or not right_spans:
        return True  # no physical ownership data — allow merge
    left_clips = {
        (s.get("physical_clip_id") or s.get("clip_id"))
        for s in left_spans
    }
    right_clips = {
        (s.get("physical_clip_id") or s.get("clip_id"))
        for s in right_spans
    }
    # Only allow merge when they share at least one physical clip
    return bool(left_clips & right_clips)


def _physical_owner_compatible_for_events(left, right) -> bool:
    """Two SubtitleEvents belong to the same physical clip."""
    left_spans = list(getattr(left, "physical_spans", []) or [])
    right_spans = list(getattr(right, "physical_spans", []) or [])
    if not left_spans or not right_spans:
        return True

    def _clip_id(span):
        if isinstance(span, dict):
            return span.get("physical_clip_id") or span.get("clip_id")
        return getattr(span, "clip_id", None) or getattr(span, "physical_clip_id", None)

    left_clips = {_clip_id(s) for s in left_spans}
    right_clips = {_clip_id(s) for s in right_spans}
    return bool(left_clips & right_clips)
