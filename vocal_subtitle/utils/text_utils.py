"""字幕文本拼接工具。"""

from __future__ import annotations

import unicodedata
from typing import Iterable


def _is_cjk_char(char: str) -> bool:
    return bool(char) and unicodedata.east_asian_width(char) in ("W", "F")


def smart_join_texts(texts: Iterable[str]) -> str:
    """拼接片段文本：CJK 相邻直接相连，拉丁文本以空格衔接。

    中文字幕合并碎片（"得" + "了吧。" → "得了吧。"）不应插入空格；
    拉丁文本（"hello" + "world" → "hello world"）保持空格。
    """
    joined = ""
    for text in texts:
        if not text:
            continue
        if not joined:
            joined = text
        elif _is_cjk_char(joined[-1]) and _is_cjk_char(text[0]):
            joined += text
        else:
            joined += " " + text
    return joined
