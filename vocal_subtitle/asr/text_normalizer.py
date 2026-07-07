"""ASR 文本后处理规范化

针对 faster-whisper 的已知输出模式做后处理纠错和规范化。

纠错类别:
1. 数字单词化恢复: "One answer" → "1. Answer"
2. 常见标点修复: 句末补标点、多余空格清理
3. 专有名词词典纠错（可配置扩展）
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# 内置专有名词纠错词典（可扩展）
_DEFAULT_CORRECTIONS: Dict[str, str] = {
    # 常见 ASR 误识别 → 正确拼写
    "mahood": "Mehood",
    "lusty": "Lestie",
    "lesti": "Lestie",
    "suzy": "Susie",
    "shao shi": "Xiao Xi",
}


class TextNormalizer:
    """ASR 输出文本规范化器

    在 ASR 识别完成后立即调用，对每条识别文本做规范化处理。

    使用示例:
        normalizer = TextNormalizer()
        text = normalizer.normalize("One answer within three rings")
        # → "1. Answer within three rings."
    """

    def __init__(
        self,
        custom_corrections: Optional[Dict[str, str]] = None,
    ):
        self.corrections = {**_DEFAULT_CORRECTIONS}
        if custom_corrections:
            self.corrections.update(custom_corrections)

        # 数字单词 → 编号模式
        # "One answer" → "1. Answer"（句子开头且后跟名词）
        self._number_word_pattern = re.compile(
            r'^(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\s+(\w)',
            re.IGNORECASE,
        )
        self._number_map = {
            "one": "1.", "two": "2.", "three": "3.", "four": "4.",
            "five": "5.", "six": "6.", "seven": "7.", "eight": "8.",
            "nine": "9.", "ten": "10.",
        }

    def normalize(self, text: str) -> str:
        """对单条字幕文本执行规范化"""
        if not text or not text.strip():
            return text

        # 步骤 1: 数字单词 → 编号 (如 "One answer" → "1. Answer")
        text = self._fix_numbered_lists(text)

        # 步骤 2: 专有名词纠错
        text = self._apply_corrections(text)

        # 步骤 3: 标点规范化
        text = self._normalize_punctuation(text)

        return text.strip()

    def _fix_numbered_lists(self, text: str) -> str:
        """将英文数字词恢复为编号格式

        匹配场景：
        1. 句首数字词 + 后续文本: "One answer" → "1. Answer"
        2. 独立数字词（短文本）: "Two" → "2." （可能是被 VAD 拆分的编号）
        """
        # 场景 1: 句首数字词
        match = self._number_word_pattern.match(text)
        if match:
            word = match.group(1).lower()
            rest_char = match.group(2)
            if word in self._number_map:
                # 仅当后跟的字母是大写或短文本时才转换
                if rest_char.isupper() or len(text.split()) <= 6:
                    rest = text[match.end(1):].strip()
                    return self._number_map[word] + " " + rest

        # 场景 2: 独立数字词（文本仅含一个数字词，如 "Two" / "three"）
        stripped = text.strip().rstrip(".,!?;:，。！？；：")
        word_lower = stripped.lower()
        if word_lower in self._number_map and len(text.split()) == 1:
            suffix = text[len(text.rstrip(".,!?;:，。！？；：")):]
            return self._number_map[word_lower] + suffix

        return text

    def _apply_corrections(self, text: str) -> str:
        """应用专有名词纠错词典

        支持两类匹配：
        1. 多词短语（如 "shao shi" → "Xiao Xi"）→ 全文本扫描
        2. 单词（如 "mahood" → "Mehood"）→ 逐词匹配
        """
        # ---- 多词短语匹配（先处理，避免被逐词匹配干扰） ----
        # 按短语长度降序排列，优先匹配更长的短语
        multi_word_phrases = sorted(
            [(k, v) for k, v in self.corrections.items() if " " in k],
            key=lambda x: len(x[0].split()), reverse=True,
        )
        for phrase_key, replacement in multi_word_phrases:
            # 大小写不敏感替换
            pattern = re.compile(re.escape(phrase_key), re.IGNORECASE)
            text = pattern.sub(replacement, text)

        # ---- 单词匹配 ----
        words = text.split()
        corrected = []
        for w in words:
            clean = w.strip(".,!?;:，。！？；：").lower()
            if clean in self.corrections:
                replacement = self.corrections[clean]
                # 保留原始大小写风格
                if w.isupper():
                    replacement = replacement.upper()
                elif w[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                # 保留尾部标点
                suffix = w[len(w.rstrip(".,!?;:，。！？；：")):]
                w = replacement + suffix
            corrected.append(w)
        return " ".join(corrected)

    def _normalize_punctuation(self, text: str) -> str:
        """标点符号规范化"""
        # 确保句子以标点结尾
        if text and text[-1].isalnum():
            text += "."

        # 修复多余空格
        text = re.sub(r'\s{2,}', ' ', text)

        return text

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """批量规范化"""
        return [self.normalize(t) for t in texts]
