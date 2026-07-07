"""LLM 语义仲裁器

对滑动窗口冗余 ASR 产生的多个假设，用 LLM 做语法/语义分析，
决定争议词的归属和最优时间轴。

核心流程:
1. 接收 BoundaryReASRResult（含原始 + 3 个窗口的 ASR 输出）
2. 构建仲裁 Prompt（上文 + 候选假设 + 下文）
3. LLM 分析：哪个词属于哪个段 + 最佳切分点
4. 解析 LLM 输出为 WordAssignment 列表
5. 根据置信度决定自动应用 / 人工复核

LLM 输出格式 (JSON):
{
  "word_assignments": [
    {"word": "unbearably", "segment": "left",  "confidence": 1.0},
    {"word": "hot",         "segment": "left",  "confidence": 0.95},
    {"word": "you'll",      "segment": "right", "confidence": 0.98},
    ...
  ],
  "boundary_adjustment": {
    "new_split_after_word": "hot",
    "left_end_time": 25.71,
    "right_start_time": 26.00
  },
  "rationale": "unbearably is an adverb modifying hot...",
  "confidence": 0.95
}
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..asr.base import TranscriptionSegment, WordTimestamp
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class WordAssignment:
    """单个词的归属决定"""

    word: str
    segment: str                       # "left" | "right"
    confidence: float                  # 0.0 ~ 1.0
    global_start: Optional[float] = None  # 修正后的全局开始时间
    global_end: Optional[float] = None    # 修正后的全局结束时间


@dataclass
class ArbitrationResult:
    """单个边界的仲裁结果"""

    boundary_index: int
    word_assignments: List[WordAssignment] = field(default_factory=list)
    left_text_final: str = ""          # 仲裁后的左段文本
    right_text_final: str = ""         # 仲裁后的右段文本
    left_end_time: Optional[float] = None   # 修正后的左段结束时间
    right_start_time: Optional[float] = None  # 修正后的右段开始时间
    confidence: float = 0.0            # 整体仲裁置信度
    rationale: str = ""                # LLM 判断理由
    auto_applied: bool = False         # 是否自动应用
    needs_review: bool = False         # 是否需人工复核
    error: Optional[str] = None


@dataclass
class ArbitrationConfig:
    """仲裁配置"""

    llm_model: str = "deepseek-v4-pro"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_timeout: float = 15.0

    auto_apply_confidence: float = 0.8   # > 此值自动应用
    review_threshold: float = 0.5        # 50-80% 标记复核
    # 降级：无 LLM 时使用纯规则
    fallback_to_rules: bool = True


# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

BOUNDARY_ARBITRATION_PROMPT = """You are a subtitle boundary arbitrator.
Your task is to determine the optimal word-level boundary between two adjacent
subtitle segments when the original boundary is uncertain.

## Context

The original ASR split these two segments at a word boundary, but due to fast
speech and minimal silence gaps, the split may be incorrect. We ran additional
ASR passes with overlapping windows and got multiple hypotheses.

## Rules

1. Analyze the GRAMMAR and SEMANTICS: which words naturally belong together?
2. An adverb (e.g., "unbearably", "very", "extremely") usually modifies an
   adjective or verb in the SAME clause — they should stay together.
3. A conjunction or preposition at the start of the right segment often
   indicates it was split from the left segment's clause.
4. A complete clause (subject + verb + object) should start a new segment.
5. If a word is clearly part of the LEFT clause grammatically, assign it to "left".
6. If a word is clearly the start of a NEW clause, assign it to "right".
7. Prefer keeping adjectives with their modifying adverbs.
8. Do NOT split a proper noun, compound word, or fixed expression.

## Input Format

You will receive:
- "left_context": the 2-3 segments before the boundary (for context)
- "right_context": the 2-3 segments after the boundary (for context)
- "original_left": original ASR text of the left segment
- "original_right": original ASR text of the right segment
- "hypotheses": alternative ASR outputs from overlapping windows

## Output Format

Return ONLY a valid JSON object (no markdown, no extra text):

{
  "word_assignments": [
    {"word": "<word>", "segment": "left|right", "confidence": 0.0-1.0}
  ],
  "boundary_adjustment": {
    "new_split_after_word": "<the last word that belongs to left>"
  },
  "rationale": "<brief explanation>",
  "confidence": 0.0-1.0
}

## Example

Input:
  original_left:  "other room in the house that gets unbearably."
  original_right: "hot you'll know exactly what i mean"
  hypotheses: {
    "left_expand": "other room in the house that gets unbearably hot",
    "fusion": "unbearably hot you'll know exactly what i mean"
  }

Output:
{
  "word_assignments": [
    {"word": "unbearably", "segment": "left", "confidence": 1.0},
    {"word": "hot", "segment": "left", "confidence": 0.95},
    {"word": "you'll", "segment": "right", "confidence": 0.98}
  ],
  "boundary_adjustment": {
    "new_split_after_word": "hot"
  },
  "rationale": "unbearably is an adverb modifying the adjective hot. They form a single semantic unit. 'you'll know...' begins a new clause.",
  "confidence": 0.95
}
"""


# ---------------------------------------------------------------------------
# 仲裁器
# ---------------------------------------------------------------------------

class BoundaryArbitrator:
    """LLM 语义仲裁器

    使用示例:
        arb = BoundaryArbitrator(config)
        result = arb.arbitrate(
            reasr_result=boundary_reasr_result,
            left_context="...the house that gets",
            right_context="you'll know exactly what i mean...",
        )
        if result.auto_applied:
            apply_word_reassignment(result, events)
    """

    def __init__(self, config: Optional[ArbitrationConfig] = None):
        self.config = config or ArbitrationConfig()
        self._llm_available: Optional[bool] = None

    def arbitrate(
        self,
        reasr_result,  # BoundaryReASRResult
        left_context: str = "",
        right_context: str = "",
        left_seg_end: float = 0.0,
        right_seg_start: float = 0.0,
    ) -> ArbitrationResult:
        """对单个边界执行 LLM 语义仲裁

        Args:
            reasr_result: 滑动窗口冗余 ASR 结果
            left_context: 上文（前几个段的文本，供上下文参考）
            right_context: 下文（后几个段的文本，供上下文参考）
            left_seg_end: 左段原始结束时间（全局坐标）
            right_seg_start: 右段原始开始时间（全局坐标）

        Returns:
            ArbitrationResult
        """
        from .boundary_reasr import SlidingWindowReASR

        # 提取窗口文本
        window_texts = SlidingWindowReASR.extract_window_texts(reasr_result)

        # 收集原始词（全局时间戳）
        original_words_left = [
            (w.start, w.end, w.word)
            for w in reasr_result.original_left_words
        ]
        original_words_right = [
            (w.start, w.end, w.word)
            for w in reasr_result.original_right_words
        ]

        # 判断争议词范围：哪些词在边界附近
        disputed_words = self._identify_disputed_words(
            original_words_left, original_words_right,
            left_seg_end, right_seg_start,
        )

        # 先尝试纯规则降级（快速路径，零成本）
        rule_result = self._rule_based_arbitration(
            reasr_result, window_texts, disputed_words,
            left_seg_end, right_seg_start,
        )
        if rule_result is not None and rule_result.confidence >= 0.9:
            rule_result.auto_applied = True
            return rule_result

        # 云端 LLM 仲裁
        if self._check_llm_available():
            try:
                return self._llm_arbitration(
                    reasr_result, window_texts,
                    left_context, right_context,
                    disputed_words,
                    left_seg_end, right_seg_start,
                )
            except Exception as e:
                logger.warning("LLM arbitration failed, using rule fallback: %s", e)
                if self.config.fallback_to_rules and rule_result is not None:
                    rule_result.auto_applied = (
                        rule_result.confidence >= self.config.auto_apply_confidence
                    )
                    return rule_result

        # 最终降级
        if rule_result is not None:
            rule_result.auto_applied = (
                rule_result.confidence >= self.config.auto_apply_confidence
            )
            return rule_result

        # 完全无法裁决 → 保持原样
        return ArbitrationResult(
            boundary_index=reasr_result.boundary_index,
            word_assignments=[],
            confidence=0.0,
            rationale="No arbitration possible — keeping original boundary",
            error="Arbitration failed",
        )

    # ------------------------------------------------------------------
    # 纯规则降级
    # ------------------------------------------------------------------

    def _rule_based_arbitration(
        self,
        reasr_result,  # BoundaryReASRResult
        window_texts: Dict[str, str],
        disputed_words: List[str],
        left_end: float,
        right_start: float,
    ) -> Optional[ArbitrationResult]:
        """纯规则仲裁（零成本，< 1ms）

        规则:
        1. 左段以逗号/分号结尾 + 右段首词是孤儿词 → "热迁移"到左段
        2. 左段以副词结尾 → 右段首词可能属于左段
        3. 窗口文本比对：如果融合窗和左扩展窗的文本都以某个词结束，该词应属左段
        """
        left_text = reasr_result.original_left_text.rstrip()
        right_text = reasr_result.original_right_text.strip()

        if not left_text or not right_text:
            return None

        sentence_endings = {".", "!", "?", "。", "！", "？"}
        left_last_char = left_text[-1] if left_text else ""
        right_words = right_text.split()
        left_words = left_text.split()

        assignments: List[WordAssignment] = []
        confidence = 0.5

        # 规则 1：左段以句号结尾但疑似假句号
        if left_last_char in sentence_endings:
            last_word = left_words[-1].rstrip(".!?。！？,，;；")
            from .boundary_confidence import ORPHAN_ADVERBS

            if ORPHAN_ADVERBS.match(last_word):
                # 左段以副词结尾 → 看看右段首词是否是被修饰的形容词
                if right_words:
                    first_rw = right_words[0].rstrip(",.!?，。！？")
                    from .boundary_confidence import ORPHAN_ADJECTIVES
                    if ORPHAN_ADJECTIVES.match(first_rw):
                        # 副词 + 形容词 → 应在一起
                        assignments.append(WordAssignment(
                            word=last_word, segment="left", confidence=1.0,
                        ))
                        assignments.append(WordAssignment(
                            word=first_rw, segment="left", confidence=0.85,
                        ))
                        confidence = 0.85

        # 规则 2：右段首词是孤儿词 + 间隙 < 50ms
        if right_words and not assignments:
            first_rw = right_words[0].rstrip(",.!?，。！？")
            from .boundary_confidence import (
                ORPHAN_ADVERBS, ORPHAN_ADJECTIVES,
                ORPHAN_CONJUNCTIONS, ORPHAN_SINGLE_SYLLABLE,
            )
            gap = right_start - left_end
            if gap < 0.05:
                if (ORPHAN_ADVERBS.match(first_rw) or
                        ORPHAN_ADJECTIVES.match(first_rw)):
                    # 可能属于左段
                    assignments.append(WordAssignment(
                        word=first_rw, segment="left", confidence=0.7,
                    ))
                    confidence = 0.7
                elif ORPHAN_CONJUNCTIONS.match(first_rw):
                    # 连词通常在两个子句之间，但间隙极小时倾向合并到左段
                    assignments.append(WordAssignment(
                        word=first_rw, segment="left", confidence=0.6,
                    ))
                    confidence = 0.6

        # 规则 3：窗口文本比对
        if window_texts:
            left_expand = window_texts.get("left_expand", "")
            fusion = window_texts.get("fusion", "")

            # 如果左扩展窗口包含了右段的首个词，则该词应属于左段
            if left_expand and right_words:
                left_expand_ending = left_expand.rstrip().split()[-3:]
                first_rw_clean = right_words[0].rstrip(",.!?，。！？").lower()
                for w in left_expand_ending:
                    if w.rstrip(",.!?，。！？").lower() == first_rw_clean:
                        if not assignments:
                            assignments.append(WordAssignment(
                                word=right_words[0].rstrip(",.!?，。！？"),
                                segment="left", confidence=0.8,
                            ))
                            confidence = 0.8
                        break

        if not assignments:
            return None

        # 构建最终文本
        moved_words = {a.word.lower() for a in assignments if a.segment == "left"}
        final_left_words = list(left_words)
        final_right_words = list(right_words)

        # 将属于左段的词从右段首部移出
        while final_right_words and final_right_words[0].rstrip(",.!?，。！？").lower() in moved_words:
            final_left_words.append(final_right_words.pop(0))

        # 清理左段中的假句号（所有被确认的 orphan 词）
        # 词在被移动到左段后，其尾部的句号也应被清除
        from .boundary_confidence import ORPHAN_ADVERBS
        clean_left_words = []
        for w in final_left_words:
            if w and w[-1] in sentence_endings and len(w) > 1:
                stem = w.rstrip(".!?。！？")
                if ORPHAN_ADVERBS.match(stem):
                    clean_left_words.append(stem)
                    continue
            clean_left_words.append(w)
        final_left_words = clean_left_words

        final_left = " ".join(final_left_words)
        final_right = " ".join(final_right_words)

        return ArbitrationResult(
            boundary_index=reasr_result.boundary_index,
            word_assignments=assignments,
            left_text_final=final_left,
            right_text_final=final_right,
            confidence=confidence,
            rationale=f"[rule] {len(moved_words)} word(s) reassigned based on orphan patterns + window alignment",
            auto_applied=confidence >= self.config.auto_apply_confidence,
            needs_review=(
                self.config.review_threshold <= confidence < self.config.auto_apply_confidence
            ),
        )

    # ------------------------------------------------------------------
    # LLM 仲裁
    # ------------------------------------------------------------------

    def _llm_arbitration(
        self,
        reasr_result,
        window_texts: Dict[str, str],
        left_context: str,
        right_context: str,
        disputed_words: List[str],
        left_end: float,
        right_start: float,
    ) -> ArbitrationResult:
        """调用云端 LLM 进行语义仲裁"""
        import requests

        cfg = self.config

        # 构建 hypotheses 对象
        hypotheses = {
            "original_left": reasr_result.original_left_text,
            "original_right": reasr_result.original_right_text,
        }
        for key, val in window_texts.items():
            if val:
                hypotheses[key] = val

        # 构建 LLM 输入
        user_input = {
            "left_context": left_context or "(none)",
            "right_context": right_context or "(none)",
            "original_left": reasr_result.original_left_text,
            "original_right": reasr_result.original_right_text,
            "hypotheses": hypotheses,
            "disputed_words": disputed_words,
        }

        messages = [
            {"role": "system", "content": BOUNDARY_ARBITRATION_PROMPT},
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False, indent=2)},
        ]

        api_url = f"{cfg.llm_base_url}/v1/chat/completions" if cfg.llm_base_url else None
        if not api_url:
            raise ValueError("No LLM API URL configured")

        headers = {"Content-Type": "application/json"}
        if cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {cfg.llm_api_key}"

        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": cfg.llm_temperature,
            "max_tokens": 1000,
        }

        response = requests.post(
            api_url, json=payload, headers=headers,
            timeout=cfg.llm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # 解析 JSON
        llm_result = self._parse_llm_json(content)

        # 构建 ArbitrationResult
        word_assignments = []
        for wa in llm_result.get("word_assignments", []):
            word_assignments.append(WordAssignment(
                word=wa.get("word", ""),
                segment=wa.get("segment", "left"),
                confidence=float(wa.get("confidence", 0.5)),
            ))

        llm_confidence = float(llm_result.get("confidence", 0.5))
        rationale = llm_result.get("rationale", "")

        # 构建最终文本
        moved_to_left = {
            a.word.lower() for a in word_assignments
            if a.segment == "left" and a.confidence > 0.5
        }

        left_words = reasr_result.original_left_text.rstrip().split()
        right_words = reasr_result.original_right_text.strip().split()

        # 移除假句号
        sentence_endings = {".", "!", "?", "。", "！", "？"}
        if left_words and left_words[-1][-1] in sentence_endings:
            clean = left_words[-1].rstrip(".!?。！？")
            from .boundary_confidence import ORPHAN_ADVERBS
            if ORPHAN_ADVERBS.match(clean):
                left_words[-1] = clean

        while right_words and right_words[0].rstrip(",.!?，。！？").lower() in moved_to_left:
            left_words.append(right_words.pop(0))

        final_left = " ".join(left_words)
        final_right = " ".join(right_words)

        # 从融合窗获取时间戳
        from .boundary_reasr import SlidingWindowReASR
        fusion_words = SlidingWindowReASR.extract_fusion_words_global(reasr_result)

        left_end_time = None
        right_start_time = None
        if fusion_words:
            # 找仲裁后左段最后一个词的时间戳
            last_left_word = left_words[-1].rstrip(",.!?，。！？").lower() if left_words else ""
            for gs, ge, w, _ in fusion_words:
                if w.rstrip(",.!?，。！？").lower() == last_left_word:
                    left_end_time = ge
                    break
            if left_end_time is None:
                left_end_time = left_end

            # 右段开始时间 = 左段结束 + 如果在融合窗中找到下一个词
            if right_words:
                first_right_word = right_words[0].rstrip(",.!?，。！？").lower()
                for gs, ge, w, _ in fusion_words:
                    if w.rstrip(",.!?，。！？").lower() == first_right_word:
                        right_start_time = gs
                        break
            if right_start_time is None:
                right_start_time = right_start

        return ArbitrationResult(
            boundary_index=reasr_result.boundary_index,
            word_assignments=word_assignments,
            left_text_final=final_left,
            right_text_final=final_right,
            left_end_time=left_end_time,
            right_start_time=right_start_time,
            confidence=llm_confidence,
            rationale=rationale,
            auto_applied=llm_confidence >= cfg.auto_apply_confidence,
            needs_review=(
                cfg.review_threshold <= llm_confidence < cfg.auto_apply_confidence
            ),
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_disputed_words(
        left_words: List[Tuple[float, float, str]],
        right_words: List[Tuple[float, float, str]],
        left_end: float,
        right_start: float,
        window_ms: float = 0.5,
    ) -> List[str]:
        """识别边界附近的争议词

        争议词 = 距离边界 < window_ms 的词
        """
        disputed = []
        for _, end_t, word in left_words:
            if abs(end_t - left_end) < window_ms:
                disputed.append(word)
        for start_t, _, word in right_words:
            if abs(start_t - right_start) < window_ms:
                disputed.append(word)
        return disputed

    @staticmethod
    def _parse_llm_json(content: str) -> dict:
        """解析 LLM 返回的 JSON（容忍 markdown 包裹）"""
        json_str = content
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            json_str = content[start:end].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            json_str = content[start:end].strip()
        elif "{" in content:
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
        return json.loads(json_str)

    def _check_llm_available(self) -> bool:
        """检查 LLM API 是否可用"""
        if self._llm_available is not None:
            return self._llm_available

        cfg = self.config
        if cfg.llm_base_url:
            self._llm_available = True
        elif os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL"):
            self._llm_available = True
        else:
            self._llm_available = False
            logger.info("No LLM API configured — boundary arbitration will use rules only")

        return self._llm_available


# ---------------------------------------------------------------------------
# 批量仲裁
# ---------------------------------------------------------------------------

def apply_arbitration_results(
    arbitration_results: Dict[int, ArbitrationResult],
    asr_results: List[List[TranscriptionSegment]],
    segments: List[SpeechSegment],
) -> Tuple[List[List[TranscriptionSegment]], List[SpeechSegment]]:
    """将仲裁结果应用到 ASR 结果和段边界

    修改:
    - asr_results: 调整文本（词重分配）
    - segments: 调整边界时间

    Args:
        arbitration_results: {boundary_index: ArbitrationResult}
        asr_results: 各段的 ASR 结果
        segments: 各语音段

    Returns:
        (updated_asr_results, updated_segments)
    """
    for idx, arb in arbitration_results.items():
        if not arb.auto_applied or idx >= len(segments) - 1:
            continue

        # 更新左段和右段的文本
        if arb.left_text_final and idx < len(asr_results):
            if asr_results[idx]:
                asr_results[idx][-1].text = arb.left_text_final
            elif arb.left_text_final:
                # 创建占位 TranscriptionSegment
                from ..asr.base import TranscriptionSegment
                asr_results[idx] = [TranscriptionSegment(
                    text=arb.left_text_final,
                    start=0.0,
                    end=segments[idx].end - segments[idx].start,
                )]

        if arb.right_text_final and idx + 1 < len(asr_results):
            if asr_results[idx + 1]:
                asr_results[idx + 1][0].text = arb.right_text_final
            elif arb.right_text_final:
                from ..asr.base import TranscriptionSegment
                asr_results[idx + 1] = [TranscriptionSegment(
                    text=arb.right_text_final,
                    start=0.0,
                    end=segments[idx + 1].end - segments[idx + 1].start,
                )]

        # 更新时间边界
        if arb.left_end_time is not None:
            segments[idx].end = arb.left_end_time
        if arb.right_start_time is not None:
            segments[idx + 1].start = arb.right_start_time

    # 统计
    applied = sum(1 for a in arbitration_results.values() if a.auto_applied)
    review = sum(1 for a in arbitration_results.values() if a.needs_review)
    if applied > 0 or review > 0:
        logger.info(
            "Boundary arbitration: %d auto-applied, %d flagged for review",
            applied, review,
        )

    return asr_results, segments
