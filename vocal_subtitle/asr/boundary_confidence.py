"""边界置信度评估器

在 ASR 识别完成后，评估每对相邻片段边界的"清晰度"。
低置信度边界（模糊边界）触发后续的滑动窗口冗余识别。

评估维度:
1. 段间时间间隙 (gap)
2. 前段末字是否被人为添加句末标点
3. 后段首字是否属于"孤儿词"（副词、形容词、连词孤立出现）
4. 边界能量斜率（复用 BoundaryRefiner 的三帧扫描）
5. 词级时间戳是否紧贴段边界

返回:
    BoundaryConfidence(score=0.0~1.0, triggers=List[str])
    score < trigger_threshold 的边界会被标记为"需要冗余识别"
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..asr.base import TranscriptionSegment
from ..vad.base import SpeechSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 孤儿词模式：段首出现这些词时，极可能是被边界撕裂的语义碎片
# ---------------------------------------------------------------------------

# 孤立副词 —— 前面可能缺少被修饰的形容词/动词
ORPHAN_ADVERBS = re.compile(
    r'^(very|quite|rather|extremely|unbearably|really|so|too|just|only|even|'
    r'almost|nearly|barely|hardly|scarcely|completely|totally|absolutely|'
    r'especially|particularly|especially|specifically|actually|basically|'
    r'literally|definitely|certainly|probably|possibly|maybe|perhaps)$',
    re.IGNORECASE,
)

# 孤立形容词 —— 前面可能缺少修饰它的副词
ORPHAN_ADJECTIVES = re.compile(
    r'^(hot|cold|warm|cool|big|small|large|tiny|good|bad|nice|great|'
    r'loud|quiet|dark|bright|fast|slow|high|low|long|short|'
    r'beautiful|ugly|expensive|cheap|heavy|light|hard|soft|'
    r'easy|difficult|important|interesting|boring|amazing|terrible|'
    r'new|old|young|wet|dry|clean|dirty|rich|poor|strong|weak)$',
    re.IGNORECASE,
)

# 孤立连词/介词 —— 如果段首有这些词，很可能上一段尾的词被切掉了
ORPHAN_CONJUNCTIONS = re.compile(
    r'^(and|but|or|so|because|if|when|that|which|who|whom|whose|'
    r'where|while|although|though|unless|until|since|after|before|'
    r'however|therefore|meanwhile|furthermore|moreover|otherwise|'
    r'to|for|with|without|about|against|between|through|during|'
    r'also|then|now|well|okay|right|yeah|yes|no)$',
    re.IGNORECASE,
)

# 孤立单音节功能词 —— 几乎不可能是独立句首
ORPHAN_SINGLE_SYLLABLE = re.compile(
    r'^(a|an|the|is|are|was|were|be|been|has|had|have|do|does|did|'
    r'will|would|shall|should|can|could|may|might|must|'
    r'it|he|she|they|we|you|i|me|him|her|us|them|'
    r'my|your|his|its|our|their|'
    r'this|that|these|those|there|here|'
    r'not|up|down|in|on|at|by|off|out|over)$',
    re.IGNORECASE,
)

# 句末标点
SENTENCE_ENDINGS = {".", "!", "?", "。", "！", "？"}
# 非句末标点（子句边界）
CLAUSE_ENDINGS = {",", "，", ";", "；", ":", "：", "、", "—", "…"}


@dataclass
class BoundaryConfidence:
    """单个边界的置信度评估结果"""

    boundary_index: int                     # 边界索引（第 i 和第 i+1 段之间）
    score: float                            # 置信度 0.0 ~ 1.0（越高越清晰）
    triggers: List[str] = field(default_factory=list)  # 触发的低置信条件
    gap_sec: float = 0.0                    # 段间时间间隙
    energy_ratio: float = 1.0               # 边界能量斜率比
    prev_text_end: str = ""                 # 前段尾文本（最后几个词）
    next_text_start: str = ""               # 后段首文本（最前几个词）
    prev_has_fake_period: bool = False      # 前段是否被 ASR 误加句号
    next_is_orphan: bool = False            # 后段首是否孤儿词
    needs_redundancy: bool = False          # 是否需要冗余处理


@dataclass
class BoundaryRedundancyConfig:
    """边界冗余配置"""

    enabled: bool = True
    min_gap_trigger: float = 0.05           # gap < 此值触发（秒）
    max_energy_slope_trigger: float = 3.0   # 能量斜率 < 此值触发
    confidence_threshold: float = 0.5       # score < 此值需要冗余
    # 孤儿词模式（可扩展 Regex）
    orphan_patterns: List[str] = field(default_factory=lambda: [
        r'^(very|quite|rather|extremely|unbearably|really|so|too|just|only|even)$',
        r'^(hot|cold|big|small|good|bad|nice|great|loud|quiet|dark|bright)$',
        r'^(and|but|or|so|because|if|when|that|which|who|to|for|with)$',
    ])
    # 段首短词：词长 < 此值且是孤立功能词则记低分
    short_word_max_chars: int = 5


class BoundaryConfidenceEstimator:
    """边界置信度评估器

    在 ASR 完成后对每对相邻段边界打分。
    低分边界触发后续滑动窗口冗余识别。

    使用示例:
        estimator = BoundaryConfidenceEstimator(config)
        boundaries = estimator.evaluate_all(
            segments, asr_results, audio, sample_rate,
        )
        low_conf = [b for b in boundaries if b.needs_redundancy]
    """

    def __init__(self, config: Optional[BoundaryRedundancyConfig] = None):
        self.config = config or BoundaryRedundancyConfig()

    def evaluate_all(
        self,
        segments: List[SpeechSegment],
        asr_results: List[List[TranscriptionSegment]],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[BoundaryConfidence]:
        """评估所有段间边界

        Args:
            segments: 语音段列表
            asr_results: 各段的 ASR 结果列表
            audio: 原始音频（可选，用于能量分析）
            sample_rate: 采样率

        Returns:
            BoundaryConfidence 列表，长度为 len(segments) - 1
        """
        cfg = self.config
        if not cfg.enabled or len(segments) < 2:
            return []

        boundaries = []
        for i in range(len(segments) - 1):
            bc = self._evaluate_single(
                i, segments[i], segments[i + 1],
                asr_results[i], asr_results[i + 1],
                audio, sample_rate,
            )
            boundaries.append(bc)

        # 统计
        low_count = sum(1 for b in boundaries if b.needs_redundancy)
        if low_count > 0:
            logger.info(
                "Boundary confidence: %d/%d boundaries need redundancy (%.0f%%)",
                low_count, len(boundaries),
                100 * low_count / max(len(boundaries), 1),
            )

        return boundaries

    def _evaluate_single(
        self,
        idx: int,
        seg_prev: SpeechSegment,
        seg_next: SpeechSegment,
        asr_prev: List[TranscriptionSegment],
        asr_next: List[TranscriptionSegment],
        audio: Optional[np.ndarray],
        sample_rate: int,
    ) -> BoundaryConfidence:
        """评估单个边界"""
        cfg = self.config

        triggers: List[str] = []
        score = 1.0  # 起始满分

        # ---- 1. 收集文本信息 ----
        prev_text = self._join_asr_text(asr_prev)
        next_text = self._join_asr_text(asr_next)

        prev_end_words = self._get_last_words(prev_text, n=3)
        next_start_words = self._get_first_words(next_text, n=3)

        # ---- 2. 段间间隙 ----
        gap = seg_next.start - seg_prev.end
        gap = max(0.0, gap)  # 负间隙（重叠）按 0 处理

        if gap < cfg.min_gap_trigger:
            score -= 0.25
            triggers.append(f"micro_gap:{gap*1000:.0f}ms")

        # ---- 3. 前段尾句号检测 ----
        prev_has_fake_period = False
        if prev_text:
            last_char = prev_text.rstrip()[-1]
            last_word = prev_text.rstrip().split()[-1] if prev_text.rstrip().split() else ""

            # 前段以句号结尾 且 最后一个词以逗号/连词结尾的子句 或 不是完整句
            if last_char in SENTENCE_ENDINGS:
                # 检查是否是"假句号"——文本不是自然句子结尾
                if self._is_likely_fake_period(prev_text):
                    prev_has_fake_period = True
                    score -= 0.30
                    triggers.append("fake_period")

        # ---- 4. 后段首孤儿词检测 ----
        next_is_orphan = False
        if next_text:
            first_word = next_text.strip().split()[0] if next_text.strip().split() else ""
            first_word_clean = first_word.rstrip(",.;:!?，。；：！？").lower()

            if len(first_word_clean) <= cfg.short_word_max_chars:
                if (ORPHAN_ADVERBS.match(first_word_clean) or
                        ORPHAN_ADJECTIVES.match(first_word_clean) or
                        ORPHAN_CONJUNCTIONS.match(first_word_clean) or
                        ORPHAN_SINGLE_SYLLABLE.match(first_word_clean)):
                    next_is_orphan = True
                    score -= 0.25
                    triggers.append(f"orphan_word:{first_word_clean}")

        # ---- 5. 词级时间戳边界检测 ----
        # 前段末词的时间戳是否紧贴段尾
        prev_words = self._get_all_words(asr_prev)
        if prev_words:
            last_word = prev_words[-1]
            last_word_end = getattr(last_word, "end", None)
            if last_word_end is not None:
                seg_duration = seg_prev.end - seg_prev.start
                word_to_end_gap = seg_duration - last_word_end
                if word_to_end_gap < 0.03:  # 末词 < 30ms 到段尾
                    score -= 0.10
                    triggers.append("word_hugs_end")

        # 后段首词的时间戳是否紧贴段首
        next_words = self._get_all_words(asr_next)
        if next_words:
            first_word = next_words[0]
            first_word_start = getattr(first_word, "start", None)
            if first_word_start is not None and first_word_start < 0.03:
                score -= 0.10
                triggers.append("word_hugs_start")

        # ---- 6. 能量斜率（复用 BoundaryRefiner） ----
        energy_ratio = 1.0
        if audio is not None:
            energy_ratio = self._estimate_energy_ratio(
                audio, sample_rate, seg_prev.end, "offset",
            )
            if energy_ratio < cfg.max_energy_slope_trigger:
                score -= 0.15
                triggers.append(f"gradual_energy:{energy_ratio:.1f}")

        # ---- 汇总 ----
        score = max(0.0, min(1.0, score))
        needs = score < cfg.confidence_threshold

        return BoundaryConfidence(
            boundary_index=idx,
            score=round(score, 3),
            triggers=triggers,
            gap_sec=round(gap, 3),
            energy_ratio=round(energy_ratio, 2),
            prev_text_end=prev_end_words,
            next_text_start=next_start_words,
            prev_has_fake_period=prev_has_fake_period,
            next_is_orphan=next_is_orphan,
            needs_redundancy=needs,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _join_asr_text(asr_segs: List[TranscriptionSegment]) -> str:
        """将 ASR 结果列表拼接为完整文本"""
        if not asr_segs:
            return ""
        return " ".join(ts.text for ts in asr_segs).strip()

    @staticmethod
    def _get_last_words(text: str, n: int = 3) -> str:
        """获取文本最后 n 个词"""
        if not text:
            return ""
        words = text.rstrip().split()
        return " ".join(words[-n:]) if len(words) >= n else " ".join(words)

    @staticmethod
    def _get_first_words(text: str, n: int = 3) -> str:
        """获取文本最前 n 个词"""
        if not text:
            return ""
        words = text.strip().split()
        return " ".join(words[:n]) if len(words) >= n else " ".join(words)

    @staticmethod
    def _get_all_words(asr_segs: List[TranscriptionSegment]) -> List:
        """从 ASR 结果中提取所有词级时间戳"""
        all_words = []
        for seg in asr_segs:
            if seg.words:
                all_words.extend(seg.words)
        return all_words

    @staticmethod
    def _is_likely_fake_period(text: str) -> bool:
        """判断文本末尾的句号是否是 ASR 误加的

        规则:
        - 文本以非句末标点结尾时 ASR 不会加句号 → 有句号可能为真
        - 文本很短（< 4 词）→ 可能是真句末
        - 文本以副词结尾 → 很可能是假句号（adverb 通常修饰后文）
        - 文本以介词/连词结尾 → 很可能是假句号
        - 文本以逗号/分号结尾且 ASR 追加了句号 → 假句号
        """
        stripped = text.rstrip()
        if not stripped:
            return False

        # 检查倒数第二个字符是否是自然子句边界
        if len(stripped) >= 2 and stripped[-1] in SENTENCE_ENDINGS:
            # 如果文本很短（< 5 词），大概率是真句末
            word_count = len(stripped.split())
            if word_count < 5:
                return False

            last_word = stripped.split()[-1].rstrip(".!?。！？,，;；:：")
            # 副词结尾 → 假句号
            if ORPHAN_ADVERBS.match(last_word):
                return True
            # 连词/介词结尾 → 假句号
            if ORPHAN_CONJUNCTIONS.match(last_word):
                return True

        return False

    @staticmethod
    def _estimate_energy_ratio(
        audio: np.ndarray,
        sample_rate: int,
        boundary_time: float,
        direction: str,
        check_frames: int = 3,
        frame_ms: int = 10,
    ) -> float:
        """估算边界处的能量变化比率

        复用 BoundaryRefiner 的三帧扫描逻辑。
        Rate > 5 表示阶跃跳变（清晰边界），< 3 表示渐变（模糊边界）。
        """
        try:
            center_sample = int(boundary_time * sample_rate)
            frame_size = int(frame_ms / 1000 * sample_rate)
            if frame_size < 1:
                return 1.0

            energies = []
            for i in range(-check_frames, check_frames):
                frame_start = center_sample + i * frame_size
                frame_end = frame_start + frame_size
                if 0 <= frame_start < len(audio) and frame_end <= len(audio):
                    frame = audio[frame_start:frame_end]
                    energies.append(float(np.sqrt(np.mean(frame ** 2))))
                else:
                    energies.append(0.0)

            if len(energies) < check_frames * 2:
                return 1.0

            pre_energy = max(energies[:check_frames])
            post_energy = max(energies[check_frames:])

            if direction == "offset":
                return pre_energy / max(post_energy, 1e-8)
            else:
                return post_energy / max(pre_energy, 1e-8)

        except Exception:
            return 1.0

    def get_low_confidence_boundaries(
        self,
        boundaries: List[BoundaryConfidence],
    ) -> List[int]:
        """提取需要冗余处理的边界索引列表"""
        return [b.boundary_index for b in boundaries if b.needs_redundancy]
