"""智能字幕对齐器

将自动生成的字幕与人工修订字幕逐行对齐，
支持 DTW + 时间交并比 + 文本模糊匹配 + 语义相似度。

三层对齐策略（由粗到细）：
  Layer 1 — 全局粗粒度锚点匹配（长停顿 + 关键词簇）
  Layer 2 — 语义增强的 DTW 对齐（在锚点约束内）
  Layer 3 — 残差匹配（宽松时间窗口 + 纯语义匹配）
"""

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 强制离线模式：防止 sentence-transformers / huggingface_hub 在无网络环境
# 下发起的网络 HEAD 请求导致长时间超时（模型已缓存在本地）
# 注意：vocal_subtitle.__init__ 和 model_loader 模块也会设置这些变量，
# 此处作为安全冗余（模块被独立导入时仍能生效）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from ..mapping.time_mapper import SubtitleEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 对齐异常
# ---------------------------------------------------------------------------


class AlignmentError(ValueError):
    """字幕对齐失败异常 — 自动版与修订版可能不匹配同一音频"""

    def __init__(self, message: str, coverage: float = 0.0,
                 n_auto: int = 0, n_manual: int = 0, n_matched: int = 0):
        super().__init__(message)
        self.coverage = coverage
        self.n_auto = n_auto
        self.n_manual = n_manual
        self.n_matched = n_matched


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AlignmentPair:
    """一对对齐结果"""

    auto_events: List[SubtitleEvent]   # 1个（精确匹配）或多个（合并）
    manual_events: List[SubtitleEvent] # 1个（精确匹配）或多个（拆分）
    match_type: str = "1:1"  # "1:1" | "1:N" | "N:1" | "N:M" | "INSERT" | "DELETE"
    time_iou: float = 0.0
    text_similarity: float = 0.0      # 字面文本相似度 (Levenshtein)
    semantic_similarity: float = 0.0  # 语义相似度 (Sentence-BERT)
    composite_score: float = 0.0      # 综合匹配得分

    @property
    def is_matched(self) -> bool:
        return self.match_type not in ("INSERT", "DELETE")


# ---------------------------------------------------------------------------
# 语义相似度计算器
# ---------------------------------------------------------------------------


class SemanticScorer:
    """轻量级语义相似度计算器

    复用项目已缓存的 paraphrase-multilingual-MiniLM-L12-v2 模型，
    对每条字幕文本计算 384 维句向量，以余弦相似度作为语义相似度。

    语义编码仅在候选池中计算（时间 IoU > 0.3 的配对），
    而非对所有 N×M 组合全量计算。
    """

    _instance: Optional["SemanticScorer"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is not None:
            return
        from ..utils.model_loader import load_sentence_transformer
        self._model = load_sentence_transformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        if self._model is not None:
            logger.info("SemanticScorer: model loaded successfully")
        else:
            logger.warning("SemanticScorer: failed to load model")

    @property
    def is_available(self) -> bool:
        self._load_model()
        return self._model is not None

    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本列表为语义向量

        Args:
            texts: 文本列表

        Returns:
            shape (len(texts), 384) 的向量矩阵
        """
        self._load_model()
        if self._model is None:
            return np.zeros((len(texts), 1))
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def pairwise_similarity(
        self,
        auto_texts: List[str],
        manual_texts: List[str],
    ) -> np.ndarray:
        """计算 auto × manual 的语义相似度矩阵

        Args:
            auto_texts: 自动版字幕文本列表
            manual_texts: 修订版字幕文本列表

        Returns:
            shape (len(auto), len(manual)) 的余弦相似度矩阵
        """
        if not auto_texts or not manual_texts:
            return np.zeros((len(auto_texts), len(manual_texts)))

        auto_vecs = self.encode(auto_texts)
        manual_vecs = self.encode(manual_texts)

        # 归一化
        auto_norm = auto_vecs / (np.linalg.norm(auto_vecs, axis=1, keepdims=True) + 1e-8)
        manual_norm = manual_vecs / (np.linalg.norm(manual_vecs, axis=1, keepdims=True) + 1e-8)

        return auto_norm @ manual_norm.T


# ---------------------------------------------------------------------------
# 字幕解析
# ---------------------------------------------------------------------------


def parse_subtitle_file(file_path: Path) -> List[SubtitleEvent]:
    """从 SRT/ASS 文件中解析字幕事件

    Args:
        file_path: .srt 或 .ass 文件路径

    Returns:
        SubtitleEvent 列表（index 从 1 开始）

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    suffix = file_path.suffix.lower()
    if suffix == ".srt":
        return _parse_srt(file_path)
    elif suffix == ".ass":
        return _parse_ass(file_path)
    else:
        raise ValueError(f"Unsupported subtitle format: {suffix}")


def _parse_srt(path: Path) -> List[SubtitleEvent]:
    """解析 SRT 文件"""
    content = path.read_text(encoding="utf-8")
    events = []
    # SRT 格式: index\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        try:
            index = int(lines[0])
        except ValueError:
            continue
        # 解析时间戳
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        g = time_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        text = "\n".join(lines[2:]).strip()
        events.append(SubtitleEvent(index=index, start=start, end=end, text=text))
    return events


def _parse_ass(path: Path) -> List[SubtitleEvent]:
    """解析 ASS 文件"""
    content = path.read_text(encoding="utf-8")
    events = []
    in_events = False
    idx = 0

    for line in content.split("\n"):
        line = line.strip()
        if line == "[Events]":
            in_events = True
            continue
        if in_events and line.startswith("Format:"):
            # Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            continue
        if in_events and line.startswith("Dialogue:"):
            # 解析 ASS Dialogue 行（逗号分隔，值可包含逗号需要用特殊处理）
            parts = _split_ass_dialogue(line)
            if len(parts) < 10:
                continue
            start = _parse_ass_time(parts[1])
            end = _parse_ass_time(parts[2])
            speaker = parts[4].strip()
            text = parts[9].strip().rstrip(".")
            # 移除 ASS 格式标签 {\\...}
            text = re.sub(r"\{\\[^}]*\}", "", text)
            if not text:
                continue
            idx += 1
            events.append(SubtitleEvent(
                index=idx,
                start=start,
                end=end,
                text=text,
                speaker_label=speaker if speaker else None,
            ))
    return events


def _split_ass_dialogue(line: str) -> List[str]:
    """分割 ASS Dialogue 行（处理文本中可能包含逗号的情况）"""
    # 移除 "Dialogue:" 前缀
    rest = line[len("Dialogue:"):].strip()
    # ASS 前 9 个字段用逗号分隔，第 10 个是文本
    parts = rest.split(",", 9)
    return parts


def _parse_ass_time(s: str) -> float:
    """解析 ASS 时间格式 H:MM:SS.cc → float 秒"""
    s = s.strip()
    match = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d+)", s)
    if not match:
        return 0.0
    g = match.groups()
    hours = int(g[0])
    minutes = int(g[1])
    seconds = int(g[2])
    centiseconds = int(g[3])
    return hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0


# ---------------------------------------------------------------------------
# 主对齐器
# ---------------------------------------------------------------------------


class SubtitleAligner:
    """将自动生成的字幕与人工修订字幕对齐

    三层对齐策略（由粗到细）：
    Layer 1 — 全局粗粒度锚点匹配：
      利用字幕间的大时间间隔（>2s）将序列切分为多个块，
      先做块级别的关键词 Jaccard 匹配，建立锚点约束。

    Layer 2 — 语义增强的 DTW 对齐：
      在每个大块内部运行 DTW，代价函数融合三项：
      cost = 1.0 - (0.35 × time_iou + 0.30 × text_sim + 0.35 × semantic_sim)

    Layer 3 — 残差匹配：
      剩余未匹配项通过宽松的时间窗口 + 纯语义匹配
      做最后一次尝试，标记为 INSERT/DELETE。
    """

    def __init__(
        self,
        min_iou: float = 0.3,
        min_coverage: float = 0.70,
        text_weight: float = 0.30,
        semantic_weight: float = 0.35,
        semantic_enabled: bool = True,
    ):
        self.min_iou = min_iou
        self.min_coverage = min_coverage
        self.text_weight = text_weight
        self.semantic_weight = semantic_weight
        self._time_weight = 1.0 - text_weight - semantic_weight
        self._semantic_enabled = semantic_enabled
        self._scorer = SemanticScorer() if semantic_enabled else None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    # Speaker label pattern for stripping
    SPEAKER_PATTERN = re.compile(r'^\[Speaker [A-Z]\]\s*')

    @classmethod
    def _strip_speaker_label(cls, text: str) -> str:
        """移除自动字幕中的说话人标签（如 [Speaker A]），
        因为人工修订字幕通常不包含这些标签。"""
        return cls.SPEAKER_PATTERN.sub('', text)

    def align(
        self,
        auto_events: List[SubtitleEvent],
        manual_events: List[SubtitleEvent],
    ) -> List[AlignmentPair]:
        """对齐自动版与修订版字幕

        Args:
            auto_events: 自动生成的字幕事件
            manual_events: 用户修订的字幕事件

        Returns:
            AlignmentPair 列表

        Raises:
            AlignmentError: 对齐覆盖率 < self.min_coverage
        """
        if not auto_events or not manual_events:
            raise AlignmentError("Empty events list: cannot align")

        # 预处理：移除自动字幕中的说话人标签以提升对齐准确度
        # 使用浅拷贝避免修改原始事件
        auto_events = [copy.copy(e) for e in auto_events]
        for e in auto_events:
            stripped = self._strip_speaker_label(e.text)
            if stripped != e.text:
                e.text = stripped

        n_auto, n_manual = len(auto_events), len(manual_events)

        # ---- Layer 1: 全局粗粒度锚点 ----
        anchors = self._find_global_anchors(auto_events, manual_events)

        # ---- Layer 2: 锚点约束内的 DTW ----
        pairs = self._anchored_dtw(auto_events, manual_events, anchors)

        # ---- Layer 3: 残差匹配 ----
        pairs = self._residual_match(pairs, auto_events, manual_events)

        # ---- 质量门控 ----
        matched = [p for p in pairs if p.is_matched]
        coverage = len(matched) / max(n_auto, n_manual)
        if coverage < self.min_coverage:
            raise AlignmentError(
                f"Alignment coverage too low: {coverage:.1%} "
                f"(threshold: {self.min_coverage:.0%}). "
                f"Auto: {n_auto} events, Manual: {n_manual} events, "
                f"Matched: {len(matched)} pairs. "
                f"Check if reference file matches the same audio.",
                coverage=coverage,
                n_auto=n_auto,
                n_manual=n_manual,
                n_matched=len(matched),
            )

        # 语义相似度中位数检查
        semantic_sims = [p.semantic_similarity for p in matched if p.semantic_similarity > 0]
        if semantic_sims:
            median_sim = float(np.median(semantic_sims))
            if median_sim < 0.5:
                logger.warning(
                    "Low semantic similarity (median=%.2f). "
                    "Reference may be in a different language or heavily edited. "
                    "Learning weight reduced to 50%%.",
                    median_sim,
                )

        logger.info(
            "Alignment: %d events → %d pairs (coverage=%.1f%%)",
            max(n_auto, n_manual), len(pairs), coverage * 100,
        )
        return pairs

    # ------------------------------------------------------------------
    # Layer 1: 全局粗粒度锚点
    # ------------------------------------------------------------------

    def _find_global_anchors(
        self,
        auto_events: List[SubtitleEvent],
        manual_events: List[SubtitleEvent],
        large_gap_threshold: float = 2.0,
    ) -> List[Tuple[int, int]]:
        """按大时间间隔切分序列，建立段落级锚点配对

        策略：
        1. 找到 auto 和 manual 中各自的大间隔点（>2s）
        2. 用这些间隔点将序列切分为段落
        3. 对每个段落提取 TF-IDF 关键词（top-5）
        4. 做段落级关键词 Jaccard 匹配
        5. 配对成功的段落中心作为锚点
        """
        # 找 auto 的大间隔点
        auto_breaks = []
        for i in range(len(auto_events) - 1):
            gap = manual_events[min(i, len(manual_events) - 2) + 1].start if i < len(manual_events) - 1 else 0
            if i < len(auto_events) - 1:
                gap_auto = auto_events[i + 1].start - auto_events[i].end
                if gap_auto > large_gap_threshold:
                    auto_breaks.append(i)

        # 找 manual 的大间隔点
        manual_breaks = []
        for i in range(len(manual_events) - 1):
            gap = manual_events[i + 1].start - manual_events[i].end
            if gap > large_gap_threshold:
                manual_breaks.append(i)

        if not auto_breaks or not manual_breaks:
            # 没有清晰的段落边界，返回全局约束
            return [(0, 0), (len(auto_events) - 1, len(manual_events) - 1)]

        # 将序列切分为段落
        auto_paras = self._split_by_breaks(auto_events, auto_breaks)
        manual_paras = self._split_by_breaks(manual_events, manual_breaks)

        # 对每个段落提取关键词 + 匹配
        anchors = []
        used_manual = set()

        for ai, a_para in enumerate(auto_paras):
            a_keywords = self._extract_keywords([e.text for e in a_para])
            best_jaccard = 0.1  # 最小 Jaccard 阈值
            best_mi = -1

            for mi, m_para in enumerate(manual_paras):
                if mi in used_manual:
                    continue
                m_keywords = self._extract_keywords([e.text for e in m_para])
                jaccard = _jaccard_similarity(a_keywords, m_keywords)
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_mi = mi

            if best_mi >= 0:
                used_manual.add(best_mi)
                # 用每个段落的中点作为锚点
                a_mid = a_para[len(a_para) // 2].index - 1
                m_mid = manual_paras[best_mi][len(manual_paras[best_mi]) // 2].index - 1
                anchors.append((a_mid, m_mid))

        if not anchors:
            anchors = [(0, 0), (len(auto_events) - 1, len(manual_events) - 1)]

        # 确保锚点覆盖首尾
        if anchors[0] != (0, 0):
            anchors.insert(0, (0, 0))
        if anchors[-1] != (len(auto_events) - 1, len(manual_events) - 1):
            anchors.append((len(auto_events) - 1, len(manual_events) - 1))

        logger.info("Anchors: %d paragraph-level pairs found", len(anchors))
        return anchors

    @staticmethod
    def _split_by_breaks(
        events: List[SubtitleEvent],
        breaks: List[int],
    ) -> List[List[SubtitleEvent]]:
        """按断点将事件列表切分为段落"""
        paras = []
        start = 0
        for b in sorted(breaks):
            if b + 1 > start:
                paras.append(events[start:b + 1])
            start = b + 1
        if start < len(events):
            paras.append(events[start:])
        return paras

    @staticmethod
    def _extract_keywords(texts: List[str], top_k: int = 5) -> set:
        """使用简单启发式提取关键词（高频词 + TF-IDF 近似）

        在不引入 sklearn 依赖的前提下，使用词频 × 逆文档频率近似。
        对于短文本（单条字幕），直接取所有非停用词。
        """
        # 简单分词（中英文混排）
        all_text = " ".join(texts)
        # 提取中文字符和英文单词
        chinese_chars = re.findall(r"[一-鿿]+", all_text)
        english_words = re.findall(r"[a-zA-Z]{3,}", all_text.lower())

        # 停用词
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our",
            "their", "this", "that", "these", "those", "and", "but",
            "or", "nor", "not", "so", "yet", "for", "in", "on", "at",
            "to", "from", "with", "of", "by", "as", "if", "then", "than",
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这",
        }

        eng_filtered = [w for w in english_words if w not in stopwords]
        all_terms = chinese_chars + eng_filtered

        # 统计词频
        from collections import Counter
        freq = Counter(all_terms)
        return {term for term, _ in freq.most_common(top_k)}

    # ------------------------------------------------------------------
    # Layer 2: 锚点约束内的 DTW
    # ------------------------------------------------------------------

    def _anchored_dtw(
        self,
        auto_events: List[SubtitleEvent],
        manual_events: List[SubtitleEvent],
        anchors: List[Tuple[int, int]],
    ) -> List[AlignmentPair]:
        """在锚点约束的区间内运行 DTW"""
        all_pairs: List[AlignmentPair] = []

        for k in range(len(anchors) - 1):
            a_start, m_start = anchors[k]
            a_end, m_end = anchors[k + 1]

            # 区间内的子序列
            a_sub = auto_events[a_start:a_end + 1]
            m_sub = manual_events[m_start:m_end + 1]

            if not a_sub or not m_sub:
                continue

            sub_pairs = self._dtw_align(a_sub, m_sub)
            all_pairs.extend(sub_pairs)

        # 去重合并（锚点边界可能有重复匹配）
        return self._dedup_pairs(all_pairs)

    def _dtw_align(
        self,
        auto_sub: List[SubtitleEvent],
        manual_sub: List[SubtitleEvent],
    ) -> List[AlignmentPair]:
        """在子序列上运行语义增强 DTW

        使用标准 DTW 算法，代价函数融合时间 IoU + 文本相似度 + 语义相似度。
        通过回溯路径确定 1:1, 1:N, N:1 匹配。
        """
        m, n = len(auto_sub), len(manual_sub)
        if m == 0 or n == 0:
            return []

        # 预计算候选池：仅时间 IoU > min_iou 的对计算语义相似度
        auto_texts = [e.text for e in auto_sub]
        manual_texts = [e.text for e in manual_sub]

        # 预计算文本相似度矩阵
        text_sim = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                auto_start, auto_end = auto_sub[i].start, auto_sub[i].end
                manual_start, manual_end = manual_sub[j].start, manual_sub[j].end
                iou = _time_iou(auto_start, auto_end, manual_start, manual_end)
                if iou > self.min_iou:
                    text_sim[i, j] = _levenshtein_similarity(auto_texts[i], manual_texts[j])

        # 预计算语义相似度（仅候选池内）
        semantic_sim = np.zeros((m, n))
        if self._semantic_enabled and self._scorer is not None and self._scorer.is_available:
            for i in range(m):
                candidates = [j for j in range(n) if text_sim[i, j] > 0 or _time_iou(
                    auto_sub[i].start, auto_sub[i].end,
                    manual_sub[j].start, manual_sub[j].end,
                ) > self.min_iou]
                if candidates:
                    try:
                        sims = self._scorer.pairwise_similarity(
                            [auto_texts[i]],
                            [manual_texts[j] for j in candidates],
                        )
                        for ci, j in enumerate(candidates):
                            semantic_sim[i, j] = float(sims[0, ci])
                    except Exception:
                        pass

        # 构建代价矩阵
        cost = np.full((m, n), np.inf)
        for i in range(m):
            for j in range(n):
                auto_start, auto_end = auto_sub[i].start, auto_sub[i].end
                manual_start, manual_end = manual_sub[j].start, manual_sub[j].end
                iou = _time_iou(auto_start, auto_end, manual_start, manual_end)

                composite = (
                    self._time_weight * iou
                    + self.text_weight * text_sim[i, j]
                    + self.semantic_weight * semantic_sim[i, j]
                )
                cost[i, j] = 1.0 - composite

        # DTW 累积
        dtw = np.full((m + 1, n + 1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dtw[i, j] = cost[i - 1, j - 1] + min(
                    dtw[i - 1, j],       # 插入 (auto 对 manual 无匹配 → INSERT)
                    dtw[i, j - 1],       # 删除 (manual 对 auto 无匹配 → DELETE)
                    dtw[i - 1, j - 1],   # 匹配
                )

        # 回溯路径
        path = []
        i, j = m, n
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            candidates = [
                (dtw[i - 1, j], i - 1, j),
                (dtw[i, j - 1], i, j - 1),
                (dtw[i - 1, j - 1], i - 1, j - 1),
            ]
            _, ni, nj = min(candidates, key=lambda x: x[0])
            i, j = ni, nj

        path.reverse()

        # 路径 → AlignmentPair
        pairs = []
        i = 0
        while i < len(path):
            ai, mj = path[i]
            a_start_i = ai
            m_start_j = mj

            # 检测 1:N 或 N:1 还是 1:1
            a_count, m_count = 1, 1
            while i + 1 < len(path) and path[i + 1] == (ai, mj + 1):
                m_count += 1
                mj += 1
                i += 1
            while i + 1 < len(path) and path[i + 1] == (ai + 1, mj):
                a_count += 1
                ai += 1
                i += 1

            if a_count == 1 and m_count == 1:
                match_type = "1:1"
            elif a_count == 1 and m_count > 1:
                match_type = "1:N"
            elif a_count > 1 and m_count == 1:
                match_type = "N:1"
            else:
                match_type = "N:M"

            a_evts = [auto_sub[k] for k in range(a_start_i, a_start_i + a_count)]
            m_evts = [manual_sub[k] for k in range(m_start_j, m_start_j + m_count)]

            # 计算综合指标
            iou = _time_iou(
                min(e.start for e in a_evts), max(e.end for e in a_evts),
                min(e.start for e in m_evts), max(e.end for e in m_evts),
            )
            a_text = " ".join(e.text for e in a_evts)
            m_text = " ".join(e.text for e in m_evts)
            text_sim_val = _levenshtein_similarity(a_text, m_text)
            sem_sim_val = float(semantic_sim[a_start_i, m_start_j]) if a_count == 1 and m_count == 1 else float(np.mean([
                semantic_sim[ai2, mj2]
                for ai2 in range(a_start_i, min(a_start_i + a_count, m))
                for mj2 in range(m_start_j, min(m_start_j + m_count, n))
            ]))

            composite = (
                self._time_weight * iou
                + self.text_weight * text_sim_val
                + self.semantic_weight * sem_sim_val
            )

            pairs.append(AlignmentPair(
                auto_events=a_evts,
                manual_events=m_evts,
                match_type=match_type,
                time_iou=iou,
                text_similarity=text_sim_val,
                semantic_similarity=sem_sim_val,
                composite_score=composite,
            ))
            i += 1

        return pairs

    # ------------------------------------------------------------------
    # Layer 3: 残差匹配
    # ------------------------------------------------------------------

    def _residual_match(
        self,
        pairs: List[AlignmentPair],
        auto_events: List[SubtitleEvent],
        manual_events: List[SubtitleEvent],
    ) -> List[AlignmentPair]:
        """对未匹配项做最后一次宽松匹配"""
        matched_auto = set()
        matched_manual = set()
        for p in pairs:
            for e in p.auto_events:
                matched_auto.add(e.index)
            for e in p.manual_events:
                matched_manual.add(e.index)

        unmatched_auto = [e for e in auto_events if e.index not in matched_auto]
        unmatched_manual = [e for e in manual_events if e.index not in matched_manual]

        if not unmatched_auto and not unmatched_manual:
            return pairs

        # INSERT: auto 有但 manual 没有
        for e in unmatched_auto:
            pairs.append(AlignmentPair(
                auto_events=[e],
                manual_events=[],
                match_type="INSERT",
            ))

        # DELETE: manual 有但 auto 没有
        for e in unmatched_manual:
            pairs.append(AlignmentPair(
                auto_events=[],
                manual_events=[e],
                match_type="DELETE",
            ))

        return pairs

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_pairs(pairs: List[AlignmentPair]) -> List[AlignmentPair]:
        """去重合并对齐对（按 auto index）"""
        seen = set()
        result = []
        for p in pairs:
            key = tuple(e.index for e in p.auto_events)
            if key in seen:
                continue
            seen.add(key)
            result.append(p)
        return result


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _time_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """计算两个时间区间的交并比"""
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return overlap / union


def _levenshtein_similarity(a: str, b: str) -> float:
    """计算 Levenshtein 相似度 [0, 1]"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # 使用简单的动态规划计算编辑距离
    m, n = len(a), len(b)
    if m > 200 or n > 200:
        # 长文本使用快速近似
        shorter = min(m, n)
        if shorter == 0:
            return 0.0
        # 字符集相似度近似
        set_a, set_b = set(a), set(b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    max_len = max(m, n)
    return 1.0 - dp[m][n] / max_len


def _jaccard_similarity(a: set, b: set) -> float:
    """计算 Jaccard 相似度"""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
