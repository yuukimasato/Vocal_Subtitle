"""时间轴仲裁层(2026-09-11 定案,层2)的纯判定函数。

三规则信任策略表(设计文档 §4.3):骨架管段级真值,ASR 管词级真值,
能量检测当裁判,文本一致性决定信任级别。

- R1 共识:分段基线文本(骨架窗口 ASR 的事件文本)与全程识别 evidence
  文本按区域字符对齐一致 → 该区域整段信骨架(吸附解除 max_snap_distance
  限幅,直接钳到骨架成员段端点;词内时刻仍用 ASR)。
- R2 盲区:ASR 词落在骨架"静音区"(无物理成员可钳)时禁止吸附裁词,
  改用 rms 能量确认;`arbitration_r2_local_noise` 开启时噪声底取词周边
  局部窗口,防止音乐残留等非均匀噪声被全局噪声底误确认。
- R3 空洞:骨架有语音、ASR 无词覆盖 → 覆盖审计 + LocalRecovery,见
  `physical/coverage.py`(空洞 × turns 换人边界联动),本模块不涉及。

本模块只做判定,不修改事件;应用点在 `validator._physical_snap_validation`。
文本归一化复用 `asr.risk_scoring.normalize_text`,与既有风险评分保持
同一约定,不引入新依赖。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from ..utils.audio_utils import AudioUtils

# R2 局部噪声窗口半径:词时刻前后各 1.5s 的底部帧估计局部噪声底。
LOCAL_NOISE_HALF_WINDOW = 1.5
# R2 能量确认阈值比:词 RMS 需超过噪声底的倍数(与 rms_energy_check 一致)。
R2_ENERGY_THRESHOLD_RATIO = 2.0


@dataclass(frozen=True)
class R1Decision:
    """单个事件的 R1 共识判定结果。"""

    similarity: float  # 对齐重合字符 / 较短方字符数
    overlap_chars: int  # 字符对齐的一致字符数
    evidence_start: float  # 命中的 evidence 候选起点
    evidence_end: float  # 命中的 evidence 候选终点


def char_overlap(baseline: str, reference: str) -> tuple[int, float]:
    """按字符对齐比较两段文本。

    Returns:
        (一致字符数, 重合率)。重合率 = 对齐重合字符 / 较短方字符数,
        与配置注释 `arbitration_r1_min_similarity` 的定义一致。
    """
    from ..asr.risk_scoring import normalize_text

    left = normalize_text(baseline)
    right = normalize_text(reference)
    if not left or not right:
        return 0, 0.0
    matcher = SequenceMatcher(None, left, right)
    overlap = sum(block.size for block in matcher.get_matching_blocks())
    denominator = min(len(left), len(right))
    if denominator <= 0:
        return 0, 0.0
    return overlap, overlap / denominator


def coerce_evidence_regions(
    evidence: Sequence[Any] | None,
) -> tuple[tuple[float, float, str], ...]:
    """把全程 evidence 候选归一为 (start, end, text) 三元组序列。

    接受带 start/end/text 属性的对象(如 CandidateEvidence)或直接的
    三元组;无法解析的条目静默跳过(evidence 是参照文本,不是主路径)。
    """
    regions: list[tuple[float, float, str]] = []
    for item in evidence or ():
        if isinstance(item, (tuple, list)):
            if len(item) != 3:
                continue
            start, end, text = item
        else:
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)
            text = getattr(item, "text", None)
        if start is None or end is None:
            continue
        try:
            regions.append((float(start), float(end), str(text or "")))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(regions, key=lambda item: (item[0], item[1])))


def r1_consensus_decisions(
    events: Sequence[Any],
    evidence_regions: Sequence[tuple[float, float, str]],
    *,
    min_overlap_chars: int,
    min_similarity: float,
) -> dict[int, R1Decision]:
    """按事件判定 R1 共识(分段基线 vs 全程 evidence)。

    对每个事件,取与其时间范围有正重叠的 evidence 候选逐一做字符对齐,
    取一致字符数最多者;一致字符数与重合率同时达标才记为共识。

    Returns:
        {id(event): R1Decision};无 evidence 或无达标事件时为空 dict。
    """
    decisions: dict[int, R1Decision] = {}
    if not evidence_regions:
        return decisions
    for event in events:
        baseline = str(getattr(event, "text", "") or "")
        if not baseline.strip():
            continue
        start = float(getattr(event, "start", 0.0))
        end = float(getattr(event, "end", 0.0))
        best: R1Decision | None = None
        for region_start, region_end, text in evidence_regions:
            if region_start >= end or region_end <= start:
                continue
            overlap, similarity = char_overlap(baseline, text)
            if overlap < min_overlap_chars or similarity < min_similarity:
                continue
            # 一致字符数优先;并列时取重合率更高者。
            if best is None or (overlap, similarity) > (
                best.overlap_chars,
                best.similarity,
            ):
                best = R1Decision(
                    similarity=round(similarity, 6),
                    overlap_chars=int(overlap),
                    evidence_start=region_start,
                    evidence_end=region_end,
                )
        if best is not None:
            decisions[id(event)] = best
    return decisions


def best_skeleton_segment(
    start: float,
    end: float,
    skeleton: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """返回与 [start, end] 重叠最大的骨架成员段;无正重叠返回 None。"""
    best: tuple[float, float] | None = None
    best_overlap = 0.0
    for seg_start, seg_end in skeleton:
        overlap = min(end, seg_end) - max(start, seg_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (float(seg_start), float(seg_end))
    return best


def event_word_spans(event: Any) -> list[tuple[float, float]]:
    """事件的词绝对时间跨度列表。

    词时间契约:SubtitleEvent.words 相对 event.start(time_mapper 契约);
    物理路径的 GlobalWord 带 raw_start/raw_end 绝对坐标,优先使用。
    """
    event_start = float(getattr(event, "start", 0.0) or 0.0)
    spans: list[tuple[float, float]] = []
    for word in getattr(event, "words", None) or ():
        start = getattr(word, "raw_start", None)
        end = getattr(word, "raw_end", None)
        absolute = start is not None and end is not None
        if not absolute:
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
        if start is None or end is None:
            continue
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        if not absolute:
            start += event_start
            end += event_start
        if end > start:
            spans.append((start, end))
    return spans


def words_beyond(
    spans: Sequence[tuple[float, float]],
    boundary: float,
    side: str,
) -> list[tuple[float, float]]:
    """找出会被端点吸附到 boundary 裁掉的词。

    end 回缩裁词尾(start < boundary < end),start 后移裁词头
    (start < boundary < end);side 指明移动的是哪个端点。
    """
    if side not in {"start", "end"}:
        raise ValueError("side must be 'start' or 'end'")
    cut: list[tuple[float, float]] = []
    for word_start, word_end in spans:
        if side == "end" and word_end > boundary + 1e-6:
            cut.append((word_start, word_end))
        elif side == "start" and word_start < boundary - 1e-6:
            cut.append((word_start, word_end))
    return cut


def local_silence_rms(
    audio: Any,
    sample_rate: int,
    center: float,
    *,
    half_window: float = LOCAL_NOISE_HALF_WINDOW,
) -> float | None:
    """词周边局部窗口的噪声底(底部 20% 帧 RMS 中位数)。

    与 AudioUtils.estimate_silence_rms 同语义,但只采样词周边
    [center-half_window, center+half_window];帧太少的极短音频返回
    None(调用方回退全局噪声底)。
    """
    frame_size = int(0.01 * sample_rate)
    step = int(0.05 * sample_rate)
    if frame_size < 1:
        return None
    total_samples = len(audio)
    start_sample = max(0, int((center - half_window) * sample_rate))
    end_sample = min(
        total_samples - frame_size, int((center + half_window) * sample_rate)
    )
    if end_sample - start_sample < frame_size:
        return None
    rms_samples = []
    for pos in range(start_sample, end_sample, step):
        frame = audio[pos : pos + frame_size]
        if len(frame) == 0:
            continue
        rms_samples.append(float(np.sqrt(np.mean(np.asarray(frame) ** 2))))
    if not rms_samples:
        return None
    rms_samples.sort()
    take = max(1, int(round(len(rms_samples) * 0.2)))
    return float(np.median(rms_samples[:take]))


def word_speech_confirmed(
    audio: Any,
    sample_rate: int,
    word_span: tuple[float, float],
    *,
    local_noise: bool,
    threshold_ratio: float = R2_ENERGY_THRESHOLD_RATIO,
) -> bool:
    """R2 能量确认:词时刻 RMS 是否显著高于所选噪声底。

    local_noise=True 用词周边局部噪声底(非均匀噪声场景),
    False 用全局噪声底(与现行 rms_energy_check 一致)。
    音频缺失时无法确认,返回 False(保守,交由幻觉裁剪路径)。
    """
    if audio is None:
        return False
    total = len(audio) / sample_rate
    start = max(0.0, float(word_span[0]))
    end = min(total, float(word_span[1]))
    if end <= start:
        return False
    if local_noise:
        floor = local_silence_rms(audio, sample_rate, (start + end) / 2.0)
        if floor is None:
            floor = AudioUtils.estimate_silence_rms(audio, sample_rate)
    else:
        floor = AudioUtils.estimate_silence_rms(audio, sample_rate)
    word_rms = AudioUtils.get_segment_rms(audio, start, end, sample_rate)
    return word_rms > floor * threshold_ratio
