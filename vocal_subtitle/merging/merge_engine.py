"""LLM 语义合并引擎 (方案五)

将声学合并与语义合并完全分离：声学层激进切分，语义层用 LLM 决策合并。

核心架构: 三级决策流水线 (Fast-Slow Path)

  相邻片段间隔 < 200ms  ──→ 快路径（规则强制合并）
  相邻片段间隔 200-1200ms ──→ 慢路径（LLM 裁决）
  相邻片段间隔 > 1200ms   ──→ 硬规则（强制不合并）

降级策略: LLM 不可用时自动回退到 Fast-Path 纯规则模式。

帧级无缝衔接 (3.7): 非句尾字幕自动衔接到下一句，消除字幕闪烁。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.text_utils import smart_join_texts
from . import merge_constraints, merge_policy
from .llm_decider import LLMMergeDecider
from .local_decider import LocalMergeDecider

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 语义边界检测模式（用于合并降级规则）
# ------------------------------------------------------------------

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


def _detect_semantic_boundary(
    current_text: str, next_text: str,
) -> bool:
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


# ------------------------------------------------------------------
# 合并决策 Prompt
# ------------------------------------------------------------------

MERGE_DECISION_PROMPT = """You are a subtitle merging expert.
Given a sequence of subtitle fragments with precise timestamps,
decide which adjacent fragments should be merged into a single subtitle line.

Hard Rules (these OVERRIDE any semantic judgment):
1. If silence gap > 1.2 seconds → ABSOLUTELY FORBIDDEN to merge.
   Even if semantically continuous, >1.2s silence creates a "full stop"
   expectation for viewers. Merging would make subtitles appear too late.
2. If speakers are different → ABSOLUTELY FORBIDDEN to merge.
3. If combined duration > 5.0 seconds → FORBIDDEN to merge.
4. If either fragment ends with terminal punctuation (.!?。！？)
   AND gap > 400ms → default to NOT merging.

Semantic Rules (only apply when Hard Rules allow):
- Merge ONLY if the combined text forms a COMPLETE semantic unit.
- Do NOT merge if the gap represents a natural sentence boundary.
- A trailing comma or incomplete clause (e.g., "Before ending the call,")
  STRONGLY suggests merging with the next fragment.
- If gap is < 300ms and both fragments are from same speaker,
  they are likely the same sentence → merge.

Input format: JSON array of fragments with id, start, end, speaker, text, gap_to_next_sec, gap_is_silent.
Output format: JSON with merge_groups array.

Example:
Input:
[
  {"id": 1, "start": 14.85, "end": 16.09, "speaker": "A",
   "text": "Before ending the call,", "gap_to_next_sec": 0.33, "gap_is_silent": true},
  {"id": 2, "start": 16.42, "end": 17.90, "speaker": "A",
   "text": "repeat key details", "gap_to_next_sec": 4.68, "gap_is_silent": true},
  {"id": 3, "start": 22.58, "end": 23.80, "speaker": "A",
   "text": "So I have you down for a non-smoking", "gap_to_next_sec": 0.20, "gap_is_silent": true},
  {"id": 4, "start": 24.00, "end": 24.80, "speaker": "A",
   "text": "king room tomorrow,", "gap_to_next_sec": 0.20, "gap_is_silent": true},
  {"id": 5, "start": 25.00, "end": 26.18, "speaker": "A",
   "text": "is that right?", "gap_to_next_sec": null, "gap_is_silent": null}
]

Output:
{
  "merge_groups": [
    {"ids": [1, 2], "reason": "Fragment 1 ends with comma (incomplete clause). Combined with fragment 2 forms complete instruction."},
    {"ids": [3, 4, 5], "reason": "Three fragments form one complete confirmation question. Micro-pauses within sentence, not boundaries."}
  ]
}
"""


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


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------

@dataclass
class MergeDecisionConfig:
    """合并决策分流配置"""

    # Fast-Slow Path 分流阈值
    fast_merge_max_gap: float = 0.30      # <300ms: 规则强制合并（同说话人+短间隔）
    llm_decision_min_gap: float = 0.30    # 300-1200ms: LLM裁决
    llm_decision_max_gap: float = 1.20
    hard_split_min_gap: float = 1.20      # >1200ms: 强制不合并

    # 合并约束
    max_combined_duration: float = 5.0    # 合并后字幕不超过5秒
    min_fragment_duration: float = 0.15   # 最小片段时长

    # LLM 降本策略
    llm_tier: str = "cascading"           # "cascading" | "all_llm" | "rule_only"
    local_nlp_gap_range: Tuple[float, float] = (0.30, 0.60)  # 本地NLP优先的间隙范围

    # LLM API 配置
    llm_model: str = "deepseek-v4-pro"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1
    llm_timeout: float = 15.0

    # 降级
    llm_fallback_to_rules: bool = True    # LLM 失败时回退到规则


# ------------------------------------------------------------------
# LLM 合并引擎
# ------------------------------------------------------------------

class LLMMergeEngine:
    """LLM 语义合并引擎

    使用示例:
        engine = LLMMergeEngine(MergeDecisionConfig())
        merged_events = engine.merge(fragments, audio, sample_rate)
    """

    def __init__(self, config: Optional[MergeDecisionConfig] = None):
        self.config = config or MergeDecisionConfig()
        self._local_model = None  # legacy compatibility view
        self._local_model_attempted = False
        self._local_decider = LocalMergeDecider(self.config)
        self._llm_decider = LLMMergeDecider(self.config)

    def merge(
        self,
        fragments: List[Dict],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """合并决策流水线

        Args:
            fragments: 片段列表，每个片段含:
                id, start, end, text, speaker, gap_to_next_sec, gap_is_silent
            audio: 音频数组（构建合并输入时使用）
            sample_rate: 采样率

        Returns:
            合并后的片段列表
        """
        if len(fragments) <= 1:
            return fragments

        cfg = self.config

        # Step 1: 补齐间隙信息
        fragments = self._ensure_gap_info(fragments, audio, sample_rate)

        # Step 2: Fast-Path 规则强制合并
        fast_merged = self._apply_fast_merges(fragments)

        # Step 3: Hard-Split 标记
        for frag in fast_merged:
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None:
                gap = 999
            frag["_hard_split"] = gap > cfg.hard_split_min_gap

            # 不同说话人检查
            next_idx = frag.get("id", 1)  # 简化：位置相邻即检查
            if gap < cfg.hard_split_min_gap and not frag.get("_hard_split"):
                pass  # 在临界区内，需要进一步判断

        # Step 4: 收集候选并按间隙范围分流
        local_nlp_candidates = []   # local_nlp_gap_range → 本地 NLP 优先
        cloud_llm_candidates = []   # cloud_llm_gap_range → 云端 LLM
        rule_decisions = {}

        for i, frag in enumerate(fast_merged):
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None:
                gap = 999

            # 已经在 fast-path 合并过的跳过
            merged_ids = frag.get("_merged_ids", [frag.get("id", i + 1)])
            last_id = merged_ids[-1] if merged_ids else frag.get("id", i + 1)

            if gap <= cfg.fast_merge_max_gap:
                # 已在快路径处理
                continue
            elif gap > cfg.hard_split_min_gap:
                rule_decisions[last_id] = False  # 强制不合并
            elif cfg.llm_decision_min_gap <= gap <= cfg.llm_decision_max_gap:
                # 按本地/云端间隙范围分流
                local_min, local_max = cfg.local_nlp_gap_range
                if local_min <= gap <= local_max and cfg.llm_tier == "cascading":
                    local_nlp_candidates.append(frag)
                else:
                    cloud_llm_candidates.append(frag)

        # Step 5a: 本地 NLP 裁决（降本第一层，文档 5.4.1）
        local_nlp_groups = []
        unresolved_from_local = []
        if local_nlp_candidates:
            local_nlp_groups, unresolved_from_local = self._local_merge_decision(
                local_nlp_candidates,
            )
            # 未解决的升级到云端 LLM
            cloud_llm_candidates.extend(unresolved_from_local)

        # Step 5b: 云端 LLM 裁决（降本第二层）
        llm_groups = list(local_nlp_groups)
        if cloud_llm_candidates and cfg.llm_tier != "rule_only":
            try:
                llm_groups.extend(self._call_llm_merge_decision(cloud_llm_candidates))
            except Exception as e:
                logger.warning("LLM merge decision failed: %s", e)
                if cfg.llm_fallback_to_rules:
                    llm_groups.extend(
                        self._fallback_rule_decisions(cloud_llm_candidates)
                    )

        # Step 6: 应用所有决策
        return self._apply_all_decisions(fast_merged, llm_groups, rule_decisions)

    # ------------------------------------------------------------------
    # Fast Path
    # ------------------------------------------------------------------

    def _apply_fast_merges(self, fragments: List[Dict]) -> List[Dict]:
        """快路径：规则强制合并 gap < fast_merge_max_gap 的相邻片段"""
        cfg = self.config
        if len(fragments) <= 1:
            return fragments

        sentence_endings = {".", "!", "?", "。", "！", "？"}

        result = []
        i = 0
        while i < len(fragments):
            frag = fragments[i].copy()
            merged_ids = [frag.get("id", i + 1)]
            merged_texts = [frag.get("text", "")]

            # 向前看：是否可以快路径合并
            j = i + 1
            while j < len(fragments):
                prev_frag = fragments[j - 1]
                curr_frag = fragments[j]
                gap = prev_frag.get("gap_to_next_sec", 999)
                if gap is None:
                    gap = 999

                prev_text = prev_frag.get("text", "").rstrip()
                prev_speaker = prev_frag.get("speaker", "")
                curr_speaker = curr_frag.get("speaker", "")

                # 重叠保护：负间隙（重叠事件）绝不合并
                if gap < -0.02:
                    break

                # 物理所有权保护：不同 physical clip 的片段不合并
                if not _physical_owner_compatible(prev_frag, curr_frag):
                    break

                # Unknown speaker labels are not evidence that two clips are
                # the same voice. Preserve the boundary until diarization or
                # an explicit speaker label can justify a merge.
                if prev_speaker in ("", "unknown") or curr_speaker in ("", "unknown"):
                    break

                # Keep the duration contract on the fast path as well as on
                # LLM/rule decisions.  The check uses the first fragment's
                # start because the group may already contain several items.
                first_start = frag.get("start", 0)
                proposed_end = curr_frag.get("end", first_start)
                if proposed_end - first_start > cfg.max_combined_duration:
                    break

                # 快路径合并条件
                can_fast_merge = (
                    gap < cfg.fast_merge_max_gap
                    and prev_speaker == curr_speaker
                    and len(prev_text) > 0
                    and prev_text[-1] not in sentence_endings
                )

                if can_fast_merge:
                    merged_ids.append(curr_frag.get("id", j + 1))
                    merged_texts.append(curr_frag.get("text", ""))
                    j += 1
                else:
                    break

            # 产出合并后的片段
            if len(merged_ids) > 1:
                frag["start"] = min(
                    fragments[k - (j - i) + (j - i)].get("start", 0)
                    if k > 0 else frag.get("start", 0)
                    for k, _ in enumerate(merged_ids)
                )
                # 取第一个片段的 start
                first_of_group = fragments[i]
                last_of_group = fragments[j - 1]
                frag["start"] = first_of_group.get("start", frag.get("start", 0))
                frag["end"] = last_of_group.get("end", frag.get("end", 0))
                frag["text"] = smart_join_texts(merged_texts)
                frag["_merged_ids"] = merged_ids
                frag["_fast_merged"] = True

            # 更新间隙信息到下一个片段
            if j < len(fragments):
                frag["gap_to_next_sec"] = fragments[j].get("start", 0) - frag.get("end", 0)
                frag["gap_is_silent"] = fragments[j - 1].get("gap_is_silent")

            result.append(frag)
            i = j

        if len(result) != len(fragments):
            logger.info(
                "Fast-merge: %d → %d fragments (gap < %.0fms)",
                len(fragments), len(result), cfg.fast_merge_max_gap * 1000,
            )
        return result

    # ------------------------------------------------------------------
    # 本地 NLP 模型 (5.4.1)
    # ------------------------------------------------------------------

    def _load_local_model(self):
        """Compatibility hook for local model loading."""
        self._local_model = self._local_decider.load_model()
        self._local_model_attempted = self._local_decider._model_attempted
        return self._local_model

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的语义相似度

        Returns:
            0.0 ~ 1.0，模型不可用时返回 0.5（中性值）
        """
        return self._local_decider.compute_similarity(text_a, text_b)

    def _local_merge_decision(
        self,
        candidates: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Compatibility hook for local rule/model decisions."""
        return self._local_decider.decide(candidates)

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _call_llm_merge_decision(
        self, candidates: List[Dict],
    ) -> List[Dict]:
        """Compatibility hook for the cloud decision service."""
        return self._llm_decider.decide_core(candidates)

    def _fallback_rule_decisions(
        self, candidates: List[Dict],
    ) -> List[Dict]:
        """Compatibility hook for deterministic fallback decisions."""
        return self._llm_decider.fallback_rule_decisions(candidates)

    # ------------------------------------------------------------------
    # 决策应用
    # ------------------------------------------------------------------

    def _apply_all_decisions(
        self,
        fast_merged: List[Dict],
        llm_groups: List[Dict],
        rule_decisions: Dict[int, bool],
    ) -> List[Dict]:
        """应用所有合并决策，产出最终片段列表"""
        if not llm_groups:
            return fast_merged

        # 构建 id → fragment 映射
        id_map = {}
        for frag in fast_merged:
            merged_ids = frag.get("_merged_ids", [frag.get("id", 0)])
            for mid in merged_ids:
                id_map[mid] = frag

        # 合并 LLM 决策的组
        merged = []
        consumed = set()

        for group in llm_groups:
            ids = group.get("ids", [])
            if not ids:
                continue

            # 检查是否已被消费
            if any(mid in consumed for mid in ids):
                continue

            first = id_map.get(ids[0])
            last = id_map.get(ids[-1])
            if first is None or last is None:
                continue

            selected = [id_map.get(mid) for mid in ids]
            if any(item is None for item in selected):
                continue
            if any(
                not _physical_owner_compatible(left, right)
                for left, right in zip(selected, selected[1:])
            ):
                continue
            if last.get("end", 0) - first.get("start", 0) > self.config.max_combined_duration:
                logger.debug(
                    "Rejecting merge group %s: duration exceeds %.2fs",
                    ids, self.config.max_combined_duration,
                )
                continue

            # ★ 所有片段必须来自同一说话人
            speakers = set()
            for mid in ids:
                f = id_map.get(mid)
                if f:
                    spk = f.get("speaker", "")
                    if spk:
                        speakers.add(spk)
            if len(speakers) > 1:
                continue  # 不同说话人 → 拒绝合并
            if any(f.get("speaker", "") in ("", "unknown") for f in selected):
                continue  # 未知说话人不能作为合并依据

            # 拼接文本（去重：同一片段可能被多个 id 引用）
            texts = []
            seen_texts = set()
            accepted_ids = []
            for mid in ids:
                frag = id_map.get(mid)
                if frag:
                    t = frag.get("text", "").strip()
                    if t and t not in seen_texts:
                        seen_texts.add(t)
                        texts.append(t)
                    accepted_ids.append(mid)

            combined_text = smart_join_texts(texts)

            # ★ 选择合并组的说话人：优先使用第一个非 "unknown" 的标签
            merged_speaker = first.get("speaker", "unknown")
            if merged_speaker == "unknown":
                for mid in ids:
                    f = id_map.get(mid)
                    if f:
                        spk = f.get("speaker", "unknown")
                        if spk and spk != "unknown":
                            merged_speaker = spk
                            break

            merged.append({
                "id": ids[0],
                "start": first.get("start", 0),
                "end": last.get("end", 0),
                "text": combined_text,
                "speaker": merged_speaker,
                "_llm_merged": True,
                "_merged_ids": ids,
            })
            consumed.update(accepted_ids)

        # 添加未被 LLM 消费的片段
        for frag in fast_merged:
            frag_ids = frag.get("_merged_ids", [frag.get("id", 0)])
            if all(mid in consumed for mid in frag_ids):
                continue
            if frag.get("id", 0) not in consumed:
                merged.append(frag)

        # 按 start 排序
        merged.sort(key=lambda f: f.get("start", 0))

        logger.info(
            "Merge complete: %d → %d fragments (fast=%d, llm=%d)",
            len(fast_merged), len(merged),
            sum(1 for f in merged if f.get("_fast_merged")),
            sum(1 for f in merged if f.get("_llm_merged")),
        )
        return merged

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _ensure_gap_info(
        self,
        fragments: List[Dict],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """确保每个片段都有 gap_to_next_sec 和 gap_is_silent 字段"""
        from ..utils.audio_utils import AudioUtils

        silence_rms = None
        if audio is not None:
            silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)

        for i in range(len(fragments)):
            if "gap_to_next_sec" in fragments[i]:
                continue

            if i < len(fragments) - 1:
                next_start = fragments[i + 1].get("start", 0)
                curr_end = fragments[i].get("end", 0)
                gap = next_start - curr_end
                fragments[i]["gap_to_next_sec"] = round(gap, 3)

                if audio is not None and silence_rms is not None and gap > 0.01:
                    gap_rms = AudioUtils.get_segment_rms(
                        audio, curr_end, next_start, sample_rate,
                    )
                    fragments[i]["gap_is_silent"] = gap_rms < silence_rms * 2.0
                else:
                    fragments[i]["gap_is_silent"] = True
            else:
                fragments[i]["gap_to_next_sec"] = None
                fragments[i]["gap_is_silent"] = None

        return fragments

    def build_merge_input(
        self,
        fragments: List[Dict],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """构建 LLM 合并决策的输入

        每个片段附带精确时间戳、ASR文本、与下一段的间隙信息。
        """
        from ..utils.audio_utils import AudioUtils

        silence_rms = None
        if audio is not None:
            silence_rms = AudioUtils.estimate_silence_rms(audio, sample_rate)

        result = []
        for i, frag in enumerate(fragments):
            item = {
                "id": i + 1,
                "start": round(frag.get("start", 0), 2),
                "end": round(frag.get("end", 0), 2),
                "duration": round(frag.get("end", 0) - frag.get("start", 0), 2),
                "speaker": frag.get("speaker", "unknown"),
                "text": frag.get("text", ""),
            }

            # 与下一段的间隙信息
            if i < len(fragments) - 1:
                next_frag = fragments[i + 1]
                gap = next_frag.get("start", 0) - frag.get("end", 0)
                item["gap_to_next_sec"] = round(gap, 3)

                if silence_rms is not None and gap > 0.01 and audio is not None:
                    gap_rms = AudioUtils.get_segment_rms(
                        audio, frag.get("end", 0),
                        next_frag.get("start", 0), sample_rate,
                    )
                    item["gap_energy_ratio"] = round(
                        gap_rms / max(silence_rms, 1e-8), 1,
                    )
                    item["gap_is_silent"] = gap_rms < silence_rms * 2.0
                else:
                    item["gap_is_silent"] = True
            else:
                item["gap_to_next_sec"] = None
                item["gap_is_silent"] = None

            result.append(item)

        return result


# ------------------------------------------------------------------
# 帧级无缝衔接 (3.7)
# ------------------------------------------------------------------

from .layout import (
    SUBTITLE_LAYOUT_RULES,
    apply_frame_seamless_stitching,
    apply_layout_suggestions,
    auto_layout_events,
    auto_line_break_fallback,
)

# Keep the old private names available while making the active engine use the
# isolated constraint and policy modules.
_physical_owner_compatible = merge_constraints.physical_owner_compatible
_physical_owner_compatible_for_events = merge_constraints.physical_owner_compatible_for_events
_detect_semantic_boundary = merge_policy.detect_semantic_boundary
