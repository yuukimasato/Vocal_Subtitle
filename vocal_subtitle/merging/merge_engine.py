"""LLMMergeEngine — thin entry point backed by domain modules.

Original class from llm_merge_engine.py, now delegates to:
- merge_constraints.py (semantic boundary, physical owner, speaker checks)
- layout.py (frame seamless, layout, line break)
- merge_policy.py (Fast-Slow Path, local NLP, LLM decisions)

The class definition and merge() orchestrator remain here for backward compat.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .merge_constraints import (
    _detect_semantic_boundary,
    _physical_owner_compatible,
)

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------

@dataclass
class MergeDecisionConfig:
    """合并决策分流配置"""

    # Fast-Slow Path 分流阈值
    fast_merge_max_gap: float = 0.20      # <200ms: 规则强制合并
    llm_decision_min_gap: float = 0.20    # 200-1200ms: LLM裁决
    llm_decision_max_gap: float = 1.20
    hard_split_min_gap: float = 1.20      # >1200ms: 强制不合并

    # 合并约束
    max_combined_duration: float = 5.0    # 合并后字幕不超过5秒
    min_fragment_duration: float = 0.15   # 最小片段时长

    # LLM 降本策略
    llm_tier: str = "cascading"           # "cascading" | "all_llm" | "rule_only"
    local_nlp_gap_range: Tuple[float, float] = (0.15, 0.60)  # 本地NLP优先的间隙范围

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
        self._local_model = None  # 延迟加载 sentence-transformers
        self._local_model_attempted = False  # 防止重复尝试加载

    def merge(
        self,
        fragments: List[Dict],
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """合并决策流水线"""
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

        # Step 4: 收集候选并按间隙范围分流
        local_nlp_candidates = []   # local_nlp_gap_range → 本地 NLP 优先
        cloud_llm_candidates = []   # cloud_llm_gap_range → 云端 LLM
        rule_decisions = {}

        for i, frag in enumerate(fast_merged):
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None:
                gap = 999

            merged_ids = frag.get("_merged_ids", [frag.get("id", i + 1)])
            last_id = merged_ids[-1] if merged_ids else frag.get("id", i + 1)

            if gap <= cfg.fast_merge_max_gap:
                continue
            elif gap > cfg.hard_split_min_gap:
                rule_decisions[last_id] = False
            elif cfg.llm_decision_min_gap <= gap <= cfg.llm_decision_max_gap:
                local_min, local_max = cfg.local_nlp_gap_range
                if local_min <= gap <= local_max and cfg.llm_tier == "cascading":
                    local_nlp_candidates.append(frag)
                else:
                    cloud_llm_candidates.append(frag)

        # Step 5a: 本地 NLP 裁决
        local_nlp_groups = []
        unresolved_from_local = []
        if local_nlp_candidates:
            local_nlp_groups, unresolved_from_local = self._local_merge_decision(
                local_nlp_candidates,
            )
            cloud_llm_candidates.extend(unresolved_from_local)

        # Step 5b: 云端 LLM 裁决
        llm_groups = list(local_nlp_groups)
        if cloud_llm_candidates and cfg.llm_tier != "rule_only":
            try:
                cloud_groups = self._call_llm_merge_decision(cloud_llm_candidates)
                llm_groups.extend(cloud_groups)
            except Exception as e:
                logger.warning("LLM merge decision failed, falling back to rules: %s", e)
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
                frag["text"] = " ".join(merged_texts)
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

    # ------------------------------------------------------------------
    # 本地 NLP 模型
    # ------------------------------------------------------------------

    def _load_local_model(self):
        """加载轻量本地语义模型（单次加载，全局复用）"""
        if self._local_model is not None:
            return self._local_model

        if self._local_model_attempted:
            return None

        self._local_model_attempted = True

        from ..utils.model_loader import load_sentence_transformer

        self._local_model = load_sentence_transformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
        if self._local_model is not None:
            logger.info("Local NLP model loaded for merge decisions")
        return self._local_model

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的语义相似度

        Returns:
            0.0 ~ 1.0，模型不可用时返回 0.5（中性值）
        """
        model = self._load_local_model()
        if model is None:
            return 0.5

        if not text_a or not text_b:
            return 0.5

        try:
            embeddings = model.encode(
                [text_a, text_b], convert_to_numpy=True,
            )
            dot = float(np.dot(embeddings[0], embeddings[1]))
            norm_a = float(np.linalg.norm(embeddings[0]))
            norm_b = float(np.linalg.norm(embeddings[1]))
            return dot / max(norm_a * norm_b, 1e-8)
        except Exception as e:
            logger.debug("Similarity computation failed: %s", e)
            return 0.5

    # ------------------------------------------------------------------
    # 本地合并决策
    # ------------------------------------------------------------------

    def _local_merge_decision(
        self,
        candidates: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """本地合并决策：规则 + 轻量语义模型

        第一层：纯规则（零成本，<1ms）
        - 逗号/分号结尾 → 一定合并
        - 下一段小写开头 + 短间隙 → 倾向合并
        - 句尾标点 + 长间隙 → 不合并

        第二层：轻量语义模型（低成本，~30ms）
        - similarity > 0.8 → 合并
        - similarity < 0.3 → 不合并
        - 中间值 → 升级到云端 LLM
        """
        comma_endings = {",", "，", "、", ";", "；"}
        sentence_endings = {".", "!", "?", "。", "！", "？"}

        decided_groups = []
        unresolved = []

        for frag in candidates:
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None or gap > 10:
                continue

            # 重叠保护：负间隙（重叠事件）绝不合并
            if gap < -0.02:
                continue

            text = frag.get("text", "").rstrip()
            frag_id = frag.get("id", 0)
            next_id = frag_id + 1

            next_text = ""
            next_speaker = ""
            next_frag = None
            for other in candidates:
                if other.get("id") == next_id:
                    next_frag = other
                    next_text = other.get("text", "").strip()
                    next_speaker = other.get("speaker", "")
                    break

            if next_frag is None:
                continue

            # 不同说话人 → 不合并
            curr_speaker = frag.get("speaker", "")
            if curr_speaker and next_speaker and curr_speaker != next_speaker:
                continue

            # 物理所有权不兼容 → 不合并
            if not _physical_owner_compatible(frag, next_frag):
                continue

            # ★ 语义边界检测：下一段是新段落开头 → 不合并
            if next_text and _detect_semantic_boundary(text, next_text):
                continue

            # 第一层：规则
            if text and text[-1] in comma_endings:
                decided_groups.append({
                    "ids": [frag_id, next_id],
                    "reason": "[local] comma ending → merge",
                })
                continue

            if text and text[-1] in sentence_endings and gap > 0.4:
                continue

            if next_text and next_text[0].islower() and gap < 0.5:
                decided_groups.append({
                    "ids": [frag_id, next_id],
                    "reason": "[local] lowercase following + short gap → merge",
                })
                continue

            # 第二层：语义相似度
            similarity = self._compute_similarity(text, next_text)
            frag["_sim_to_next"] = similarity

            if similarity > 0.8:
                decided_groups.append({
                    "ids": [frag_id, next_id],
                    "reason": f"[local] semantic similarity {similarity:.2f} > 0.8 → merge",
                })
            elif similarity < 0.3:
                continue
            else:
                unresolved.append(frag)

        return decided_groups, unresolved

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _call_llm_merge_decision(
        self, candidates: List[Dict],
    ) -> List[Dict]:
        """调用 LLM 进行合并决策"""
        import requests

        cfg = self.config

        llm_input = []
        for frag in candidates:
            gap_is_silent = frag.get("gap_is_silent")
            llm_input.append({
                "id": frag.get("id", 0),
                "start": round(float(frag.get("start", 0)), 2),
                "end": round(float(frag.get("end", 0)), 2),
                "speaker": str(frag.get("speaker", "unknown")),
                "text": str(frag.get("text", "")),
                "gap_to_next_sec": (
                    round(float(frag.get("gap_to_next_sec", 0)), 3)
                    if frag.get("gap_to_next_sec") is not None else None
                ),
                "gap_is_silent": (
                    bool(gap_is_silent) if gap_is_silent is not None else None
                ),
            })

        messages = [
            {"role": "system", "content": MERGE_DECISION_PROMPT},
            {"role": "user", "content": json.dumps(llm_input, ensure_ascii=False, indent=2)},
        ]

        api_url = f"{cfg.llm_base_url}/v1/chat/completions" if cfg.llm_base_url else None
        if not api_url:
            logger.warning("No LLM API URL configured, using fallback rules")
            return self._fallback_rule_decisions(candidates)

        headers = {"Content-Type": "application/json"}
        if cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {cfg.llm_api_key}"

        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": cfg.llm_temperature,
            "max_tokens": 2000,
        }

        try:
            response = requests.post(
                api_url, json=payload, headers=headers,
                timeout=cfg.llm_timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]

            json_match = None
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                json_str = content[start:end].strip()
            elif "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            result = json.loads(json_str)
            groups = result.get("merge_groups", [])

            logger.info(
                "LLM merge: %d candidates → %d groups",
                len(candidates), len(groups),
            )
            return groups

        except Exception as e:
            logger.error("LLM merge API call failed: %s", e)
            raise

    def _fallback_rule_decisions(
        self, candidates: List[Dict],
    ) -> List[Dict]:
        """LLM 不可用时的降级规则决策（含语义边界检测）"""
        cfg = self.config
        groups = []
        sentence_endings = {".", "!", "?", "。", "！", "？"}
        comma_endings = {",", "，", ";", "；"}

        for frag in candidates:
            gap = frag.get("gap_to_next_sec", 999)
            if gap is None:
                continue

            if gap < -0.02:
                continue

            text = frag.get("text", "").rstrip()
            next_id = frag.get("id", 0) + 1

            next_text = ""
            next_speaker = ""
            for other in candidates:
                if other.get("id") == next_id:
                    next_text = other.get("text", "").strip()
                    next_speaker = other.get("speaker", "")
                    break

            curr_speaker = frag.get("speaker", "")
            if curr_speaker and next_speaker and curr_speaker != next_speaker:
                continue

            if next_text and _detect_semantic_boundary(text, next_text):
                continue

            if text and text[-1] in comma_endings and gap < 0.6:
                groups.append({
                    "ids": [frag["id"], next_id],
                    "reason": "[fallback] comma + moderate gap → merge",
                })
            elif text and text[-1] in sentence_endings and gap > 0.4:
                pass
            elif gap < 0.4:
                groups.append({
                    "ids": [frag["id"], next_id],
                    "reason": "[fallback] short gap → merge",
                })

        return groups

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

        id_map = {}
        for frag in fast_merged:
            merged_ids = frag.get("_merged_ids", [frag.get("id", 0)])
            for mid in merged_ids:
                id_map[mid] = frag

        merged = []
        consumed = set()

        for group in llm_groups:
            ids = group.get("ids", [])
            if not ids:
                continue

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

            speakers = set()
            for mid in ids:
                f = id_map.get(mid)
                if f:
                    spk = f.get("speaker", "")
                    if spk:
                        speakers.add(spk)
            if len(speakers) > 1:
                continue
            if any(f.get("speaker", "") in ("", "unknown") for f in selected):
                continue

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

            combined_text = " ".join(texts)

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

        for frag in fast_merged:
            frag_ids = frag.get("_merged_ids", [frag.get("id", 0)])
            if all(mid in consumed for mid in frag_ids):
                continue
            if frag.get("id", 0) not in consumed:
                merged.append(frag)

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
        """构建 LLM 合并决策的输入"""
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
