"""LLM 语义合并引擎 (方案五)

将声学合并与语义合并完全分离：声学层激进切分，语义层用 LLM 决策合并。

核心架构: 三级决策流水线 (Fast-Slow Path)

  相邻片段间隔 < 200ms  ──→ 快路径（规则强制合并）
  相邻片段间隔 200-1200ms ──→ 慢路径（LLM 裁决）
  相邻片段间隔 > 1200ms   ──→ 硬规则（强制不合并）

降级策略: LLM 不可用时自动回退到 Fast-Path 纯规则模式。

帧级无缝衔接 (3.7): 非句尾字幕自动衔接到下一句，消除字幕闪烁。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

    def _load_local_model(self):
        """加载轻量本地语义模型（单次加载，全局复用）

        委托给统一的 model_loader 工具，确保离线优先策略：
        1. 本地缓存 → 即时返回（零网络）
        2. 本地无缓存 → 限时下载 + 镜像站回退
        3. 全部失败 → 返回 None（优雅降级到规则模式）
        """
        if self._local_model is not None:
            return self._local_model

        # 已经尝试过且失败 → 静默返回 None，避免重复警告
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

    def _local_merge_decision(
        self,
        candidates: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """本地合并决策：规则 + 轻量语义模型（文档 5.4.1）

        第一层：纯规则（零成本，<1ms）
        - 逗号/分号结尾 → 一定合并
        - 下一段小写开头 + 短间隙 → 倾向合并
        - 句尾标点 + 长间隙 → 不合并

        第二层：轻量语义模型（低成本，~30ms）
        - similarity > 0.8 → 合并
        - similarity < 0.3 → 不合并
        - 中间值 → 升级到云端 LLM

        Returns:
            (decided_groups, unresolved_candidates):
            - decided_groups: 本地已裁决的合并组
            - unresolved_candidates: 无法本地判断的，升级到云端 LLM
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

            # 查找下一段
            next_text = ""
            next_speaker = ""
            for other in candidates:
                if other.get("id") == next_id:
                    next_text = other.get("text", "").strip()
                    next_speaker = other.get("speaker", "")
                    break

            # ★ 不同说话人 → 绝不合并
            curr_speaker = frag.get("speaker", "")
            if curr_speaker and next_speaker and curr_speaker != next_speaker:
                continue

            # 语义边界检测：下一段是新段落开头 → 不合并
            if next_text and _detect_semantic_boundary(text, next_text):
                continue

            # ---- 第一层：纯规则 ----
            # 规则1：标点未完 → 一定合并
            if text and text[-1] in comma_endings:
                decided_groups.append({
                    "ids": [frag_id, next_id],
                    "reason": "[local] comma/clause continuation → merge",
                })
                continue

            # 规则2：句尾标点 + 长间隙（> 0.6s）→ 不合并
            if text and text[-1] in sentence_endings and gap > 0.6:
                # 不合并，无需生成 group（单独保留 = 不合并）
                continue

            # 规则3：句尾标点 + 中短间隙 → 倾向不合并
            if text and text[-1] in sentence_endings and gap > 0.35:
                continue

            # ---- 第二层：轻量语义模型 ----
            local_min, local_max = self.config.local_nlp_gap_range
            if local_min <= gap <= local_max:
                if next_text:
                    similarity = self._compute_similarity(text, next_text)
                else:
                    similarity = 0.5  # 无下一段信息 → 中性

                if similarity > 0.8:
                    decided_groups.append({
                        "ids": [frag_id, next_id],
                        "reason": f"[local] high similarity ({similarity:.2f}) → merge",
                    })
                    continue
                elif similarity < 0.3:
                    # 语义不相似 → 不合并
                    continue
                # else: 0.3~0.8 → 不确定 → 升级到云端 LLM

            # 无法本地判断 → 升级
            unresolved.append(frag)

        if decided_groups:
            logger.info(
                "Local NLP: %d candidates → %d merged, %d → cloud LLM",
                len(candidates), len(decided_groups), len(unresolved),
            )
        return decided_groups, unresolved

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _call_llm_merge_decision(
        self, candidates: List[Dict],
    ) -> List[Dict]:
        """调用 LLM 进行合并决策

        Returns:
            [{"ids": [1, 2], "reason": "..."}, ...]
        """
        import requests

        cfg = self.config

        # 构建输入
        # 使用 _safe_json_value 确保所有值都是 JSON 可序列化的
        # （numpy bool/float 类型会导致 "Object of type bool is not JSON serializable"）
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

        # 构建请求
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

            # 解析 JSON 响应
            # 尝试提取 JSON 块
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

            # 重叠保护：负间隙（重叠事件）绝不合并
            if gap < -0.02:
                continue

            text = frag.get("text", "").rstrip()
            next_id = frag.get("id", 0) + 1

            # 查找下一段
            next_text = ""
            next_speaker = ""
            for other in candidates:
                if other.get("id") == next_id:
                    next_text = other.get("text", "").strip()
                    next_speaker = other.get("speaker", "")
                    break

            # ★ 不同说话人 → 绝不合并
            curr_speaker = frag.get("speaker", "")
            if curr_speaker and next_speaker and curr_speaker != next_speaker:
                continue

            # 语义边界检测：下一段是新段落开头 → 不合并
            if next_text and _detect_semantic_boundary(text, next_text):
                continue

            # 标点规则
            if text and text[-1] in comma_endings and gap < 0.6:
                # 逗号结尾 + 中短间隙 → 合并
                groups.append({
                    "ids": [frag["id"], next_id],
                    "reason": "[fallback] comma + moderate gap → merge",
                })
            elif text and text[-1] in sentence_endings and gap > 0.4:
                # 句尾标点 + 较长间隙 → 不合并（已在 hard-split 中处理）
                pass
            elif gap < 0.4:
                # 短间隙 → 倾向合并
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

            # 拼接文本（去重：同一片段可能被多个 id 引用）
            texts = []
            seen_texts = set()
            for mid in ids:
                frag = id_map.get(mid)
                if frag:
                    t = frag.get("text", "").strip()
                    if t and t not in seen_texts:
                        seen_texts.add(t)
                        texts.append(t)
                    consumed.add(mid)

            combined_text = " ".join(texts)

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

def apply_frame_seamless_stitching(
    events: List,
    max_stitch_gap: float = 0.12,
) -> List:
    """帧级无缝衔接

    对非句尾字幕（不以 .!?。！？ 结尾），
    将其结束时间延伸至下一句开始时间，消除字幕闪烁。

    ★ 说话人安全检查：仅在同一说话人（或说话人信息不可用）时衔接。
    不同说话人的字幕即使间隙很小也不衔接，避免 A 的字幕覆盖 B 的语音。

    Args:
        events: SubtitleEvent 列表 (含 start, end, text, speaker_id 属性)
        max_stitch_gap: 最多衔接的间隙（秒）

    Returns:
        原地修改后的 events
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

            # 当前字幕的文本
            text = getattr(curr, "text", "")
            text_ends_with_terminal = (
                text.rstrip()[-1] in sentence_endings
                if text.rstrip() else False
            )

            if not text_ends_with_terminal:
                curr.end = nxt.start  # 无缝衔接

    return events


# ------------------------------------------------------------------
# 5.12.4 字幕断句排版联动
# ------------------------------------------------------------------

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


def apply_layout_suggestions(
    events: List,
    layout_suggestions: List[Dict],
) -> List:
    """将 LLM 的断行建议应用到字幕事件

    对于 ASS 格式：使用 \\N 作为换行标记
    对于 SRT 格式：使用 \\n

    Args:
        events: SubtitleEvent 列表
        layout_suggestions: LLM 输出的断行建议

    Returns:
        原地修改后的 events
    """
    for suggestion in layout_suggestions:
        group_id = suggestion.get("group_id", -1)
        if 0 <= group_id < len(events):
            line1 = suggestion.get("line1", "")
            line2 = suggestion.get("line2", "")
            if line1 and line2:
                events[group_id].text = f"{line1}\\N{line2}"
                # 标记已应用排版
                if hasattr(events[group_id], "_layout_applied"):
                    events[group_id]._layout_applied = True
                else:
                    setattr(events[group_id], "_layout_applied", True)

    return events


def auto_line_break_fallback(
    text: str,
    max_chars_per_line: int = 20,
) -> str:
    """纯规则的自动断行（LLM 不可用时的降级方案，文档 5.12.4）

    在以下位置寻找断行点（按优先级）：
    1. 逗号/分号/破折号后
    2. 介词/连词前（and, but, or, 在, 给, 为）
    3. 中点字符数找空格（最后手段）

    Args:
        text: 待断行的字幕文本
        max_chars_per_line: 单行最大字符数（CJK 20, Latin 40）

    Returns:
        含 \\N 换行标记的文本
    """
    if len(text) <= max_chars_per_line:
        return text

    # 渐进式搜索窗口：从 max_chars_per_line 逐步扩展到 2.5×
    # 偏好距离 max_chars_per_line 最近的自然断点
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
    """对字幕事件列表自动应用断行规则

    检查每条字幕，对超长的自动应用 auto_line_break_fallback。

    Args:
        events: SubtitleEvent 列表
        max_chars_cjk: CJK 字符单行上限
        max_chars_latin: 拉丁字符单行上限

    Returns:
        原地修改后的 events
    """
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
