"""Few-shot 示例构建器

将用户的合并/拆分动作提取为 LLM Prompt 示例，
动态注入到后续 LLM 调用中（语义合并决策 + LLM 字幕优化）。

缓存管理: LRU + 置信度衰减双重淘汰机制。
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class FewShotExample:
    """一条 Few-shot 示例"""
    example_type: str  # "merge" | "split" | "format"
    fragments: List[str] = field(default_factory=list)
    decision: str = ""       # "MERGE" | "SPLIT"
    reason: str = ""
    split_after: str = ""    # 拆分点文本
    rule: str = ""            # 格式规则
    weight: float = 1.0      # 当前权重 [0, 1]
    created_at: str = ""
    last_hit_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_hit_at:
            self.last_hit_at = now

    def touch(self):
        """更新最近命中时间"""
        self.last_hit_at = datetime.now().isoformat()

    def decayed_weight(self, half_life_days: int = 180, current_time: Optional[datetime] = None) -> float:
        """计算衰减后的权重"""
        import math
        try:
            record_time = datetime.fromisoformat(self.created_at)
        except (ValueError, TypeError):
            return self.weight
        now = current_time or datetime.now()
        elapsed_days = (now - record_time).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return self.weight
        decay = math.exp(-elapsed_days / half_life_days)
        return self.weight * decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.example_type,
            "fragments": self.fragments,
            "decision": self.decision,
            "reason": self.reason,
            "split_after": self.split_after,
            "rule": self.rule,
            "weight": self.weight,
            "created_at": self.created_at,
            "last_hit_at": self.last_hit_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FewShotExample":
        return cls(
            example_type=data.get("type", ""),
            fragments=data.get("fragments", []),
            decision=data.get("decision", ""),
            reason=data.get("reason", ""),
            split_after=data.get("split_after", ""),
            rule=data.get("rule", ""),
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", ""),
            last_hit_at=data.get("last_hit_at", ""),
        )


# ---------------------------------------------------------------------------
# 缓存管理器
# ---------------------------------------------------------------------------


class FewShotCacheManager:
    """Few-shot 示例缓存管理器

    淘汰策略（优先级递降）：
    1. 置信度衰减淘汰: decayed_weight < 0.15 → 自动移除
    2. LRU 淘汰: 缓存超过 max_capacity（默认 20 条）时
    3. 重复检测: 新增示例与缓存中某条的语义相似度 > 0.9 → 更新而非新增
    """

    def __init__(self, max_capacity: int = 20, half_life_days: int = 180):
        self.max_capacity = max_capacity
        self.half_life_days = half_life_days
        self._cache: List[FewShotExample] = []

    def add(self, example: FewShotExample) -> None:
        """添加示例（含自动淘汰逻辑）"""
        # 1. 重复检测
        for existing in self._cache:
            if self._is_duplicate(existing, example):
                existing.touch()
                existing.weight = max(existing.weight, example.weight)
                logger.debug("FewShot cache: duplicate detected, updated existing example")
                return

        # 2. 衰减淘汰
        self._evict_decayed()

        # 3. LRU 淘汰
        if len(self._cache) >= self.max_capacity:
            self._cache.sort(key=lambda ex: ex.last_hit_at)
            removed = self._cache.pop(0)
            logger.debug("FewShot cache: LRU eviction removed example (last_hit=%s)", removed.last_hit_at)

        self._cache.append(example)
        logger.info("FewShot cache: added example (type=%s), total=%d", example.example_type, len(self._cache))

    def get_active_examples(
        self,
        max_count: int = 3,
        min_weight: float = 0.3,
    ) -> List[FewShotExample]:
        """获取当前活跃的示例（按权重排序，取 Top-K）

        Args:
            max_count: 最多返回 N 条
            min_weight: 最低权重阈值

        Returns:
            排序后的活跃示例列表
        """
        now = datetime.now()
        active = [
            ex for ex in self._cache
            if ex.decayed_weight(self.half_life_days, now) >= min_weight
        ]
        active.sort(
            key=lambda ex: ex.decayed_weight(self.half_life_days, now),
            reverse=True,
        )
        return active[:max_count]

    def load_from_file(self, path: Path) -> None:
        """从 JSON 文件加载缓存"""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._cache = [FewShotExample.from_dict(item) for item in data]
            logger.info("FewShot cache: loaded %d examples from %s", len(self._cache), path)
        except Exception as e:
            logger.warning("Failed to load few-shot cache: %s", e)

    def save_to_file(self, path: Path) -> None:
        """保存缓存到 JSON 文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [ex.to_dict() for ex in self._cache]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("FewShot cache: saved %d examples to %s", len(self._cache), path)

    def size(self) -> int:
        return len(self._cache)

    def _evict_decayed(self) -> None:
        """移除衰减权重过低的示例"""
        now = datetime.now()
        before = len(self._cache)
        self._cache = [
            ex for ex in self._cache
            if ex.decayed_weight(self.half_life_days, now) >= 0.15
        ]
        removed = before - len(self._cache)
        if removed > 0:
            logger.debug("FewShot cache: decay eviction removed %d examples", removed)

    @staticmethod
    def _is_duplicate(a: FewShotExample, b: FewShotExample) -> bool:
        """判断两个示例是否为重复（类型相同 + 文本高度重叠）"""
        if a.example_type != b.example_type:
            return False
        a_text = " ".join(a.fragments)
        b_text = " ".join(b.fragments)
        if not a_text or not b_text:
            return False
        # 简单 Jaccard 重叠检测
        a_words = set(a_text)
        b_words = set(b_text)
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words) / len(a_words | b_words)
        return overlap > 0.9


# ---------------------------------------------------------------------------
# Few-shot 示例构建器
# ---------------------------------------------------------------------------


class FewShotBuilder:
    """将用户修订历史构建为 LLM Few-shot 示例

    用于增强:
    - LLM 语义合并决策 (MergeDecisionConfig → LLM prompt)
    - LLM 字幕优化 (LLMOptimizeConfig → 后处理 prompt)
    """

    def __init__(
        self,
        cache_manager: Optional[FewShotCacheManager] = None,
        max_examples: int = 3,
    ):
        self._cache = cache_manager or FewShotCacheManager()
        self._max_examples = max_examples

    def build_merge_examples(
        self,
        merge_actions: List[Any],
        max_examples: Optional[int] = None,
    ) -> List[FewShotExample]:
        """将合并/拆分动作构建为 Few-shot 示例

        Args:
            merge_actions: MergeAction 列表
            max_examples: 最多生成 N 条新示例

        Returns:
            本次新增的示例列表
        """
        max_n = max_examples or self._max_examples
        new_examples = []

        for action in merge_actions[:max_n]:
            action_type = getattr(action, "action_type", "")
            if action_type == "merge":
                example = FewShotExample(
                    example_type="merge",
                    fragments=[getattr(action, "text_auto", "")],
                    decision="MERGE",
                    reason=f"用户合并了 {getattr(action, 'auto_count', 0)} 个片段（间隙 {getattr(action, 'gap_between', 0):.2f}s）",
                )
            elif action_type == "split":
                text_manual = getattr(action, "text_manual", "")
                example = FewShotExample(
                    example_type="split",
                    fragments=[text_manual],
                    decision="SPLIT",
                    reason=f"用户拆分了 {getattr(action, 'manual_count', 0)} 个片段",
                )
            else:
                continue

            new_examples.append(example)
            self._cache.add(example)

        return new_examples

    def build_format_examples(
        self,
        text_edits: List[Any],
        max_examples: Optional[int] = None,
    ) -> List[FewShotExample]:
        """将文本格式修改构建为 Few-shot 示例

        Args:
            text_edits: TextEdit 列表
            max_examples: 最多生成 N 条
        """
        max_n = max_examples or self._max_examples
        new_examples = []

        for edit in text_edits[:max_n]:
            edit_type = getattr(edit, "edit_type", "")
            if edit_type != "punctuation":
                continue
            rule = f"句末标点: '{getattr(edit, 'auto_text', '')}' → '{getattr(edit, 'manual_text', '')}'"
            example = FewShotExample(
                example_type="format",
                rule=rule,
                reason="用户偏好特定标点格式",
            )
            new_examples.append(example)
            self._cache.add(example)

        return new_examples

    def inject_into_prompt(
        self,
        base_prompt: str,
        max_examples: Optional[int] = None,
        min_weight: float = 0.3,
    ) -> str:
        """将 Few-shot 示例注入基础 Prompt

        格式:
        --- User Preference Examples ---
        [示例1]: ...
        [示例2]: ...
        --- End User Preferences ---

        Args:
            base_prompt: 基础 LLM Prompt
            max_examples: 最多注入 N 条
            min_weight: 最低权重

        Returns:
            注入后的 Prompt
        """
        examples = self._cache.get_active_examples(
            max_count=max_examples or self._max_examples,
            min_weight=min_weight,
        )

        if not examples:
            return base_prompt

        lines = ["\n--- User Preference Examples ---"]
        for i, ex in enumerate(examples, 1):
            if ex.example_type == "merge":
                fragments_text = " | ".join(ex.fragments)
                lines.append(f"[示例{i}] 合并: \"{fragments_text}\" → {ex.decision}")
                if ex.reason:
                    lines.append(f"  理由: {ex.reason}")
            elif ex.example_type == "split":
                fragments_text = " | ".join(ex.fragments)
                lines.append(f"[示例{i}] 拆分: \"{fragments_text}\"")
                if ex.split_after:
                    lines.append(f"  拆分点: \"{ex.split_after}\"")
                if ex.reason:
                    lines.append(f"  理由: {ex.reason}")
            elif ex.example_type == "format":
                lines.append(f"[示例{i}] 格式偏好: {ex.rule}")

        lines.append("--- End User Preferences ---\n")

        injected = "\n".join(lines)
        return base_prompt + "\n" + injected

    def get_cache_path(self, profile_name: str) -> Path:
        """获取缓存文件路径"""
        from pathlib import Path
        return Path.home() / ".vocal_subtitle" / "few_shot_cache" / f"{profile_name}.json"

    def load_cache(self, profile_name: str = "user_default") -> None:
        """加载缓存"""
        path = self.get_cache_path(profile_name)
        self._cache.load_from_file(path)

    def save_cache(self, profile_name: str = "user_default") -> None:
        """保存缓存"""
        path = self.get_cache_path(profile_name)
        self._cache.save_to_file(path)
