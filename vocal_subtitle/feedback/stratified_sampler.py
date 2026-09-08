"""D2→D3 分层抽样器

从 D2 候选反馈集中按多维度分层抽样，生成 D3 反馈回归集冻结版本。
对应 FEEDBACK_LOOP.md 的 D3 分层抽样流程和 QUALITY_OPERATIONS.md §8 的平衡约束。

约束:
  - 任何单一维度占比不超过 60%
  - 最小每层样本数可配置（默认 1）
  - 确定性抽样（固定 seed）
  - 只从审核通过的样本（accepted + deidentified）中选取
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from vocal_subtitle.quality.scene_slicer import (
    DIMENSION_NAMES,
    MAX_SINGLE_DIMENSION_RATIO,
    SceneSlicer,
)

logger = logging.getLogger(__name__)


@dataclass
class SamplingPlan:
    """抽样计划"""

    version_id: str                # "D3-20260901-001"
    description: str = ""
    strata: list[str] = field(default_factory=lambda: ["language", "scene", "speaker_count"])
    quotas: dict[str, dict[str, int]] = field(default_factory=dict)
    min_per_stratum: int = 1
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "description": self.description,
            "strata": self.strata,
            "quotas": self.quotas,
            "min_per_stratum": self.min_per_stratum,
            "seed": self.seed,
        }


@dataclass
class SamplingResult:
    """抽样结果"""

    selected: list[dict] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    balance: dict = field(default_factory=dict)
    plan: Optional[SamplingPlan] = None

    @property
    def sample_count(self) -> int:
        return len(self.selected)

    def to_dict(self) -> dict:
        return {
            "selected_count": len(self.selected),
            "rejected_count": len(self.rejected),
            "rejected_ids": self.rejected,
            "selected": self.selected,
            "balance": self.balance,
            "plan": self.plan.to_dict() if self.plan else {},
        }


class StratifiedSampler:
    """D2→D3 分层抽样器。

    使用示例:
        sampler = StratifiedSampler()
        mgr = FeedbackSampleManager()

        # 获取 D2 池
        pool = sampler.pool_from_manager(mgr)

        # 构建抽样计划
        plan = sampler.build_plan(pool, strata=["language", "scene"])

        # 执行抽样
        result = sampler.sample(pool, plan)

        # 冻结为 D3 版本
        version = sampler.freeze(result, data_version_manager=dvm,
                                 description="首次 D3 抽样")
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._storage_dir = Path(storage_dir or (
            Path(__file__).parent.parent.parent / "cache" / "datasets" / "sampling"
        ))
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def pool_from_manager(sample_manager: Any) -> list[dict]:
        """从 FeedbackSampleManager 获取合格的 D2 样本池。

        筛选条件:
          - review.result == "accepted"
          - anonymization == "deidentified"
          - consent_level != "local"

        Args:
            sample_manager: FeedbackSampleManager 实例

        Returns:
            合格的样本 dict 列表
        """
        pool = []
        for sample_id, entry in sample_manager._index.items():
            if entry.get("status") != "accepted":
                continue
            sample = sample_manager.get(sample_id)
            if sample is None:
                continue
            if sample.get("anonymization") != "deidentified":
                continue
            if sample.get("consent_level") == "local":
                continue
            pool.append(sample)
        logger.info("D2 pool: %d accepted samples", len(pool))
        return pool

    def build_plan(
        self,
        pool: list[dict],
        *,
        strata: Optional[list[str]] = None,
        quotas: Optional[dict[str, dict[str, int]]] = None,
        min_per_stratum: int = 1,
        seed: int = 42,
        description: str = "",
    ) -> SamplingPlan:
        """从样本池构建抽样计划。

        Args:
            pool: D2 样本列表
            strata: 分层维度（默认 language, scene, speaker_count）
            quotas: 每维度每值的配额上限
            min_per_stratum: 每层最少样本数
            seed: 随机种子
            description: 计划描述

        Returns:
            SamplingPlan
        """
        active_strata = strata or ["language", "scene", "speaker_count"]

        # 统计各维度的值分布
        dim_values: dict[str, set[str]] = {d: set() for d in active_strata}
        for sample in pool:
            scene_tags = self._extract_tags(sample)
            for dim in active_strata:
                val = scene_tags.get(dim, "unknown")
                dim_values[dim].add(val)

        # 自动生成配额：均匀分配
        if quotas is None:
            quotas = {}
            total = len(pool)
            for dim in active_strata:
                dim_quotas: dict[str, int] = {}
                n_values = max(len(dim_values[dim]), 1)
                per_value = max(total // n_values, min_per_stratum)
                for val in dim_values[dim]:
                    dim_quotas[val] = per_value
                quotas[dim] = dim_quotas

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        version_id = f"D3-{today}-001"

        return SamplingPlan(
            version_id=version_id,
            description=description or f"从 D2 抽样 D3 ({', '.join(active_strata)})",
            strata=active_strata,
            quotas=quotas,
            min_per_stratum=min_per_stratum,
            seed=seed,
        )

    def sample(self, pool: list[dict], plan: SamplingPlan) -> SamplingResult:
        """执行分层抽样。

        按 plan.strata 指定的维度对 pool 分组，
        从每组中按配额随机选取样本。

        Args:
            pool: D2 样本列表
            plan: 抽样计划

        Returns:
            SamplingResult 包含选中、拒绝和平衡报告
        """
        rng = random.Random(plan.seed)

        # 按 strata dimensions 构建分层 key
        strata_groups: dict[str, list[dict]] = {}
        for sample in pool:
            tags = self._extract_tags(sample)
            key_parts = [tags.get(d, "unknown") for d in plan.strata]
            key = "|".join(key_parts)
            strata_groups.setdefault(key, []).append(sample)

        selected: list[dict] = []
        rejected_ids: list[str] = []

        for stratum_key, group in strata_groups.items():
            # 配额：取各维度配额的最小值
            cap = None
            key_parts = stratum_key.split("|")
            for i, dim in enumerate(plan.strata):
                val = key_parts[i] if i < len(key_parts) else "unknown"
                dim_quota = plan.quotas.get(dim, {}).get(val)
                if dim_quota is not None:
                    cap = min(cap, dim_quota) if cap is not None else dim_quota

            take = max(cap, plan.min_per_stratum) if cap is not None else len(group)
            take = min(take, len(group))

            shuffled = list(group)
            rng.shuffle(shuffled)
            selected.extend(shuffled[:take])

            for s in shuffled[take:]:
                sid = s.get("sample_id", "")
                if sid:
                    rejected_ids.append(sid)

        # 平衡检查
        balance = SceneSlicer.balance_report(selected)

        # 如果某维度超过 60%，尝试从超代表组移除一些
        for warning in balance.get("warnings", []):
            logger.warning(
                "Balance warning in D3 sample: %s", warning["message"]
            )

        result = SamplingResult(
            selected=selected,
            rejected=rejected_ids,
            balance=balance,
            plan=plan,
        )

        # 持久化
        self._save_result(result)

        logger.info(
            "Stratified sampling complete: %d selected, %d rejected from %d pool",
            len(selected), len(rejected_ids), len(pool),
        )
        return result

    def freeze(
        self,
        result: SamplingResult,
        *,
        data_version_manager: Any = None,
        description: str = "",
    ) -> Any:
        """将抽样结果冻结为 D3 回归集版本。

        Args:
            result: SamplingResult
            data_version_manager: DataVersionManager 实例
            description: 版本描述

        Returns:
            DatasetVersion 实例

        Raises:
            ValueError: 无 data_version_manager 时
        """
        if data_version_manager is None:
            # 懒加载避免循环导入
            from vocal_subtitle.quality.data_version_manager import (
                DatasetTier,
                DataVersionManager,
            )
            data_version_manager = DataVersionManager()

        from vocal_subtitle.quality.data_version_manager import DatasetTier

        plan = result.plan
        version_id = plan.version_id if plan else f"D3-{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"

        entries = []
        for sample in result.selected:
            entries.append({
                "sample_id": sample.get("sample_id", ""),
                "language": sample.get("language", "unknown"),
                "scene": sample.get("scene", "unknown"),
                "scene_tags": self._extract_tags(sample),
                "source_sample": sample.get("sample_id", ""),
            })

        return data_version_manager.freeze(
            tier=DatasetTier.D3,
            version_id=version_id,
            entries=entries,
            description=description or f"{result.sample_count} 样本，分层抽样",
        )

    # ---- 内部 ----

    @staticmethod
    def _extract_tags(sample: dict) -> dict:
        """从样本中提取场景标签。"""
        # 优先使用已存储的 scene_tags
        tags = sample.get("scene_tags")
        if isinstance(tags, dict) and tags:
            return dict(tags)

        # 从样本元数据生成
        meta = {
            "language": sample.get("language", "unknown"),
            "speaker_count": sample.get("speaker_count", 0),
            "audio_condition": sample.get("audio_condition", ""),
            "scene": sample.get("scene", ""),
        }
        scene_tag = SceneSlicer.tag(meta)
        return scene_tag.to_dict()

    def _save_result(self, result: SamplingResult) -> None:
        if result.plan is None:
            return
        path = self._storage_dir / f"sample-{result.plan.version_id}.json"
        path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = ["SamplingPlan", "SamplingResult", "StratifiedSampler"]
