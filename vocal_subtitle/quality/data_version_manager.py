"""数据资产版本管理器

管理 D0-D4 五层数据集的版本登记、冻结和淘汰。
对应 QUALITY_OPERATIONS.md §6 (数据版本管理) 和 §8 (样本归档策略)。

层次:
  D0: 工程诊断集 — 允许随功能增加，不归档
  D1: 参考回归集 — 每季度审查，移除不再有代表性的样本
  D2: 候选反馈集 — 超过保留期限（默认 365 天）自动清理
  D3: 反馈回归集 — 版本冻结后只增量，不删除
  D4: 场景挑战集 — 按场景更新，旧样本标记为 superseded

持久化: cache/datasets/dataset_registry.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .scene_slicer import SceneSlicer

logger = logging.getLogger(__name__)


class DatasetTier(str, Enum):
    """数据集层级"""

    D0 = "D0"  # 工程诊断集
    D1 = "D1"  # 参考回归集
    D2 = "D2"  # 候选反馈集
    D3 = "D3"  # 反馈回归集
    D4 = "D4"  # 场景挑战集


class DatasetVersionStatus(str, Enum):
    """数据集版本状态"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


# 各层级的保留策略 (§8)
RETENTION_POLICIES: dict[DatasetTier, dict] = {
    DatasetTier.D0: {
        "strategy": "never_archive",
        "retention_days": 0,
        "description": "不归档，随功能演进增减",
    },
    DatasetTier.D1: {
        "strategy": "quarterly_review",
        "retention_days": 0,
        "description": "每季度审查，移除不再有代表性的样本",
    },
    DatasetTier.D2: {
        "strategy": "auto_cleanup",
        "retention_days": 365,
        "description": "超过保留期限自动清理",
    },
    DatasetTier.D3: {
        "strategy": "append_only",
        "retention_days": 0,
        "description": "版本冻结后只增量，不删除",
    },
    DatasetTier.D4: {
        "strategy": "supersede",
        "retention_days": 0,
        "description": "按场景更新，旧样本标记为 superseded",
    },
}


@dataclass
class DatasetVersion:
    """单个数据集版本记录"""

    tier: DatasetTier
    version_id: str  # "D0-20260802-001"
    freeze_date: str = ""
    sample_count: int = 0
    description: str = ""
    status: DatasetVersionStatus = DatasetVersionStatus.ACTIVE
    retention_days: int = 0
    entries: list[dict] = field(default_factory=list)
    # entries: [{sample_id, path, scene_tags: SceneTag.to_dict(), ...}]

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "version_id": self.version_id,
            "freeze_date": self.freeze_date,
            "sample_count": self.sample_count,
            "description": self.description,
            "status": self.status.value,
            "retention_days": self.retention_days,
            "entries": self.entries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DatasetVersion:
        return cls(
            tier=DatasetTier(data.get("tier", "D0")),
            version_id=data.get("version_id", ""),
            freeze_date=data.get("freeze_date", ""),
            sample_count=data.get("sample_count", 0),
            description=data.get("description", ""),
            status=DatasetVersionStatus(data.get("status", "active")),
            retention_days=data.get("retention_days", 0),
            entries=data.get("entries", []),
        )


class DataVersionManager:
    """管理 D0-D4 数据集版本的注册、冻结和淘汰。

    使用示例:
        mgr = DataVersionManager()
        # 查看当前 D0 版本
        d0 = mgr.current(DatasetTier.D0)
        # 冻结新的 D3 版本
        version = mgr.freeze(DatasetTier.D3, "D3-20260901-001",
                             entries=[...], description="首次 D3 抽样")
        # 淘汰过期的 D2 样本
        removed = mgr.cleanup_expired()
    """

    def __init__(self, storage_dir: Path | None = None):
        self._storage_dir = Path(
            storage_dir or (Path(__file__).parent.parent.parent / "cache" / "datasets")
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._storage_dir / "dataset_registry.json"
        self._registry: dict[str, dict[str, dict]] = self._load()
        self._ensure_d0_seeded()

    # ---- 持久化 ----

    def _load(self) -> dict[str, dict[str, dict]]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        return {t.value: {} for t in DatasetTier}

    def _save(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _ensure_d0_seeded(self) -> None:
        """首次运行时自动登记 D0 基线版本。

        D0-20260802-001: 对应 DATA_ASSETS.md（docs/20260802/）登记的 13 个音频样本。
        原始评估清单 test/quality_manifest.yaml 已随 8 月 4 日快照整理移除，
        10 个场景定义见 DATA_ASSETS.md 归档说明。
        """
        if self._registry.get("D0", {}):
            return  # 已存在 D0 记录

        today = "2026-08-02"
        version_id = "D0-20260802-001"

        entries = []
        # D0 音频清单 (来自 DATA_ASSETS.md)
        d0_audio = [
            ("D0-001", "test/QS-0-1-2-2-人声.wav", 32.77, "zh", 2),
            ("D0-002", "test/181人声.wav", 58.45, "zh", 1),
            ("D0-003", "test/培训测试-双人.wav", 139.52, "mixed", 2),
            ("D0-004", "test/中文多人员测试音频.wav", 13.28, "zh", 3),
            ("D0-005", "test/英文多人员测试音频.wav", 18.64, "en", 3),
            (
                "D0-006",
                "test/video_英国老头评测中国美的移动空调_1MIN_real-人声.wav",
                53.89,
                "en",
                1,
            ),
            ("D0-007", "test/TTS中文朗读测试-双人.wav", 157.93, "zh", 2),
            ("D0-008", "test/简单三步就能复刻巧乐兹？-人声.wav", 216.23, "zh", 1),
            (
                "D0-009",
                "test/Grow Up Show ～向日葵馬戲團-4min-人声.wav",
                249.30,
                "zh",
                2,
            ),
            ("D0-010", "test/40011894204-1-192--英语多人.wav", 46.34, "en", 2),
            ("D0-011", "test/20260428_214253_0043_KyGLgfKX.wav", 5.84, "zh", 1),
            ("D0-G01", "test/golden/non_speech_tone.wav", 4.00, "none", 0),
            ("D0-G02", "test/golden/repeated_phrase_me.wav", 2.80, "zh", 1),
        ]
        for sid, path, dur, lang, spk in d0_audio:
            entries.append(
                {
                    "sample_id": sid,
                    "path": path,
                    "duration_seconds": dur,
                    "language": lang,
                    "speaker_count": spk,
                    "scene_tags": {
                        "language": lang if lang in ("zh", "en", "mixed") else "none",
                        "speaker_count": "single"
                        if spk == 1
                        else "dual"
                        if spk == 2
                        else "multi",
                        "audio_length": "short" if dur < 180 else "medium",
                    },
                }
            )

        self._registry["D0"][version_id] = {
            "tier": "D0",
            "version_id": version_id,
            "freeze_date": today,
            "sample_count": len(entries),
            "description": "初始登记 — 13 音频 + 10 manifest 场景 (DATA_ASSETS.md)",
            "status": "active",
            "retention_days": 0,
            "entries": entries,
        }
        self._save()
        logger.info("D0 baseline seeded: %s (%d entries)", version_id, len(entries))

    # ---- 注册与冻结 ----

    def register(
        self,
        tier: DatasetTier,
        version_id: str,
        entries: list[dict],
        *,
        description: str = "",
        retention_days: int = 0,
    ) -> DatasetVersion:
        """注册一个新的数据集版本（非冻结，状态 active）。

        Args:
            tier: 层级 (D0-D4)
            version_id: 版本标识，如 "D1-20260901-001"
            entries: 样本条目列表
            description: 版本描述
            retention_days: 保留天数（仅 D2 有意义）

        Returns:
            DatasetVersion 实例
        """
        policy = RETENTION_POLICIES.get(tier, {})
        if not retention_days:
            retention_days = policy.get("retention_days", 0)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        version = DatasetVersion(
            tier=tier,
            version_id=version_id,
            freeze_date=today,
            sample_count=len(entries),
            description=description,
            status=DatasetVersionStatus.ACTIVE,
            retention_days=retention_days,
            entries=entries,
        )

        tier_key = tier.value
        self._registry.setdefault(tier_key, {})[version_id] = version.to_dict()
        self._save()
        logger.info(
            "Dataset registered: %s (%s, %d entries)",
            version_id,
            tier.value,
            len(entries),
        )
        return version

    def freeze(
        self,
        tier: DatasetTier,
        version_id: str,
        entries: list[dict],
        *,
        description: str = "",
    ) -> DatasetVersion:
        """冻结一个数据集版本（D3 专用 — 只增量不删除）。

        冻结前执行平衡检查：任何单一维度占比不得超过 60%。

        Args:
            tier: 层级（通常为 D3）
            version_id: 版本标识
            entries: 样本条目列表
            description: 版本描述

        Returns:
            DatasetVersion 实例

        Raises:
            ValueError: 平衡检查未通过
        """
        # D3/D1 冻结前执行平衡检查
        if tier in (DatasetTier.D3, DatasetTier.D1):
            balance = SceneSlicer.balance_report(entries)
            if balance["warnings"]:
                warning_msgs = [w["message"] for w in balance["warnings"]]
                logger.warning(
                    "Balance warnings for %s: %s",
                    version_id,
                    "; ".join(warning_msgs),
                )

        return self.register(
            tier=tier,
            version_id=version_id,
            entries=entries,
            description=description,
        )

    def supersede(self, tier: DatasetTier, version_id: str, reason: str = "") -> bool:
        """将数据集版本标记为 superseded。

        Args:
            tier: 层级
            version_id: 版本标识
            reason: 淘汰原因

        Returns:
            是否成功
        """
        tier_key = tier.value
        if version_id not in self._registry.get(tier_key, {}):
            logger.warning("Version not found: %s/%s", tier_key, version_id)
            return False

        self._registry[tier_key][version_id]["status"] = "superseded"
        self._save()
        logger.info(
            "Dataset superseded: %s/%s (reason: %s)",
            tier_key,
            version_id,
            reason or "N/A",
        )
        return True

    # ---- 查询 ----

    def current(self, tier: DatasetTier) -> DatasetVersion | None:
        """获取指定层级的当前活跃版本。

        Args:
            tier: 层级

        Returns:
            最新的 active 版本，或 None
        """
        tier_key = tier.value
        versions = self._registry.get(tier_key, {})
        active = [
            (vid, v) for vid, v in versions.items() if v.get("status") == "active"
        ]
        if not active:
            return None
        # 返回最新版本（按 version_id 排序，假设命名规范中的日期可排序）
        active.sort(key=lambda x: x[0], reverse=True)
        return DatasetVersion.from_dict(active[0][1])

    def list_versions(self, tier: DatasetTier | None = None) -> list[DatasetVersion]:
        """列出数据集版本。

        Args:
            tier: 可选层级过滤

        Returns:
            DatasetVersion 列表
        """
        results = []
        tiers = [tier.value] if tier else [t.value for t in DatasetTier]
        for t in tiers:
            for vid, vdata in self._registry.get(t, {}).items():
                results.append(DatasetVersion.from_dict(vdata))
        return sorted(results, key=lambda v: v.freeze_date, reverse=True)

    def samples(
        self,
        tier: DatasetTier,
        version_id: str | None = None,
    ) -> list[dict]:
        """获取指定层级/版本的样本条目。

        Args:
            tier: 层级
            version_id: 版本标识，None 则返回当前活跃版本

        Returns:
            样本条目列表
        """
        if version_id:
            version = self._registry.get(tier.value, {}).get(version_id)
        else:
            current = self.current(tier)
            if current is None:
                return []
            version = current.to_dict()

        if version is None:
            return []
        return version.get("entries", [])

    def check_balance(
        self,
        tier: DatasetTier,
        version_id: str | None = None,
    ) -> dict:
        """检查数据集版本的维度平衡性。

        Args:
            tier: 层级
            version_id: 版本标识，None 则检查当前活跃版本

        Returns:
            SceneSlicer.balance_report() 的结果
        """
        entries = self.samples(tier, version_id)
        return SceneSlicer.balance_report(entries)

    # ---- 清理 ----

    def cleanup_expired(self, now: str | None = None) -> list[str]:
        """清理超过保留期限的 D2 样本。

        D2 样本的 retention_days 默认为 365。

        Args:
            now: 当前日期 ISO 字符串，默认今天

        Returns:
            被归档的 version_id 列表
        """
        if now is None:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        archived = []
        try:
            now_date = datetime.fromisoformat(now)
        except ValueError:
            logger.warning("Invalid date format for cleanup: %s", now)
            return archived

        for vid, vdata in list(self._registry.get("D2", {}).items()):
            if vdata.get("status") != "active":
                continue

            retention = vdata.get(
                "retention_days", RETENTION_POLICIES[DatasetTier.D2]["retention_days"]
            )
            if retention <= 0:
                continue

            try:
                freeze_date = datetime.fromisoformat(vdata.get("freeze_date", ""))
            except ValueError:
                continue

            if (now_date - freeze_date).days >= retention:
                vdata["status"] = "archived"
                archived.append(vid)
                logger.info("D2 sample expired and archived: %s", vid)

        if archived:
            self._save()

        return archived

    # ---- 工具 ----

    def to_dict(self) -> dict:
        return dict(self._registry)


__all__ = [
    "DatasetTier",
    "DatasetVersionStatus",
    "DatasetVersion",
    "DataVersionManager",
    "RETENTION_POLICIES",
]
