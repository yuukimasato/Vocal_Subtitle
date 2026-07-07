"""用户配置文件管理器

管理 ~/.vocal_subtitle/profiles/ 下的用户配置文件，
支持 CRUD、自动备份、回滚、置信度衰减。
"""

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import FeedbackConfig, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数分级半衰期映射
# ---------------------------------------------------------------------------

PARAM_TIER_HALF_LIFE: Dict[str, Optional[int]] = {
    # 长期偏好 — 半衰期 180 天（或不衰减）
    "subtitle.max_chars_cjk": 180,
    "subtitle.max_chars_latin": 180,
    "subtitle.max_duration": 180,
    "subtitle.min_duration": 180,
    "subtitle.max_lines": None,  # 行业标准，不衰减
    # 中期偏好 — 半衰期 90 天（默认）
    "merge_decision.fast_merge_max_gap": 90,
    "merge_decision.llm_decision_min_gap": 90,
    "merge_decision.llm_decision_max_gap": 90,
    "merge_decision.hard_split_min_gap": 90,
    "merge_decision.max_combined_duration": 90,
    # 短期环境 — 半衰期 60 天（声学相关字段）
    "merging.padding": 60,
    "merging.padding_max": 60,
    "merging.padding_min": 60,
    "merging.min_silence_gap": 60,
    "vad.threshold": 60,
    "vad.min_silence_duration_ms": 60,
    "noise_reduction.spectral_noise_reduction_db": 60,
    "noise_reduction.burst_noise_threshold_db": 60,
}

# 默认半衰期
DEFAULT_HALF_LIFE_DAYS = 90

# 最大保留备份数
MAX_BACKUPS = 3


def get_param_half_life(param_path: str) -> Optional[int]:
    """获取参数的分级半衰期（天），None 表示不衰减"""
    import fnmatch

    for pattern, days in PARAM_TIER_HALF_LIFE.items():
        if fnmatch.fnmatch(param_path, pattern):
            return days
    return DEFAULT_HALF_LIFE_DAYS


class UserProfileManager:
    """用户配置文件管理器

    管理 ~/.vocal_subtitle/profiles/ 下的 YAML 配置文件，
    作为默认配置的增量"补丁"。

    使用示例:
        mgr = UserProfileManager()
        profile = mgr.load("user_default")
        overrides = profile.get("overrides", {})
        config = mgr.merge_with_base(base_config, overrides)
    """

    def __init__(self, config: Optional[FeedbackConfig] = None):
        self._config = config or FeedbackConfig()
        profile_dir = os.path.expanduser(self._config.user_profile_dir)
        self._profile_dir = Path(profile_dir)
        self._profile_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 配置文件路径
    # ------------------------------------------------------------------

    def _profile_path(self, profile_name: str) -> Path:
        return self._profile_dir / f"{profile_name}.yaml"

    def _backup_path(self, profile_name: str, index: int) -> Path:
        return self._profile_dir / f"{profile_name}.yaml.bak.{index}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def load(self, profile_name: str = "user_default") -> Dict[str, Any]:
        """加载用户配置，若不存在返回默认空配置"""
        path = self._profile_path(profile_name)
        if not path.exists():
            return self._create_default(profile_name)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or self._create_default(profile_name)

    def save(self, profile: Dict[str, Any]) -> None:
        """保存用户配置，自动轮转备份"""
        profile_name = profile.get("profile_id", "user_default")
        path = self._profile_path(profile_name)

        # 更新时间戳
        profile["updated_at"] = datetime.now().isoformat()

        # 若原文件存在，先做备份轮转
        if path.exists():
            self._rotate_backups(profile_name)

        # 写入新配置
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(profile, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info("User profile saved: %s (feedback_count=%d)", profile_name, profile.get("feedback_count", 0))

    def delete(self, profile_name: str) -> bool:
        """删除用户配置及其备份"""
        path = self._profile_path(profile_name)
        deleted = False
        if path.exists():
            path.unlink()
            deleted = True
        for i in range(1, MAX_BACKUPS + 1):
            bak = self._backup_path(profile_name, i)
            if bak.exists():
                bak.unlink()
        if deleted:
            logger.info("User profile deleted: %s", profile_name)
        return deleted

    def list_profiles(self) -> List[str]:
        """列出所有用户配置名称"""
        profiles = []
        for f in self._profile_dir.glob("*.yaml"):
            if ".bak." not in f.name:
                profiles.append(f.stem)
        return sorted(profiles)

    # ------------------------------------------------------------------
    # 备份与回滚
    # ------------------------------------------------------------------

    def _rotate_backups(self, profile_name: str) -> None:
        """轮转备份：bak.2 → bak.3, bak.1 → bak.2, 当前 → bak.1"""
        # 删除最旧的备份
        oldest = self._backup_path(profile_name, MAX_BACKUPS)
        oldest.unlink(missing_ok=True)

        # 轮转
        for i in range(MAX_BACKUPS - 1, 0, -1):
            src = self._backup_path(profile_name, i)
            dst = self._backup_path(profile_name, i + 1)
            if src.exists():
                shutil.copy2(src, dst)

        # 复制当前版本为 bak.1
        current = self._profile_path(profile_name)
        if current.exists():
            shutil.copy2(current, self._backup_path(profile_name, 1))

    def rollback(self, profile_name: str = "user_default") -> Dict[str, Any]:
        """回滚到上一个备份版本

        Returns:
            回滚后的配置字典

        Raises:
            FileNotFoundError: 无可用备份
        """
        bak1 = self._backup_path(profile_name, 1)
        if not bak1.exists():
            raise FileNotFoundError(f"No backup found for profile '{profile_name}'")

        current = self._profile_path(profile_name)
        # 当前版本保存为临时副本
        if current.exists():
            shutil.copy2(current, self._backup_path(profile_name, 0))

        # 恢复 bak.1
        shutil.copy2(bak1, current)
        logger.info("User profile rolled back: %s", profile_name)

        return self.load(profile_name)

    def reset(self, profile_name: str = "user_default") -> Dict[str, Any]:
        """重置为系统默认（删除用户配置，返回空配置）"""
        # 先备份当前版本
        self._rotate_backups(profile_name)
        # 删除当前配置
        self._profile_path(profile_name).unlink(missing_ok=True)
        logger.info("User profile reset: %s", profile_name)
        return self._create_default(profile_name)

    # ------------------------------------------------------------------
    # 配置合并
    # ------------------------------------------------------------------

    @staticmethod
    def merge_with_base(
        base_config: PipelineConfig,
        user_overrides: Dict[str, Any],
    ) -> PipelineConfig:
        """将用户覆盖合并到基础配置，生成最终运行配置

        Args:
            base_config: 从场景模板加载的 PipelineConfig
            user_overrides: 用户配置中的 overrides 字典

        Returns:
            合并后的 PipelineConfig
        """
        from ..config import ConfigLoader

        return ConfigLoader.apply_user_profile_overrides(base_config, user_overrides)

    # ------------------------------------------------------------------
    # 置信度衰减
    # ------------------------------------------------------------------

    @staticmethod
    def decay_weight(
        timestamp: str,
        half_life_days: Optional[int] = None,
        current_time: Optional[datetime] = None,
    ) -> float:
        """计算历史记录的时间衰减权重

        使用指数衰减: weight = exp(-t / half_life)

        Args:
            timestamp: 记录的 ISO 时间戳
            half_life_days: 半衰期（天），None 表示不衰减（返回 1.0）
            current_time: 当前时间，默认 now()

        Returns:
            衰减权重 [0, 1]
        """
        if half_life_days is None:
            return 1.0

        try:
            record_time = datetime.fromisoformat(timestamp)
        except (ValueError, TypeError):
            return 1.0

        now = current_time or datetime.now()
        elapsed_days = (now - record_time).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return 1.0

        import math
        return math.exp(-elapsed_days / half_life_days)

    @staticmethod
    def compute_effective_weight(
        timestamp: str,
        param_path: str,
        current_time: Optional[datetime] = None,
    ) -> float:
        """计算某条历史记录对某参数的有效权重

        结合参数分级半衰期和时间衰减。

        Args:
            timestamp: 记录时间戳
            param_path: 参数路径 (如 "merging.padding")
            current_time: 当前时间

        Returns:
            有效权重 [0, 1]
        """
        half_life = get_param_half_life(param_path)
        return UserProfileManager.decay_weight(timestamp, half_life, current_time)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _create_default(self, profile_name: str) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        return {
            "profile_id": profile_name,
            "base_profile": "default",
            "description": "",
            "created_at": now,
            "updated_at": now,
            "feedback_count": 0,
            "is_active": True,
            "fingerprint": {},
            "overrides": {},
            "history": [],
            "few_shot_examples": [],
        }
