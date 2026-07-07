"""缓存管理模块

基于 diskcache 实现磁盘持久化缓存，加速重复处理。
支持文件级缓存存储、统计信息查询和过期清理。
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """磁盘缓存管理器

    为不同处理阶段提供分层缓存，支持 TTL 自动过期。
    新增：文件持久化存储、缓存统计、磁盘用量查询。

    使用示例:
        cache = CacheManager(cache_dir="./cache")
        cache.set("separation", "file_hash_abc", result)
        cached = cache.get("separation", "file_hash_abc")
        info = cache.get_stats()
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl_separation: int = 86400 * 7,  # 默认 7 天
        ttl_transcription: int = 604800,   # 7 天
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._ttl_map = {
            "separation": ttl_separation,
            "transcription": ttl_transcription,
            "vad": ttl_separation,
            "subtitle": ttl_transcription,
            "pipeline": ttl_transcription,
        }

        # 使用独立目录避免不同阶段键冲突
        self._caches: dict = {}

        # 持久化文件存储目录
        self._files_dir = self._cache_dir / "persistent_files"
        self._files_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache(self, stage: str):
        """获取指定阶段的缓存实例"""
        if stage not in self._caches:
            try:
                import diskcache

                stage_dir = self._cache_dir / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                self._caches[stage] = diskcache.Cache(str(stage_dir))
            except ImportError:
                logger.warning(
                    "diskcache not available, using in-memory cache"
                )
                self._caches[stage] = {}
        return self._caches[stage]

    @staticmethod
    def make_key(input_path: Path, **params) -> str:
        """基于文件路径和参数生成缓存键

        Args:
            input_path: 输入文件路径
            **params: 影响结果的参数

        Returns:
            缓存键字符串 (SHA256)
        """
        payload = {
            "path": str(input_path.resolve()),
            "params": params,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, stage: str, key: str) -> Optional[Any]:
        """获取缓存值

        Args:
            stage: 处理阶段名称
            key: 缓存键

        Returns:
            缓存值或 None（过期或不存在）
        """
        cache = self._get_cache(stage)
        try:
            if isinstance(cache, dict):
                # 内存降级模式：检查 TTL 过期
                entry = cache.get(key)
                if entry is None:
                    return None
                value, expires_at = entry
                if time.time() > expires_at:
                    del cache[key]
                    return None
                return value
            return cache.get(key)
        except Exception as e:
            logger.warning("Cache get error for stage=%s: %s", stage, e)
            return None

    def set(
        self,
        stage: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """设置缓存值

        Args:
            stage: 处理阶段名称
            key: 缓存键
            value: 要缓存的值
            ttl: 过期时间（秒），默认使用阶段默认值
        """
        if ttl is None:
            ttl = self._ttl_map.get(stage, 86400)

        cache = self._get_cache(stage)
        try:
            if isinstance(cache, dict):
                # 内存降级模式：dict 不支持 expire 参数
                cache[key] = (value, time.time() + ttl)
            else:
                cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.warning("Cache set error for stage=%s: %s", stage, e)

    def delete(self, stage: str, key: str) -> bool:
        """删除缓存项"""
        cache = self._get_cache(stage)
        try:
            if isinstance(cache, dict):
                if key in cache:
                    del cache[key]
                    return True
                return False
            return cache.delete(key)
        except Exception:
            return False

    def clear_stage(self, stage: str) -> None:
        """清除整个阶段的缓存"""
        cache = self._get_cache(stage)
        try:
            if isinstance(cache, dict):
                cache.clear()
            else:
                cache.clear()
        except Exception as e:
            logger.warning("Cache clear error for stage=%s: %s", stage, e)
        # 同时清理对应的磁盘目录
        stage_dir = self._cache_dir / stage
        if stage_dir.exists():
            try:
                shutil.rmtree(stage_dir)
                stage_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning("Failed to remove stage dir: %s", e)
        # 清除实例引用以便重新创建
        if stage in self._caches:
            del self._caches[stage]

    def clear_all(self) -> None:
        """清除所有阶段缓存（不含持久化文件）

        持久化文件（分离音频、字幕等）与任务历史生命周期绑定，
        仅在清除历史记录时一并清理。
        """
        for stage in list(self._caches.keys()):
            self.clear_stage(stage)
        self._caches.clear()

    def clear_persistent_files(self) -> int:
        """清除持久化文件目录中的所有文件

        与 clear_history 联动：用户清除全部历史记录时调用。

        Returns:
            删除的文件数
        """
        deleted = 0
        if self._files_dir.exists():
            for item in self._files_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning("Failed to remove persistent item %s: %s", item, e)
        return deleted

    # ------------------------------------------------------------------
    # 持久化文件存储（用于分离结果等大文件）
    # ------------------------------------------------------------------

    def set_file(self, key: str, src_path: Path) -> Path:
        """将文件复制到持久化缓存目录

        Args:
            key: 缓存键（通常是 file_hash + engine + model）
            src_path: 源文件路径 (str 或 Path)

        Returns:
            缓存中的文件路径
        """
        src_path = Path(src_path)
        key_safe = hashlib.md5(key.encode()).hexdigest()[:16]
        dest_dir = self._files_dir / key_safe[:2] / key_safe[2:4]
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f"{key_safe}.wav"
        if not dest_path.exists() or dest_path.stat().st_size == 0:
            try:
                shutil.copy2(src_path, dest_path)
                logger.info("Cached file: %s -> %s", src_path.name, dest_path)
            except Exception as e:
                logger.warning("Failed to cache file: %s", e)
                return src_path  # 回退到原始路径

        return dest_path

    def get_file(self, key: str) -> Optional[Path]:
        """获取缓存的持久化文件

        Args:
            key: 缓存键

        Returns:
            文件路径或 None
        """
        key_safe = hashlib.md5(key.encode()).hexdigest()[:16]
        dest_path = self._files_dir / key_safe[:2] / key_safe[2:4] / f"{key_safe}.wav"
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path
        return None

    # ------------------------------------------------------------------
    # 统计与信息
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息

        Returns:
            {
                "stages": {"separation": {"count": N, "size_mb": M}, ...},
                "total_mb": float,
                "total_items": int,
                "files_dir_mb": float,
                "cache_dir": str,
            }
        """
        stages = {}
        total_items = 0

        for stage_dir in sorted(self._cache_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            stage_name = stage_dir.name
            if stage_name == "persistent_files":
                continue

            size_mb = self._dir_size_mb(stage_dir)
            cache = self._get_cache(stage_name)
            count = len(cache) if hasattr(cache, "__len__") else 0
            total_items += count
            stages[stage_name] = {
                "count": count,
                "size_mb": round(size_mb, 2),
            }

        total_cache_mb = self._dir_size_mb(self._cache_dir)
        files_mb = self._dir_size_mb(self._files_dir)

        return {
            "stages": stages,
            "total_mb": round(total_cache_mb, 2),
            "total_items": total_items,
            "files_dir_mb": round(files_mb, 2),
            "cache_dir": str(self._cache_dir.resolve()),
            "ttl_map": dict(self._ttl_map),
        }

    def get_disk_usage_mb(self) -> float:
        """获取缓存总磁盘用量 (MB)"""
        return self._dir_size_mb(self._cache_dir)

    def clear_expired(self) -> int:
        """清理所有阶段的过期缓存项

        Returns:
            清理的项数
        """
        cleaned = 0
        for stage in list(self._caches.keys()):
            cache = self._get_cache(stage)
            if hasattr(cache, "expire"):
                try:
                    cache.expire()
                    cleaned += 1
                except Exception:
                    pass
        return cleaned

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_size_mb(directory: Path) -> float:
        """计算目录总大小 (MB)"""
        if not directory.exists():
            return 0.0
        total = 0
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total / (1024 * 1024)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir
