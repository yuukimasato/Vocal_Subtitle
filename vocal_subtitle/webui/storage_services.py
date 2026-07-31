"""Storage and cache lifecycle services used by WebUI routes."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..config import ConfigLoader
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class WebUIStorageService:
    """Own cache construction and filesystem cleanup outside HTTP handlers."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def _cache(self) -> CacheManager:
        config = ConfigLoader().load_profile("default")
        cache_cfg = config.cache
        return CacheManager(
            cache_dir=cache_cfg.directory,
            ttl_separation=cache_cfg.ttl_separation,
            ttl_transcription=cache_cfg.ttl_transcription,
        )

    def clear_persistent_files(self) -> int:
        return self._cache().clear_persistent_files()

    def clear_stage(self, stage: str) -> Any:
        return self._cache().clear_stage(stage)

    def directory_size_mb(self, directory: Path) -> float:
        if not directory.exists():
            return 0.0
        total = 0
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    pass
        return total / (1024 * 1024)

    def clear_uploads(self, *, skip_names: Iterable[str] = ()) -> int:
        skip = set(skip_names)
        cleaned = 0
        upload_dir = self.state.upload_dir
        if not upload_dir.exists():
            return 0
        for item in upload_dir.iterdir():
            if item.is_dir() and item.name in skip:
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                cleaned += 1
            except OSError as exc:
                logger.warning("Failed to clean upload item %s: %s", item, exc)
        return cleaned


__all__ = ["WebUIStorageService"]
