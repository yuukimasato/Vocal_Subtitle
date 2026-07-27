"""持久化文件管理器

管理任务产出的持久化存储，支持按文件类型选择性持久化，
每种类型可配置独立的保留时长 (TTL)。

持久化设置存储在 cache/persistence_settings.json，
持久化文件存储在 cache/persistent_files/{task_id}/。
"""

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认持久化设置文件路径
DEFAULT_PERSISTENCE_DIR = Path(__file__).parent.parent.parent / "cache"
DEFAULT_PERSISTENCE_SETTINGS_PATH = DEFAULT_PERSISTENCE_DIR / "persistence_settings.json"
DEFAULT_FILES_DIR = DEFAULT_PERSISTENCE_DIR / "persistent_files"


@dataclass
class PersistenceSettings:
    """文件持久化设置"""

    # 哪些文件类型需要持久化
    persist_asr_subtitle: bool = True       # ASR 字幕文件
    persist_llm_subtitle: bool = True       # LLM 优化字幕文件
    persist_final_ass: bool = False         # 最终 ASS 字幕
    persist_final_srt: bool = True          # 最终 SRT 字幕
    persist_vocals: bool = True             # 人声分离音频
    persist_accompaniment: bool = False     # 背景声分离音频

    # 各类型的保留天数
    ttl_subtitle_days: int = 90             # 字幕文件（较小，保留更久）
    ttl_audio_days: int = 30                # 音频文件（较大，保留较短）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persist_asr_subtitle": self.persist_asr_subtitle,
            "persist_llm_subtitle": self.persist_llm_subtitle,
            "persist_final_ass": self.persist_final_ass,
            "persist_final_srt": self.persist_final_srt,
            "persist_vocals": self.persist_vocals,
            "persist_accompaniment": self.persist_accompaniment,
            "ttl_subtitle_days": self.ttl_subtitle_days,
            "ttl_audio_days": self.ttl_audio_days,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersistenceSettings":
        return cls(
            persist_asr_subtitle=d.get("persist_asr_subtitle", True),
            persist_llm_subtitle=d.get("persist_llm_subtitle", True),
            persist_final_ass=d.get("persist_final_ass", False),
            persist_final_srt=d.get("persist_final_srt", True),
            persist_vocals=d.get("persist_vocals", True),
            persist_accompaniment=d.get("persist_accompaniment", False),
            ttl_subtitle_days=d.get("ttl_subtitle_days", 90),
            ttl_audio_days=d.get("ttl_audio_days", 30),
        )


class PersistenceManager:
    """持久化文件管理器

    使用示例:
        mgr = PersistenceManager()
        settings = mgr.get_settings()
        mgr.persist_task("abc123", task_result, settings)
        mgr.cleanup_expired()
    """

    def __init__(
        self,
        settings_path: Optional[Path] = None,
        files_dir: Optional[Path] = None,
    ):
        self._settings_path = Path(settings_path or DEFAULT_PERSISTENCE_SETTINGS_PATH)
        self._files_dir = Path(files_dir or DEFAULT_FILES_DIR)
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._files_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 设置管理
    # ------------------------------------------------------------------

    def get_settings(self) -> PersistenceSettings:
        """获取当前持久化设置"""
        if self._settings_path.exists():
            try:
                raw = self._settings_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                return PersistenceSettings.from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load persistence settings: %s", e)
        # 返回默认设置
        settings = PersistenceSettings()
        self.save_settings(settings)
        return settings

    def save_settings(self, settings: PersistenceSettings) -> None:
        """保存持久化设置到磁盘"""
        try:
            self._settings_path.write_text(
                json.dumps(settings.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Persistence settings saved")
        except IOError as e:
            logger.error("Failed to save persistence settings: %s", e)

    # ------------------------------------------------------------------
    # 文件持久化
    # ------------------------------------------------------------------

    def persist_task(
        self,
        task_id: str,
        task_result: Dict[str, Any],
        settings: Optional[PersistenceSettings] = None,
    ) -> Dict[str, Any]:
        """将任务产出的文件持久化到 per-task 目录

        Args:
            task_id: 任务 ID
            task_result: 任务结果字典（含 subtitle_path, vocals_path 等）
            settings: 持久化设置，默认使用全局设置

        Returns:
            持久化文件信息字典: {"files": [...], "task_dir": "..."}
        """
        if settings is None:
            settings = self.get_settings()

        task_dir = self._files_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        persisted = []
        now_iso = datetime.now().isoformat()

        # 辅助函数：复制文件并记录元数据
        def _copy_file(src_path_str: Optional[str], label: str, ttl_days: int):
            if not src_path_str:
                return
            src = Path(src_path_str)
            if not src.exists():
                logger.warning("Source file not found for persistence: %s", src)
                return
            dst = task_dir / src.name
            try:
                shutil.copy2(src, dst)
                file_info = {
                    "label": label,
                    "path": str(dst),
                    "size_bytes": dst.stat().st_size,
                    "ttl_days": ttl_days,
                    "persisted_at": now_iso,
                    "expires_at": (datetime.now() + timedelta(days=ttl_days)).isoformat(),
                }
                persisted.append(file_info)
                logger.info("Persisted %s: %s", label, dst.name)
            except (IOError, OSError) as e:
                logger.warning("Failed to persist %s: %s", label, e)

        # ASR 字幕（干净版）
        if settings.persist_asr_subtitle:
            _copy_file(
                task_result.get("subtitle_path"),
                "ASR字幕",
                settings.ttl_subtitle_days,
            )

        # LLM 优化字幕
        if settings.persist_llm_subtitle:
            _copy_file(
                task_result.get("llm_subtitle_path"),
                "LLM优化字幕",
                settings.ttl_subtitle_days,
            )

        # 人声分离音频
        if settings.persist_vocals:
            _copy_file(
                task_result.get("vocals_path"),
                "人声音频",
                settings.ttl_audio_days,
            )

        # 背景声分离音频
        if settings.persist_accompaniment:
            _copy_file(
                task_result.get("accompaniment_path"),
                "背景声音频",
                settings.ttl_audio_days,
            )

        # 从字幕事件生成 ASS/SRT 最终版本并持久化
        events = task_result.get("events", [])
        if events and (settings.persist_final_ass or settings.persist_final_srt):
            try:
                from ..mapping.subtitle_builder import SubtitleBuilder, SubtitleRule
                from ..config import SubtitleBuildConfig

                rebuilt_events = [
                    type("SubtitleEvent", (), {
                        "index": e["index"],
                        "start": e["start"],
                        "end": e["end"],
                        "text": e["text"],
                        "original_text": e.get("original_text"),
                        "speaker_id": e.get("speaker_id"),
                        "speaker_label": e.get("speaker_label"),
                        "duration": e["end"] - e["start"],
                    })
                    for e in events
                ]

                builder = SubtitleBuilder(
                    rule=SubtitleRule(
                        min_duration=0.8,
                        max_duration=5.0,
                        max_chars_cjk=20,
                        max_chars_latin=42,
                        max_lines=2,
                    )
                )

                if settings.persist_final_srt:
                    srt_text = builder.build_to_string(rebuilt_events, fmt="srt")
                    srt_path = task_dir / "final.srt"
                    srt_path.write_text(srt_text, encoding="utf-8")
                    file_info = {
                        "label": "最终SRT字幕",
                        "path": str(srt_path),
                        "size_bytes": srt_path.stat().st_size,
                        "ttl_days": settings.ttl_subtitle_days,
                        "persisted_at": now_iso,
                        "expires_at": (datetime.now() + timedelta(days=settings.ttl_subtitle_days)).isoformat(),
                    }
                    persisted.append(file_info)
                    logger.info("Generated and persisted final SRT")

                if settings.persist_final_ass:
                    ass_text = builder.build_to_string(rebuilt_events, fmt="ass")
                    ass_path = task_dir / "final.ass"
                    ass_path.write_text(ass_text, encoding="utf-8")
                    file_info = {
                        "label": "最终ASS字幕",
                        "path": str(ass_path),
                        "size_bytes": ass_path.stat().st_size,
                        "ttl_days": settings.ttl_subtitle_days,
                        "persisted_at": now_iso,
                        "expires_at": (datetime.now() + timedelta(days=settings.ttl_subtitle_days)).isoformat(),
                    }
                    persisted.append(file_info)
                    logger.info("Generated and persisted final ASS")

            except Exception as e:
                logger.warning("Failed to generate persisted subtitle files: %s", e)

        # 写入元数据清单
        manifest = {
            "task_id": task_id,
            "persisted_at": now_iso,
            "files": persisted,
        }
        manifest_path = task_dir / "manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except IOError as e:
            logger.warning("Failed to write persistence manifest: %s", e)

        return manifest

    def get_persisted_files(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务的持久化文件信息"""
        manifest_path = self._files_dir / task_id / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return None

    def delete_persisted_task(self, task_id: str) -> bool:
        """删除任务的持久化文件"""
        task_dir = self._files_dir / task_id
        if not task_dir.exists():
            return False
        try:
            shutil.rmtree(task_dir)
            logger.info("Deleted persisted files for task %s", task_id)
            return True
        except OSError as e:
            logger.warning("Failed to delete persisted files for %s: %s", task_id, e)
            return False

    # ------------------------------------------------------------------
    # 过期清理
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """清理所有过期的持久化文件

        Returns:
            清理的任务目录数
        """
        cleaned = 0
        now = datetime.now()

        if not self._files_dir.exists():
            return 0

        for task_dir in self._files_dir.iterdir():
            if not task_dir.is_dir():
                continue

            manifest_path = task_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                files = manifest.get("files", [])

                # 检查是否有未过期的文件
                has_valid = False
                for f in files:
                    expires_str = f.get("expires_at", "")
                    if expires_str:
                        try:
                            expires_at = datetime.fromisoformat(expires_str)
                            if now < expires_at:
                                has_valid = True
                                break
                        except ValueError:
                            pass

                if not has_valid:
                    shutil.rmtree(task_dir)
                    cleaned += 1
                    logger.info("Cleaned expired persisted files: %s", task_dir.name)

            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.warning("Failed to process persistence manifest %s: %s", task_dir.name, e)

        return cleaned

    def get_persistence_stats(self) -> Dict[str, Any]:
        """获取持久化存储统计信息"""
        total_size = 0
        task_count = 0
        now = datetime.now()

        if not self._files_dir.exists():
            return {"total_size_mb": 0.0, "task_count": 0, "expired_count": 0}

        expired_count = 0
        for task_dir in self._files_dir.iterdir():
            if not task_dir.is_dir():
                continue
            task_count += 1
            for dirpath, _, filenames in os.walk(task_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass

            # 检查是否已过期
            manifest_path = task_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    files = manifest.get("files", [])
                    all_expired = True
                    for f in files:
                        expires_str = f.get("expires_at", "")
                        if expires_str:
                            try:
                                if now < datetime.fromisoformat(expires_str):
                                    all_expired = False
                                    break
                            except ValueError:
                                pass
                    if all_expired and files:
                        expired_count += 1
                except (json.JSONDecodeError, IOError):
                    pass

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "task_count": task_count,
            "expired_count": expired_count,
        }
