"""会话目录管理器

基于输入文件 SHA256 哈希的目录结构，用于快速查重和组织处理产物。

目录结构:
    cache/uploads/{sha256[:16]}/
      ├── input{ext}               # 原始输入副本
      ├── metadata.json            # 元数据（原始文件名、时间戳、校验和等）
      ├── Human_Voice_Audio.wav    # 人声分离结果
      ├── BGM.wav                  # 背景声/伴奏
      ├── ASR-generated.srt        # ASR 干净版字幕
      ├── ASR-generated.vtt
      ├── ASR-generated.ass
      ├── LLM-optimized.srt        # LLM 优化版字幕
      ├── LLM-optimized.vtt
      └── LLM-optimized.ass

查重: os.path.exists(session_dir) → 已处理过
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .file_hasher import compute_file_hash

logger = logging.getLogger(__name__)

# 输出文件的标准化命名
OUTPUT_NAMES = {
    "vocals": "Human_Voice_Audio.wav",
    "accompaniment": "BGM.wav",
    "asr_srt": "ASR-generated.srt",
    "asr_vtt": "ASR-generated.vtt",
    "asr_ass": "ASR-generated.ass",
    "llm_srt": "LLM-optimized.srt",
    "llm_vtt": "LLM-optimized.vtt",
    "llm_ass": "LLM-optimized.ass",
}

# 字幕格式对应的 OUTPUT_NAMES key
ASR_FORMAT_KEYS = {"srt": "asr_srt", "vtt": "asr_vtt", "ass": "asr_ass"}
LLM_FORMAT_KEYS = {"srt": "llm_srt", "vtt": "llm_vtt", "ass": "llm_ass"}

# 所有字幕输出格式
ALL_SUBTITLE_FORMATS = ("srt", "vtt", "ass")


class SessionManager:
    """管理基于哈希的会话目录

    使用示例:
        mgr = SessionManager(Path("/path/to/uploads"))
        session_dir, is_duplicate = mgr.get_or_create(input_path, "original.aac")
        # 如果 is_duplicate 为 True，说明该文件已被处理过
    """

    def __init__(self, upload_dir: Path):
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_session_key(file_path: Path) -> str:
        """计算文件的会话目录键（SHA256 前 16 位）"""
        return compute_file_hash(file_path)[:16]

    @staticmethod
    def compute_session_key_from_bytes(data: bytes) -> str:
        """从字节数据计算会话目录键"""
        import hashlib
        return hashlib.sha256(data).hexdigest()[:16]

    def session_dir(self, session_key: str) -> Path:
        """获取会话目录路径"""
        return self._upload_dir / session_key

    def exists(self, session_key: str) -> bool:
        """检查会话目录是否已存在（即该文件已被处理过）"""
        return self.session_dir(session_key).exists()

    def get_or_create(
        self,
        input_path: Path,
        original_filename: str,
    ) -> tuple[Path, bool]:
        """获取或创建会话目录

        Args:
            input_path: 输入文件路径（已保存到磁盘）
            original_filename: 用户上传的原始文件名

        Returns:
            (session_dir, is_duplicate): 会话目录路径和是否重复
        """
        session_key = self.compute_session_key(input_path)
        session_dir = self.session_dir(session_key)
        is_duplicate = session_dir.exists()
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir, is_duplicate

    def save_input_copy(
        self,
        session_dir: Path,
        input_path: Path,
    ) -> Path:
        """将输入文件复制到会话目录（如尚未存在）

        Returns:
            会话目录中的输入文件路径
        """
        suffix = input_path.suffix or ".wav"
        dest = session_dir / f"input{suffix}"
        if not dest.exists():
            import shutil
            shutil.copy2(input_path, dest)
        return dest

    def write_metadata(
        self,
        session_dir: Path,
        *,
        original_filename: str,
        input_sha256: str,
        profile: str,
        config_hash: str,
        task_id: str = "",
        outputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Path:
        """写入 metadata.json

        Args:
            session_dir: 会话目录
            original_filename: 用户上传的原始文件名
            input_sha256: 输入文件 SHA256（完整）
            profile: 使用的场景模板名称
            config_hash: 配置哈希
            task_id: 任务 ID（可选）
            outputs: 输出文件信息，如 {"Human_Voice_Audio.wav": {"sha256": "...", "size": 12345}}

        Returns:
            metadata.json 路径
        """
        metadata = {
            "original_filename": original_filename,
            "input_sha256": input_sha256,
            "processed_at": datetime.now().isoformat(),
            "profile": profile,
            "config_hash": config_hash,
            "task_id": task_id,
            "outputs": outputs or {},
        }

        meta_path = session_dir / "metadata.json"

        # 如果已存在，合并 outputs 并保留非空字段
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                # 合并 outputs
                existing_outputs = existing.get("outputs", {})
                existing_outputs.update(metadata["outputs"])
                metadata["outputs"] = existing_outputs
                # 保留首次处理时间
                if "processed_at" in existing:
                    metadata["processed_at"] = existing["processed_at"]
                # 用已有非空值覆盖新的空值（保留原始信息）
                for field in ("original_filename", "input_sha256", "profile",
                              "config_hash", "task_id"):
                    if not metadata.get(field) and existing.get(field):
                        metadata[field] = existing[field]
            except (json.JSONDecodeError, IOError):
                pass

        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Session metadata written: %s", meta_path)
        return meta_path

    def read_metadata(self, session_dir: Path) -> Optional[Dict[str, Any]]:
        """读取 metadata.json"""
        meta_path = session_dir / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return None

    def get_output_path(self, session_dir: Path, name_key: str) -> Path:
        """获取标准化输出文件路径

        Args:
            session_dir: 会话目录
            name_key: OUTPUT_NAMES 中的键，如 "asr_srt", "llm_vtt", "vocals"

        Returns:
            标准化文件路径
        """
        return session_dir / OUTPUT_NAMES[name_key]

    def all_output_paths(self, session_dir: Path) -> Dict[str, Path]:
        """获取所有可能的输出文件路径"""
        return {key: session_dir / name for key, name in OUTPUT_NAMES.items()}

    def register_output(
        self,
        session_dir: Path,
        name_key: str,
        file_path: Path,
    ) -> None:
        """将文件复制/移动到标准化位置，并在 metadata.json 中注册

        Args:
            session_dir: 会话目录
            name_key: OUTPUT_NAMES 中的键
            file_path: 源文件路径
        """
        import shutil

        dest = self.get_output_path(session_dir, name_key)
        if Path(file_path) != dest:
            shutil.copy2(file_path, dest)

        # 更新 metadata
        file_hash = compute_file_hash(dest) if dest.exists() else ""
        file_size = dest.stat().st_size if dest.exists() else 0

        self.write_metadata(
            session_dir,
            original_filename="",  # 不覆盖已有值
            input_sha256="",
            profile="",
            config_hash="",
            outputs={
                dest.name: {
                    "sha256": file_hash,
                    "size": file_size,
                    "label": name_key,
                }
            },
        )
