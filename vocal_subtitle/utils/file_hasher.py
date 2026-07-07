"""文件哈希工具

为缓存键和任务历史提供文件内容指纹。
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

# 分块读取大小 (8KB)
_CHUNK_SIZE = 8192


def compute_file_hash(
    path: Path,
    algorithm: str = "sha256",
    max_size_mb: Optional[float] = None,
) -> str:
    """计算文件内容的哈希值

    Args:
        path: 文件路径
        algorithm: 哈希算法 (sha256 / md5 / sha1)
        max_size_mb: 如果文件超过此大小则只对首尾采样，
                     避免大文件哈希过慢。None 表示始终全量。

    Returns:
        十六进制哈希字符串

    Raises:
        FileNotFoundError: 文件不存在
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size = path.stat().st_size
    hasher = hashlib.new(algorithm)

    # 大文件采样策略：对首 4MB + 尾 4MB 做哈希
    if max_size_mb is not None and file_size > max_size_mb * 1024 * 1024:
        sample_size = 4 * 1024 * 1024  # 4MB
        with open(path, "rb") as f:
            # 头部
            hasher.update(f.read(sample_size))
            # 尾部
            f.seek(-sample_size, 2)
            hasher.update(f.read(sample_size))
        # 追加文件大小以确保唯一性
        hasher.update(str(file_size).encode())
    else:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)

    return hasher.hexdigest()


def compute_config_hash(config) -> str:
    """计算配置对象的哈希值

    将 PipelineConfig 序列化为确定性 JSON，加上核心管道代码
    的版本指纹后计算 SHA256。用于完整管道缓存键。

    代码指纹确保管道逻辑修复后旧缓存自动失效，
    避免代码层面的 bug 修复被全管道缓存绕过。

    Args:
        config: PipelineConfig 对象

    Returns:
        十六进制 SHA256 哈希字符串
    """
    from dataclasses import asdict

    config_dict = asdict(config)
    # 排序保证确定性
    raw = json.dumps(config_dict, sort_keys=True, default=str)

    # 混入核心管道代码的指纹 — 代码变更自动使旧缓存失效
    raw += _pipeline_code_fingerprint()

    return hashlib.sha256(raw.encode()).hexdigest()


_PIPELINE_FINGERPRINT_CACHE: Optional[str] = None


def _pipeline_code_fingerprint() -> str:
    """计算核心管道源码的轻量指纹

    对 pipeline.py 和 ASR 引擎文件做采样哈希，
    确保代码级修复能使全管道缓存自动失效。
    """
    global _PIPELINE_FINGERPRINT_CACHE
    if _PIPELINE_FINGERPRINT_CACHE is not None:
        return _PIPELINE_FINGERPRINT_CACHE

    import os

    sources: list[str] = []
    # 核心管道文件（相对于项目根目录）
    core_files = [
        "vocal_subtitle/pipeline.py",
        "vocal_subtitle/asr/faster_whisper_engine.py",
        "vocal_subtitle/asr/funasr_engine.py",
        "vocal_subtitle/asr/whisper_cpp_engine.py",
        "vocal_subtitle/asr/base.py",
        "vocal_subtitle/asr/boundary_reasr.py",
    ]

    # 查找项目根目录（包含 vocal_subtitle 包的目录）
    import vocal_subtitle
    project_root = Path(vocal_subtitle.__file__).parent.parent

    for rel_path in core_files:
        fpath = project_root / rel_path
        if fpath.exists():
            # 采样：取文件首尾 1KB + 文件大小
            stat = fpath.stat()
            sources.append(str(stat.st_size))
            with open(fpath, "rb") as f:
                sources.append(f.read(1024).hex())
                if stat.st_size > 2048:
                    f.seek(-1024, os.SEEK_END)
                    sources.append(f.read(1024).hex())

    _PIPELINE_FINGERPRINT_CACHE = "|".join(sources)
    return _PIPELINE_FINGERPRINT_CACHE
