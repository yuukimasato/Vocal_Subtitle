"""缓存身份键(2026-09-15 重构计划 Task 7)。

缓存键由内容 hash、规范化配置、引擎/模型身份、阶段版本与 schema 版本
共同决定,不再绑定绝对路径:同内容不同路径命中,内容或版本变化失效。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

SCHEMA_VERSION = "cache-key-v1"


def normalize_params(params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """参数规范化:排序、去 None、数值/路径统一字符串化。"""
    normalized: Dict[str, Any] = {}
    for key in sorted(dict(params or {}).keys()):
        value = params[key]
        if value is None:
            continue
        if isinstance(value, Path):
            value = str(value)
        normalized[str(key)] = value
    return normalized


def build_cache_key(
    *,
    content_hash: str,
    stage: str,
    stage_version: str,
    engine: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    params: Optional[Mapping[str, Any]] = None,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """内容 + 版本身份键。

    Args:
        content_hash: 输入内容 hash(见 ``utils.file_hasher.compute_file_hash``)。
        stage: 阶段名(separation / asr / acoustic / ...)。
        stage_version: 阶段实现版本;变化使旧缓存失效。
        engine / model: 引擎与模型身份。
        config: 影响该阶段结果的规范化配置子集。
        params: 额外参数(排序、去 None 后并入)。
        schema_version: 键 schema 版本;升级后全部失效。
    """
    payload = {
        "schema_version": schema_version,
        "content_hash": content_hash,
        "stage": str(stage),
        "stage_version": str(stage_version),
        "engine": str(engine) if engine else None,
        "model": str(model) if model else None,
        "config": normalize_params(config),
        "params": normalize_params(params),
    }
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_identity(key: str) -> Dict[str, str]:
    """诊断用:返回键的前缀指纹(不含原始内容)。"""
    return {"schema_version": SCHEMA_VERSION, "key_prefix": key[:16]}


__all__ = ["build_cache_key", "normalize_params", "key_identity", "SCHEMA_VERSION"]
