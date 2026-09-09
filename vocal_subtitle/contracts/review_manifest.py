"""review-manifest-v1 契约（管线 → 编辑器）

为每份字幕产物生成带 per-cue 出处（provenance）的审核清单，
供 subtitle-editor 载入后在编辑日志中记录 cue 级出处。

契约规则（docs/全栈架构定型与审核学习整合方案-2026-09-10.md §1.2/§3.1）：
  - 字段只允许追加可选字段（向后兼容），破坏性变更必须升版本号并存双读。
  - words/speaker_*/confidence/avg_logprob 为可选字段，有则填；
    数据复用 SubtitleEvent 既有字段，管线不新增计算。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REVIEW_MANIFEST_SCHEMA = "review-manifest-v1"


def manifest_filename(subtitle_path: str | Path) -> str:
    """字幕文件对应的清单文件名：<字幕同名>.manifest.json"""
    path = Path(subtitle_path)
    return f"{path.stem}.manifest.json"


def subtitle_sha256(path: str | Path) -> Optional[str]:
    """字幕文件内容哈希（清单生成时计算；读取失败返回 None）"""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _event_field(event: Any, key: str, default: Any = None) -> Any:
    """兼容 dict（webui 序列化事件）与 SubtitleEvent（dataclass）两种形态"""
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _words_of_event(event: Any) -> Optional[List[Dict[str, Any]]]:
    """提取词级时间戳为 [{w, t0, t1}]，无词级数据返回 None。

    SubtitleEvent.words 相对 event.start；这里还原为绝对时间。
    """
    start = _event_field(event, "start")
    raw_words = _event_field(event, "words") or []
    words: List[Dict[str, Any]] = []
    for word in raw_words:
        if isinstance(word, dict):
            w = word.get("word") or word.get("w")
            w_start = word.get("start")
            w_end = word.get("end")
            confidence = word.get("confidence")
        else:
            w = getattr(word, "word", None)
            w_start = getattr(word, "start", None)
            w_end = getattr(word, "end", None)
            confidence = getattr(word, "confidence", None)
        if w is None or w_start is None or w_end is None:
            continue
        entry: Dict[str, Any] = {
            "w": w,
            "t0": round(float(start) + float(w_start), 3),
            "t1": round(float(start) + float(w_end), 3),
        }
        if confidence is not None:
            entry["confidence"] = round(float(confidence), 4)
        words.append(entry)
    return words or None


def _source_stage(event: Any) -> str:
    """cue 文本来源阶段：LLM 覆盖过（original_text 保留）→ llm-optimized，否则 asr"""
    if _event_field(event, "original_text") is not None:
        return "llm-optimized"
    return "asr"


def _cue_entry(event: Any) -> Dict[str, Any]:
    """单条 cue 的出处条目（index 与字幕事件序号一致，1 起）"""
    cue: Dict[str, Any] = {
        "index": int(_event_field(event, "index", 0)),
        "source_stage": _source_stage(event),
    }
    words = _words_of_event(event)
    if words:
        confidences = [w["confidence"] for w in words if "confidence" in w]
        if confidences:
            cue["confidence"] = round(sum(confidences) / len(confidences), 4)
        cue["words"] = words
    speaker_id = _event_field(event, "speaker_id")
    if speaker_id is not None:
        cue["speaker_id"] = speaker_id
    speaker_label = _event_field(event, "speaker_label")
    if speaker_label:
        cue["speaker_label"] = speaker_label
    return cue


def build_review_manifest(
    *,
    task_id: str,
    run_id: str,
    subtitle_path: str | Path,
    events: List[Any],
    input_name: Optional[str] = None,
    input_sha256: Optional[str] = None,
    duration: Optional[float] = None,
    profile: Optional[str] = None,
    engines: Optional[Dict[str, str]] = None,
    stage: str = "final",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """从任务结果构建 review-manifest-v1 清单字典。

    Args:
        task_id: 任务 ID
        run_id: 管线运行 ID
        subtitle_path: 清单对应的字幕产物路径（哈希按其内容计算）
        events: 字幕事件列表（dict 或 SubtitleEvent 均可）
        input_name: 原始输入文件名
        input_sha256: 原始输入内容哈希（可选）
        duration: 音频时长秒（可选）
        profile: 场景模板名（可选）
        engines: 各阶段引擎名（可选，如 {"asr": "funasr"}）
        stage: 字幕阶段标识，默认 "final"
        created_at: ISO 时间戳（缺省取当前 UTC 时间）
    """
    subtitle_path = Path(subtitle_path)
    manifest: Dict[str, Any] = {
        "schema": REVIEW_MANIFEST_SCHEMA,
        "run_id": run_id,
        "task_id": task_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "input": {},
        "subtitle": {
            "file": subtitle_path.name,
            "format": subtitle_path.suffix.lstrip(".").lower() or "srt",
            "hash": subtitle_sha256(subtitle_path),
            "stage": stage,
        },
        "cues": [_cue_entry(event) for event in events],
    }
    if input_name:
        manifest["input"]["filename"] = input_name
    if input_sha256:
        manifest["input"]["sha256"] = input_sha256
    if duration is not None:
        manifest["input"]["duration"] = round(float(duration), 3)
    if profile:
        manifest["profile"] = profile
    if engines:
        manifest["engines"] = {k: v for k, v in engines.items() if v}
    return manifest


def write_review_manifest(subtitle_path: str | Path, manifest: Dict[str, Any]) -> Optional[Path]:
    """把清单写到字幕产物同目录（<字幕同名>.manifest.json）。

    Returns:
        写入路径；失败（非致命）返回 None。
    """
    path = Path(subtitle_path).parent / manifest_filename(subtitle_path)
    try:
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to write review manifest %s: %s", path, exc)
        return None
    return path


__all__ = [
    "REVIEW_MANIFEST_SCHEMA",
    "build_review_manifest",
    "manifest_filename",
    "subtitle_sha256",
    "write_review_manifest",
]
