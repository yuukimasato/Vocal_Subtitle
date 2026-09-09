"""编辑日志（edit-journal-v1）本地汇聚端点（V1.5 sink）。

能力探测 `GET /api/journal/sink` 供编辑器 feature-detect（standalone file:// 亦可用）；
`POST /api/journal/sink` 批量接收 NDJSON 或 JSON 事件，按 (session_id, seq) 去重后
追加落盘到 cache/journal_sink/<session_id>.jsonl。

CORS：应用级 CORSMiddleware 对任意来源（含 file:// 的 null origin）回显许可，
standalone 编辑器可直接 POST。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()

JOURNAL_SCHEMA = "edit-journal-v1"

# 事件必填字段（schema 校验）；diff 允许为空数组（无变化的 commit 不应出现，宽容处理）
_EVENT_REQUIRED_FIELDS = {"schema", "type", "session_id", "seq", "command"}
_SAFE_SESSION_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _sink_dir() -> Path:
    directory = state.upload_dir.parent / "journal_sink"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _session_filename(session_id: str) -> str:
    safe = _SAFE_SESSION_ID.sub("_", session_id).strip("._") or "unknown-session"
    return f"{safe}.jsonl"


def _load_seen_seq(path: Path) -> set[int]:
    """读取已落盘事件的 seq 集合（重复导出/重放 POST 幂等）"""
    seen: set[int] = set()
    if not path.exists():
        return seen
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "event":
                try:
                    seen.add(int(record.get("seq")))
                except (TypeError, ValueError):
                    continue
    except OSError:
        pass
    return seen


def _parse_payload(body: bytes, content_type: str) -> Optional[List[Dict[str, Any]]]:
    """解析 NDJSON（application/x-ndjson）与 JSON（对象/数组）两种形态。

    整体 JSON 优先尝试（单行 NDJSON 本身也是合法 JSON，语义一致）；
    解析失败且不像整体 JSON 时按 NDJSON 逐行解析。
    """
    text = body.decode("utf-8", errors="replace")
    if not text.strip():
        return None
    stripped = text.lstrip()
    if "json" in content_type or stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None  # 多行 NDJSON 不是整体合法 JSON → 走逐行解析
        if isinstance(data, list):
            return [record for record in data if isinstance(record, dict)]
        if isinstance(data, dict):
            events = data.get("events")
            if isinstance(events, list):
                return [record for record in events if isinstance(record, dict)]
            return [data]
        if stripped.startswith("[") or "json" in content_type and not stripped.count("\n"):
            return None
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict):
            return None
        records.append(record)
    return records or None


def _validate(records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
    """schema 校验：首行 header（可选），其后 event；返回 (合格记录, 拒绝数)"""
    valid: List[Dict[str, Any]] = []
    rejected = 0
    for record in records:
        if record.get("schema") != JOURNAL_SCHEMA:
            rejected += 1
            continue
        if record.get("type") == "header":
            valid.append(record)
            continue
        if record.get("type") == "event" and _EVENT_REQUIRED_FIELDS <= set(record):
            valid.append(record)
            continue
        rejected += 1
    return valid, rejected


@router.get("/journal/sink", status_code=202)
async def journal_sink_probe():
    """能力探测：编辑器据此判定后端支持 journal sink（V1.5）"""
    return {
        "accepted": True,
        "schema": JOURNAL_SCHEMA,
        "capabilities": ["ndjson", "json", "dedupe-by-session-seq"],
    }


@router.post("/journal/sink")
async def journal_sink(request: Request):
    """批量接收编辑日志事件（幂等：按 session_id + seq 去重）"""
    body = await request.body()
    records = _parse_payload(body, request.headers.get("content-type", ""))
    if records is None:
        return JSONResponse(
            status_code=400,
            content={"accepted": 0, "rejected": 0, "error": "invalid payload: expected NDJSON or JSON events"},
        )
    valid, rejected = _validate(records)
    if not valid:
        return JSONResponse(
            status_code=422,
            content={"accepted": 0, "rejected": rejected, "error": "no valid edit-journal-v1 records"},
        )

    # 按 session 分组落盘；header 记录到对应会话文件首部（仅首次）
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for record in valid:
        session_id = str(record.get("session_id") or record.get("id") or "unknown-session")
        by_session.setdefault(session_id, []).append(record)

    accepted = 0
    duplicates = 0
    for session_id, group in by_session.items():
        path = _sink_dir() / _session_filename(session_id)
        seen = _load_seen_seq(path)
        lines: List[str] = []
        for record in group:
            if record.get("type") == "event":
                try:
                    seq = int(record.get("seq"))
                except (TypeError, ValueError):
                    seq = -1
                if seq in seen:
                    duplicates += 1
                    continue
                seen.add(seq)
            elif record.get("type") == "header" and path.exists():
                # header 每次导出都会重发；文件已在即跳过（事件由 seq 去重）
                continue
            lines.append(json.dumps(record, ensure_ascii=False))
        if lines:
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            accepted += len(lines)

    return Response(
        status_code=202,
        media_type="application/json",
        content=json.dumps(
            {"accepted": accepted, "duplicates": duplicates, "rejected": rejected},
            ensure_ascii=False,
        ),
    )


__all__ = ["JOURNAL_SCHEMA", "journal_sink", "journal_sink_probe", "router"]
