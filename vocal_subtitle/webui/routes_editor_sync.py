"""编辑台字幕同步端点（editor-sync-v1）。

字幕打轴工作台（8631）每次编辑提交/导出后把当前字幕稿 POST 到这里，
处理台工作台可随时列出并下载最新稿（下载永远是最新调整的字幕）。

- `GET /api/editor-sync/capability` 能力探测；
- `POST /api/editor-sync/subtitle` 接收 JSON 载荷（name/format/content 必填），
  按字幕文件名 slug 化为 id，最新稿覆盖写（meta.json 记录更新时间、行数等）；
- `GET /api/editor-sync/subtitles` 列出全部同步稿（按保存时间倒序）；
- `GET /api/editor-sync/subtitles/{id}/download` 下载当前稿。

存储位置 `cache/editor_sync/`（与 journal_sink 同级的本地缓存目录）。
CORS：应用级 CORSMiddleware 对任意来源（含 file:// 的 null origin）回显许可，
standalone 编辑器可直接 POST。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from .runtime_state import state

logger = logging.getLogger(__name__)
router = APIRouter()

CAPABILITY = "editor-sync-v1"
ALLOWED_FORMATS = {"srt", "vtt", "ass", "ssa"}
MAX_CONTENT_CHARS = 5_000_000
# id 允许字母数字/中文/点/下划线/连字符；其余（路径分隔符等）一律替换为下划线
_SAFE_ID = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff._-]+")


def _sync_dir() -> Path:
    directory = state.upload_dir.parent / "editor_sync"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _slugify(name: str) -> str:
    slug = _SAFE_ID.sub("_", name.strip()).strip("._")
    return slug or "subtitle"


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return meta if isinstance(meta, dict) else None


@router.get("/editor-sync/capability")
async def editor_sync_capability():
    """能力探测：编辑器据此判断处理台支持字幕同步"""
    return {
        "accepted": True,
        "capability": CAPABILITY,
        "max_content_chars": MAX_CONTENT_CHARS,
    }


@router.post("/editor-sync/subtitle")
async def editor_sync_save(request: Request):
    """接收编辑台当前字幕稿（最新稿覆盖写，幂等）"""
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400, content={"ok": False, "error": "invalid JSON body"}
        )

    name = str(body.get("name") or "").strip()
    fmt = str(body.get("format") or "").strip().lower()
    content = body.get("content")
    if not name:
        return JSONResponse(
            status_code=422, content={"ok": False, "error": "name is required"}
        )
    if fmt not in ALLOWED_FORMATS:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": f"format must be one of {sorted(ALLOWED_FORMATS)}",
            },
        )
    if not isinstance(content, str) or not content.strip():
        return JSONResponse(
            status_code=422, content={"ok": False, "error": "content is required"}
        )
    if len(content) > MAX_CONTENT_CHARS:
        return JSONResponse(
            status_code=413, content={"ok": False, "error": "content too large"}
        )

    try:
        cue_count = max(0, int(body.get("cue_count") or 0))
    except (TypeError, ValueError):
        cue_count = 0

    sync_dir = _sync_dir()
    doc_id = _slugify(name)
    content_path = sync_dir / doc_id
    saved_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "id": doc_id,
        "name": name,
        "format": fmt,
        "media_name": str(body.get("media_name") or ""),
        "cue_count": cue_count,
        "include_speakers": bool(body.get("include_speakers", True)),
        "run_id": body.get("run_id") or None,
        "task_id": body.get("task_id") or None,
        "client_updated_at": body.get("updated_at") or None,
        "saved_at": saved_at,
        "size": len(content.encode("utf-8")),
    }
    try:
        content_path.write_text(content, encoding="utf-8")
        (sync_dir / f"{doc_id}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as err:
        logger.warning("editor_sync 写盘失败: %s", err)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "failed to persist subtitle"},
        )

    return {
        "ok": True,
        "id": doc_id,
        "saved_at": saved_at,
        "download": f"/api/editor-sync/subtitles/{doc_id}/download",
    }


@router.get("/editor-sync/subtitles")
async def editor_sync_list():
    """列出全部同步稿（按保存时间倒序）"""
    items: list[dict[str, Any]] = []
    for meta_path in _sync_dir().glob("*.meta.json"):
        meta = _load_meta(meta_path)
        if meta and meta.get("id"):
            items.append(meta)
    items.sort(key=lambda m: str(m.get("saved_at") or ""), reverse=True)
    return {"items": items, "count": len(items)}


@router.get("/editor-sync/subtitles/{doc_id}/download")
async def editor_sync_download(doc_id: str):
    """下载最新同步稿（Content-Disposition 用原始文件名）"""
    safe = _slugify(doc_id)
    content_path = _sync_dir() / safe
    if not content_path.is_file():
        return JSONResponse(status_code=404, content={"detail": "subtitle not found"})
    meta = _load_meta(_sync_dir() / f"{safe}.meta.json")
    filename = (meta or {}).get("name") or content_path.name
    return FileResponse(
        content_path, media_type="text/plain; charset=utf-8", filename=filename
    )


__all__ = [
    "CAPABILITY",
    "editor_sync_capability",
    "editor_sync_download",
    "editor_sync_list",
    "editor_sync_save",
    "router",
]
