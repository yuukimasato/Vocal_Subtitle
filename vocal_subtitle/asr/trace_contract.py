"""Shared source/offset/window identity helpers for segmented paths."""

from __future__ import annotations

import hashlib
from typing import Any

TRACE_SCHEMA_VERSION = "offline-trace-v1"


def attach_event_trace(
    event: Any,
    *,
    source_id: str,
    offset_id: str | None = None,
    window_id: str | None = None,
) -> Any:
    """Attach one idempotent evidence-contract record to a subtitle event."""
    source_word_ids = list(getattr(event, "source_word_ids", ()) or ())
    dedup_key = hashlib.sha1(
        "|".join(
            map(
                str,
                (
                    source_id,
                    offset_id,
                    window_id,
                    *source_word_ids,
                    getattr(event, "start", None),
                    getattr(event, "end", None),
                ),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    context = {
        **dict(getattr(event, "trace_context", {}) or {}),
        "schema_version": TRACE_SCHEMA_VERSION,
        "source_id": source_id,
        "offset_id": offset_id,
        "window_id": window_id,
        "candidate_id": f"{source_id}:event:{int(getattr(event, 'index', 0) or 0):06d}",
        "dedup_key": dedup_key,
    }
    event.trace_context = context
    marker = {
        "stage": "evidence_contract",
        "source_id": source_id,
        "offset_id": offset_id,
        "window_id": window_id,
        "dedup_key": dedup_key,
    }
    traces = list(getattr(event, "revision_trace", ()) or ())
    if not any(
        item.get("stage") == "evidence_contract" and item.get("dedup_key") == dedup_key
        for item in traces
        if isinstance(item, dict)
    ):
        traces.append(marker)
    event.revision_trace = traces
    return event


__all__ = ["TRACE_SCHEMA_VERSION", "attach_event_trace"]
