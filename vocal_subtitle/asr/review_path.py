"""Independent global-ASR review and fallback policy."""

from __future__ import annotations

import re

from .contracts import ASRFailureRequest, ASRReviewRequest


class ASRReviewService:
    """Validate global results without depending on Pipeline state."""

    @staticmethod
    def validate(request: ASRReviewRequest) -> None:
        if not request.events:
            raise RuntimeError("global ASR returned no usable events")
        if not ASRReviewService.is_usable_transcript(request.transcript):
            raise RuntimeError("global ASR returned an invalid transcript")
        coverage = request.diagnostics.get("physical_coverage", {})
        if coverage and coverage.get("complete") is False:
            raise RuntimeError("global ASR physical coverage is incomplete")

    @staticmethod
    def is_usable_transcript(transcript: object) -> bool:
        status = getattr(transcript, "status", "unknown")
        words = getattr(transcript, "words", [])
        return status in {"ok", "degraded"} and bool(words)

    @staticmethod
    def classify_failure(request: ASRFailureRequest) -> str:
        exc = request.error
        category = getattr(exc, "category", None)
        if category:
            return str(category)
        msg = str(exc).lower()
        type_name = type(exc).__name__.lower()
        if isinstance(exc, ImportError):
            return "dependency_unavailable"
        if isinstance(exc, MemoryError):
            return "resource_unavailable"
        if "quality gate" in msg or "quality_gate" in msg:
            return "quality_gate_failed"
        if any(item in msg for item in ("degraded", "empty result", "no transcript")):
            return "invalid_result"
        if any(item in type_name + msg for item in ("memory", "oom", "cuda out")):
            return "resource_unavailable"
        return "execution_failed"

    @staticmethod
    def safe_failure_reason(exc: Exception) -> str:
        return re.sub(r"(api_key|token|key)=[\S]+", r"\1=***", str(exc))


class ASRReviewPath(ASRReviewService):
    """Backward-compatible class name for ASR review."""


__all__ = ["ASRReviewPath", "ASRReviewService"]
