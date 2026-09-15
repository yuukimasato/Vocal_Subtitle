"""Stable cache contracts for ASR evidence stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from .evidence import (
    DECISION_POLICY_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    RISK_POLICY_VERSION,
)

EVIDENCE_CACHE_STAGE = "evidence"
CACHE_KEY_VERSION = "evidence-cache-v2"


class EvidenceCachePort(Protocol):
    """Minimal cache capability required by evidence review."""

    def get(self, stage: str, key: str) -> Any: ...

    def set(self, stage: str, key: str, value: Any, ttl: int | None = None) -> None: ...


@dataclass(frozen=True)
class EvidenceCacheKeyContext:
    """Every input that can change one bounded evidence result."""

    input_hash: str
    audio_hash: str
    physical_timeline_version: str
    window_start: float
    window_end: float
    phase: str
    engine: str
    model: str
    language: str | None
    route_version: str
    evidence_schema_version: str = EVIDENCE_SCHEMA_VERSION
    risk_policy_version: str = RISK_POLICY_VERSION
    decision_policy_version: str = DECISION_POLICY_VERSION
    sample_rate: int = 16000
    physical_clip_id: str = ""
    review_policy_version: str = "review-policy-v1"
    pair_route_version: str = ""
    cache_key_version: str = CACHE_KEY_VERSION

    def __post_init__(self) -> None:
        if not self.input_hash and not self.audio_hash:
            raise ValueError("at least one audio/input hash is required")
        if not self.phase.strip():
            raise ValueError("phase must not be empty")
        if self.window_start < 0 or self.window_end <= self.window_start:
            raise ValueError("window must satisfy 0 <= start < end")
        if isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "audio_hash": self.audio_hash,
            "physical_timeline_version": self.physical_timeline_version,
            "window": {"start": self.window_start, "end": self.window_end},
            "phase": self.phase,
            "engine": self.engine,
            "model": self.model,
            "language": self.language,
            "route_version": self.route_version,
            "sample_rate": self.sample_rate,
            "evidence_schema_version": self.evidence_schema_version,
            "risk_policy_version": self.risk_policy_version,
            "decision_policy_version": self.decision_policy_version,
            "physical_clip_id": self.physical_clip_id,
            "review_policy_version": self.review_policy_version,
            "pair_route_version": self.pair_route_version,
            "cache_key_version": self.cache_key_version,
        }

    def key(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audio_fingerprint(audio: Any) -> str:
    """Hash an in-memory audio buffer without coupling to a numeric library."""
    if audio is None:
        return ""
    try:
        raw = audio.tobytes() if hasattr(audio, "tobytes") else bytes(audio)
    except (TypeError, ValueError):
        return ""
    metadata = {
        "type": type(audio).__name__,
        "dtype": str(getattr(audio, "dtype", "")),
        "shape": tuple(getattr(audio, "shape", ())),
        "length": len(raw),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8"))
    digest.update(raw)
    return digest.hexdigest()


__all__ = [
    "EVIDENCE_CACHE_STAGE",
    "CACHE_KEY_VERSION",
    "EvidenceCacheKeyContext",
    "EvidenceCachePort",
    "audio_fingerprint",
]
