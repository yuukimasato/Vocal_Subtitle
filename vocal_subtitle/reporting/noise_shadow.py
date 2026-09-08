"""Noise diagnostics and non-invasive threshold recommendations."""

from __future__ import annotations

from typing import Any, Mapping


def _bounded(value: float, low: float, high: float) -> float:
    return round(max(low, min(high, value)), 3)


def build_noise_shadow(
    *,
    noise_floor_db: float | None = None,
    current_vad_threshold: float | None = None,
    current_skeleton_noise_db: float | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return suggestions and explicitly prove they were not applied."""
    details = dict(diagnostics or {})
    if noise_floor_db is None:
        for key in ("noise_floor_db", "estimated_noise_floor_db"):
            if details.get(key) is not None:
                noise_floor_db = float(details[key])
                break
    if noise_floor_db is None:
        return {
            "schema_version": "noise-shadow-v1",
            "status": "not_evaluable",
            "reason": "noise_floor_missing",
            "suggested_vad_threshold": None,
            "suggested_skeleton_noise_db": None,
            "applied": False,
            "overrides": {},
        }
    skeleton = _bounded(noise_floor_db + 10.0, -45.0, -30.0)
    suggested_vad = (
        _bounded(float(current_vad_threshold), 0.0, 1.0)
        if current_vad_threshold is not None
        else None
    )
    return {
        "schema_version": "noise-shadow-v1",
        "status": "advisory",
        "noise_floor_db": round(float(noise_floor_db), 3),
        "suggested_vad_threshold": suggested_vad,
        "suggested_skeleton_noise_db": skeleton,
        "current_vad_threshold": current_vad_threshold,
        "current_skeleton_noise_db": current_skeleton_noise_db,
        "applied": False,
        "overrides": {},
        "reason": "shadow_only_until_quality_gate",
    }


__all__ = ["build_noise_shadow"]
