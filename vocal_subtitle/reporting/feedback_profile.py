"""Read-only feedback profile evidence for run reports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def inspect_feedback_profile(config: Any) -> dict[str, Any]:
    feedback = getattr(config, "feedback", None)
    profile_name = str(getattr(feedback, "active_profile", "user_default"))
    profile_dir = Path(
        str(getattr(feedback, "user_profile_dir", "~/.vocal_subtitle/profiles"))
    ).expanduser()
    path = profile_dir / f"{profile_name}.yaml"
    result: dict[str, Any] = {
        "schema_version": "feedback-profile-v1",
        "profile": profile_name,
        "path": str(path),
        "requested": bool(getattr(feedback, "enabled", False)),
        "loaded": False,
        "applied": False,
        "mode": "disabled" if not getattr(feedback, "enabled", False) else "shadow",
        "fallback": False,
        "overrides_hash": None,
    }
    if not path.exists():
        result.update(
            {"status": "fallback_base", "fallback": True, "reason": "profile_missing"}
        )
        return result
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        overrides = payload.get("overrides", {}) if isinstance(payload, dict) else {}
        digest = hashlib.sha256(
            yaml.safe_dump(overrides, sort_keys=True, allow_unicode=True).encode(
                "utf-8"
            )
        ).hexdigest()
        result.update(
            {
                "status": "loaded_shadow",
                "loaded": True,
                "reason": "profile_loaded_but_not_applied_to_primary_run",
                "overrides_hash": digest,
                "override_count": len(overrides) if isinstance(overrides, dict) else 0,
                "profile_version": payload.get("updated_at")
                if isinstance(payload, dict)
                else None,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "fallback_base",
                "fallback": True,
                "reason": f"profile_load_failed:{type(exc).__name__}",
            }
        )
    return result


__all__ = ["inspect_feedback_profile"]
