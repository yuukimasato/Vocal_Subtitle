"""Independent capability maturity evidence for offline run reports."""

from __future__ import annotations

from typing import Any, Mapping


MATURITY_FIELDS = (
    "implemented",
    "runnable",
    "benchmarked",
    "gated",
    "release_default",
)


def _entry(*, implemented: bool, runnable: bool, benchmarked: bool, gated: bool, release_default: bool, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "implemented": bool(implemented),
        "runnable": bool(runnable),
        "benchmarked": bool(benchmarked),
        "gated": bool(gated),
        "release_default": bool(release_default),
        "evidence": dict(evidence or {}),
    }


def build_capability_maturity(
    *,
    evidence: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Build maturity facts without changing release status semantics."""
    review = dict(evidence or {})
    optional = review.get("optional_engines") or {}
    quality = dict(quality or {})
    gate_evidence = bool(quality.get("golden_gate_version") or quality.get("gate_report"))
    def optional_state(name: str) -> tuple[bool, bool, dict[str, Any]]:
        payload = optional.get(name) or {}
        status = str(payload.get("status", "unavailable"))
        runnable = status not in {"unavailable", "disabled", "model_missing", "port_missing"}
        return True, runnable, {
            "status": status,
            "reason": payload.get("reason"),
            "windows_processed": len(payload.get("windows") or []),
        }

    context_impl, context_runnable, context_evidence = optional_state("context_reasr")
    qwen_impl, qwen_runnable, qwen_evidence = optional_state("qwen")
    return {
        "schema_version": "capability-maturity-v1",
        "release_status_semantics": "independent_from_capability_maturity",
        "capabilities": {
            "segmented_primary": _entry(
                implemented=True,
                runnable=bool(review.get("candidate_count", 0) or review.get("status") in {"ok", "completed"}),
                benchmarked=bool(review.get("candidate_count", 0)),
                gated=gate_evidence,
                release_default=True,
                evidence={"candidate_count": review.get("candidate_count", 0)},
            ),
            "context_reasr": _entry(
                implemented=context_impl,
                runnable=context_runnable,
                benchmarked=bool(context_evidence.get("windows_processed")),
                gated=False,
                release_default=False,
                evidence=context_evidence,
            ),
            "qwen_review": _entry(
                implemented=qwen_impl,
                runnable=qwen_runnable,
                benchmarked=bool(qwen_evidence.get("windows_processed")),
                gated=False,
                release_default=False,
                evidence=qwen_evidence,
            ),
            "noise_shadow": _entry(
                implemented=True,
                runnable=True,
                benchmarked=bool(review.get("noise_shadow")),
                gated=False,
                release_default=False,
                evidence={"applied": False},
            ),
            "feedback_profile": _entry(
                implemented=True,
                runnable=bool(getattr(getattr(config, "feedback", None), "enabled", False)),
                benchmarked=False,
                gated=False,
                release_default=False,
                evidence={"active_profile": getattr(getattr(config, "feedback", None), "active_profile", None)},
            ),
        },
    }


__all__ = ["MATURITY_FIELDS", "build_capability_maturity"]
