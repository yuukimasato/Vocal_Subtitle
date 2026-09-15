"""Collect optional non-ASR evidence for already scheduled review windows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from .evidence import CandidateEvidence, DecisionEvidenceBundle, EvidenceWord
from .review_engines import ForcedAlignerPort, SEDPort, SemanticReviewPort
from .review_scheduler import ReviewWindow
from .review_telemetry import resource_snapshot, timed_call


def _json_value(value: Any) -> Any:
    if isinstance(value, EvidenceWord):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return _json_value(value.to_dict())
        except Exception:
            pass
    return str(value)


class SecondaryEvidenceCollector:
    """Run auxiliary evidence providers without changing subtitle decisions."""

    def collect(
        self,
        *,
        audio: Any,
        sample_rate: int,
        windows: Sequence[ReviewWindow],
        candidates: Iterable[CandidateEvidence],
        language: str | None,
        config: Any,
        forced_aligner: ForcedAlignerPort | None = None,
        sed: SEDPort | None = None,
        semantic_review: SemanticReviewPort | None = None,
    ) -> dict[str, Any]:
        candidate_list = list(candidates)
        by_id = {item.id: item for item in candidate_list}
        result = {
            "forced_aligner": self._collect_forced_alignment(
                audio=audio,
                sample_rate=sample_rate,
                windows=windows,
                by_id=by_id,
                language=language,
                enabled=bool(getattr(config, "forced_aligner_enabled", False)),
                engine=forced_aligner,
            ),
            "sed": self._collect_sed(
                audio=audio,
                sample_rate=sample_rate,
                windows=windows,
                language=language,
                enabled=bool(getattr(config, "sed_enabled", False)),
                engine=sed,
            ),
            "semantic_review": self._collect_semantic(
                windows=windows,
                by_id=by_id,
                enabled=bool(getattr(config, "semantic_review_enabled", False)),
                engine=semantic_review,
            ),
        }
        result["bundles"] = [
            item.to_dict() for item in secondary_evidence_to_bundles(result, windows)
        ]
        return result

    @staticmethod
    def _availability(engine: Any) -> dict[str, Any] | None:
        if engine is None or not hasattr(engine, "availability"):
            return None
        try:
            return dict(engine.availability())
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": "availability_check_failed",
                "error": str(exc),
            }

    def _stage_status(
        self,
        *,
        enabled: bool,
        engine: Any,
        availability: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not enabled:
            return {"status": "disabled", "reason": "feature_disabled", "windows": []}
        if engine is None:
            return {"status": "unavailable", "reason": "port_missing", "windows": []}
        if availability and availability.get("status") == "unavailable":
            return {
                "status": "unavailable",
                "reason": availability.get("reason", "engine_unavailable"),
                "engine": getattr(engine, "name", None),
                "windows": [],
            }
        return {"status": "ok", "engine": getattr(engine, "name", None), "windows": []}

    def _collect_forced_alignment(
        self,
        *,
        audio: Any,
        sample_rate: int,
        windows: Sequence[ReviewWindow],
        by_id: dict[str, CandidateEvidence],
        language: str | None,
        enabled: bool,
        engine: ForcedAlignerPort | None,
    ) -> dict[str, Any]:
        result = self._stage_status(
            enabled=enabled,
            engine=engine,
            availability=self._availability(engine),
        )
        if result["status"] != "ok" or audio is None:
            if enabled and audio is None and result["status"] == "ok":
                result.update({"status": "unavailable", "reason": "audio_missing"})
            return result
        for window in windows:
            for candidate_id in window.candidate_ids:
                candidate = by_id.get(candidate_id)
                if candidate is None:
                    continue
                item = {
                    "window_id": window.id,
                    "candidate_id": candidate.id,
                    "text": candidate.text,
                }
                try:
                    raw, elapsed, resources = timed_call(
                        engine.align,
                        audio,
                        sample_rate,
                        candidate.text,
                        window,
                        language=language,
                    )
                    words = [_json_value(word) for word in (raw or ())]
                    item.update(
                        {
                            "status": "ok",
                            "word_count": len(words),
                            "words": words,
                            "wall_time_seconds": elapsed,
                            "resources": resources,
                        }
                    )
                except Exception as exc:
                    item.update(
                        {
                            "status": "failed",
                            "error": str(exc),
                            "resources": resource_snapshot(),
                        }
                    )
                result["windows"].append(item)
        result["status"] = (
            "degraded"
            if any(item["status"] == "failed" for item in result["windows"])
            else "ok"
        )
        result["window_count"] = len(result["windows"])
        return result

    def _collect_sed(
        self,
        *,
        audio: Any,
        sample_rate: int,
        windows: Sequence[ReviewWindow],
        language: str | None,
        enabled: bool,
        engine: SEDPort | None,
    ) -> dict[str, Any]:
        del language
        result = self._stage_status(
            enabled=enabled,
            engine=engine,
            availability=self._availability(engine),
        )
        if result["status"] != "ok" or audio is None:
            if enabled and audio is None and result["status"] == "ok":
                result.update({"status": "unavailable", "reason": "audio_missing"})
            return result
        for window in windows:
            item = {"window_id": window.id, "start": window.start, "end": window.end}
            try:
                raw, elapsed, resources = timed_call(
                    engine.detect,
                    audio,
                    sample_rate,
                    window,
                )
                item.update(
                    {
                        "status": "ok",
                        "evidence": _json_value(raw),
                        "wall_time_seconds": elapsed,
                        "resources": resources,
                    }
                )
            except Exception as exc:
                item.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "resources": resource_snapshot(),
                    }
                )
            result["windows"].append(item)
        result["status"] = (
            "degraded"
            if any(item["status"] == "failed" for item in result["windows"])
            else "ok"
        )
        result["window_count"] = len(result["windows"])
        return result

    def _collect_semantic(
        self,
        *,
        windows: Sequence[ReviewWindow],
        by_id: dict[str, CandidateEvidence],
        enabled: bool,
        engine: SemanticReviewPort | None,
    ) -> dict[str, Any]:
        result = self._stage_status(
            enabled=enabled,
            engine=engine,
            availability=self._availability(engine),
        )
        if result["status"] != "ok":
            return result
        for window in windows:
            context = " ".join(
                by_id[item].text for item in window.candidate_ids if item in by_id
            )
            for candidate_id in window.candidate_ids:
                candidate = by_id.get(candidate_id)
                if candidate is None:
                    continue
                item = {
                    "window_id": window.id,
                    "candidate_id": candidate.id,
                    "text": candidate.text,
                }
                try:
                    raw, elapsed, resources = timed_call(
                        engine.review,
                        candidate.text,
                        context,
                    )
                    item.update(
                        {
                            "status": "ok",
                            "evidence": _json_value(raw),
                            "wall_time_seconds": elapsed,
                            "resources": resources,
                        }
                    )
                except Exception as exc:
                    item.update(
                        {
                            "status": "failed",
                            "error": str(exc),
                            "resources": resource_snapshot(),
                        }
                    )
                result["windows"].append(item)
        result["status"] = (
            "degraded"
            if any(item["status"] == "failed" for item in result["windows"])
            else "ok"
        )
        result["window_count"] = len(result["windows"])
        return result


def secondary_evidence_to_bundles(
    payload: dict[str, Any],
    windows: Sequence[ReviewWindow],
) -> list[DecisionEvidenceBundle]:
    """Normalize provider-specific diagnostics into one decision contract."""
    candidate_ids_by_window = {
        window.id: tuple(window.candidate_ids) for window in windows
    }
    bundles: list[DecisionEvidenceBundle] = []
    for source in ("forced_aligner", "sed", "semantic_review"):
        for item in payload.get(source, {}).get("windows", ()):
            window_id = item.get("window_id")
            candidate_ids = tuple(
                item.get("candidate_id")
                and (item["candidate_id"],)
                or candidate_ids_by_window.get(window_id, ())
            )
            evidence = item.get("evidence")
            if evidence is None and "words" in item:
                evidence = {
                    "words": item.get("words", ()),
                    "word_count": item.get("word_count", 0),
                }
            bundles.append(
                DecisionEvidenceBundle(
                    source=source,
                    status=str(item.get("status", "unknown")),
                    candidate_ids=candidate_ids,
                    window_id=window_id,
                    evidence=evidence
                    if isinstance(evidence, dict)
                    else {"value": evidence},
                    diagnostics={
                        key: item[key]
                        for key in ("error", "wall_time_seconds", "resources")
                        if key in item
                    },
                )
            )
    return bundles


__all__ = ["SecondaryEvidenceCollector", "secondary_evidence_to_bundles"]
