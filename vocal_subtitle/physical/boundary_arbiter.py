"""Structured decisions produced by physical boundary arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BoundaryCandidate:
    """One scored boundary candidate plus hard-constraint rejections."""

    label: str
    time: float
    score: float
    evidence_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    features: tuple[tuple[str, float], ...] = ()
    score_components: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "time": self.time,
            "score": self.score,
            "evidence_ids": list(self.evidence_ids),
            "rejection_reasons": list(self.rejection_reasons),
            "features": {key: value for key, value in self.features},
            "score_components": {
                key: value for key, value in self.score_components
            },
        }


@dataclass(frozen=True)
class BoundaryDecision:
    """Auditable acceptance or rejection of one word endpoint."""

    accepted: bool
    boundary_time: float
    boundary_type: str
    confidence: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    candidate_scores: tuple[tuple[str, float], ...] = ()
    rejected_candidates: tuple[str, ...] = ()
    candidate_diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "boundary_time": self.boundary_time,
            "boundary_type": self.boundary_type,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
            "candidate_scores": [
                {"label": label, "score": score}
                for label, score in self.candidate_scores
            ],
            "rejected_candidates": list(self.rejected_candidates),
            "candidate_diagnostics": [
                dict(item) for item in self.candidate_diagnostics
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoundaryDecision":
        return cls(
            accepted=bool(payload["accepted"]),
            boundary_time=float(payload["boundary_time"]),
            boundary_type=str(payload.get("boundary_type", "start")),
            confidence=float(payload.get("confidence", 0.0)),
            evidence_ids=tuple(payload.get("evidence_ids", ())),
            reason_codes=tuple(payload.get("reason_codes", ())),
            candidate_scores=tuple(
                (str(item["label"]), float(item["score"]))
                for item in payload.get("candidate_scores", ())
            ),
            rejected_candidates=tuple(payload.get("rejected_candidates", ())),
            candidate_diagnostics=tuple(
                dict(item)
                for item in payload.get("candidate_diagnostics", ())
                if isinstance(item, Mapping)
            ),
        )


class BoundaryArbiter:
    """Apply hard constraints before choosing the highest scored candidate."""

    def decide(
        self,
        boundary_type: str,
        candidates: Sequence[BoundaryCandidate],
        *,
        fallback_time: float,
        accepted_reason: str,
        missing_reason: str,
    ) -> BoundaryDecision:
        if boundary_type not in {"start", "end"}:
            raise ValueError("boundary_type must be 'start' or 'end'")

        legal = [item for item in candidates if not item.rejection_reasons]
        rejected = tuple(
            f"{item.label}:{reason}"
            for item in candidates
            for reason in item.rejection_reasons
        )
        scores = tuple((item.label, item.score) for item in candidates)
        diagnostics = tuple(item.to_dict() for item in candidates)
        if not legal:
            return BoundaryDecision(
                accepted=False,
                boundary_time=fallback_time,
                boundary_type=boundary_type,
                confidence=0.0,
                reason_codes=(missing_reason,),
                candidate_scores=scores,
                rejected_candidates=rejected,
                candidate_diagnostics=diagnostics,
            )

        selected = max(legal, key=lambda item: (item.score, -item.time, item.label))
        return BoundaryDecision(
            accepted=True,
            boundary_time=selected.time,
            boundary_type=boundary_type,
            confidence=max(0.0, min(1.0, selected.score)),
            evidence_ids=selected.evidence_ids,
            reason_codes=(accepted_reason,),
            candidate_scores=scores,
            rejected_candidates=rejected,
            candidate_diagnostics=diagnostics,
        )
