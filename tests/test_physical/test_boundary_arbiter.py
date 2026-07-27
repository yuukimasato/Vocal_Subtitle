from vocal_subtitle.physical.boundary_arbiter import (
    BoundaryArbiter,
    BoundaryCandidate,
)


def test_illegal_high_score_candidate_never_wins():
    decision = BoundaryArbiter().decide(
        "start",
        [
            BoundaryCandidate(
                "outside_clip",
                0.1,
                0.99,
                rejection_reasons=("outside_physical_clip",),
            ),
            BoundaryCandidate("asr_start", 0.3, 0.55, ("e1",)),
        ],
        fallback_time=0.25,
        accepted_reason="aligned_start",
        missing_reason="no_legal_start_candidate",
    )

    assert decision.accepted is True
    assert decision.boundary_time == 0.3
    assert decision.evidence_ids == ("e1",)
    assert "outside_clip:outside_physical_clip" in decision.rejected_candidates


def test_no_legal_candidate_has_structured_degradation_reason():
    decision = BoundaryArbiter().decide(
        "end",
        [
            BoundaryCandidate(
                "after_next_word",
                1.2,
                0.9,
                rejection_reasons=("breaks_monotonicity",),
            )
        ],
        fallback_time=1.0,
        accepted_reason="aligned_end",
        missing_reason="no_legal_end_candidate",
    )

    assert decision.accepted is False
    assert decision.boundary_time == 1.0
    assert decision.reason_codes == ("no_legal_end_candidate",)
    assert decision.rejected_candidates == (
        "after_next_word:breaks_monotonicity",
    )


def test_decision_serializes_candidate_features_and_components():
    candidate = BoundaryCandidate(
        "rms_start",
        0.4,
        0.8,
        evidence_ids=("rms-1",),
        features=(("rms_gradient", 0.25), ("vad_probability", 0.9)),
        score_components=(("rms", 0.8), ("vad", 0.9)),
    )
    decision = BoundaryArbiter().decide(
        "start",
        [candidate],
        fallback_time=0.5,
        accepted_reason="aligned_start",
        missing_reason="no_legal_start_candidate",
    )

    payload = decision.to_dict()
    detail = payload["candidate_diagnostics"][0]
    assert detail["features"]["rms_gradient"] == 0.25
    assert detail["score_components"]["vad"] == 0.9
    restored = decision.from_dict(payload)
    assert restored.candidate_diagnostics[0]["label"] == "rms_start"
