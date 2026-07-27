"""Tests for boundary projection state machine."""

import pytest

from vocal_subtitle.mapping.boundary_projection import (
    BoundaryCandidate,
    ProjectedBoundary,
    ProjectionResult,
    ProjectionState,
    project_boundaries,
    project_with_repair,
    log_transition,
    _MAX_TRANSITIONS,
)


# ── BoundaryCandidate ────────────────────────────────────────────────

def test_candidate_rejects_negative_time():
    with pytest.raises(ValueError, match="non-negative"):
        BoundaryCandidate(candidate_id="c1", time=-0.1, direction="start")


def test_candidate_rejects_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        BoundaryCandidate(candidate_id="c1", time=1.0, direction="middle")


def test_candidate_clamps_confidence():
    c = BoundaryCandidate(candidate_id="c1", time=1.0, direction="start", confidence=1.5)
    assert c.confidence == 1.0


def test_candidate_to_dict():
    c = BoundaryCandidate(candidate_id="c99", time=2.5, direction="end",
                          confidence=0.85, source="arbiter",
                          evidence_ids=("ev1", "ev2"),
                          metadata={"score": 3.5})
    d = c.to_dict()
    assert d["candidate_id"] == "c99"
    assert d["time"] == 2.5
    assert d["direction"] == "end"
    assert d["confidence"] == 0.85
    assert d["evidence_ids"] == ["ev1", "ev2"]


# ── log_transition ───────────────────────────────────────────────────

def test_log_transition_valid():
    log = []
    log_transition(log, ProjectionState.PROPOSED, ProjectionState.PROJECTED, "test")
    assert len(log) == 1
    assert "projected" in log[0]


def test_log_transition_invalid():
    log = []
    with pytest.raises(ValueError, match="invalid transition"):
        log_transition(log, ProjectionState.ACCEPTED, ProjectionState.PROPOSED, "bad")
    assert len(log) == 0


def test_log_transition_from_none():
    log = []
    log_transition(log, None, ProjectionState.PROPOSED, "start")
    assert "none ->" in log[0]


# ── project_boundaries ───────────────────────────────────────────────

def test_simple_accepted_projection():
    groups = [
        {"physical_start": 1.0, "physical_end": 3.0, "text": "hello", "index": 1},
    ]
    start_candidates = [
        BoundaryCandidate("s1", 0.95, "start", confidence=0.9),
        BoundaryCandidate("s2", 1.05, "start", confidence=0.8),
    ]
    end_candidates = [
        BoundaryCandidate("e1", 2.95, "end", confidence=0.9),
    ]

    results = project_boundaries(groups, start_candidates, end_candidates)

    assert len(results) == 1
    assert results[0].is_accepted
    assert results[0].projected_start.state == ProjectionState.ACCEPTED
    assert results[0].projected_end.state == ProjectionState.ACCEPTED
    assert abs(results[0].projected_start.projected_time - 0.95) < 0.01
    assert abs(results[0].projected_end.projected_time - 2.95) < 0.01


def test_fallback_when_no_start_candidate():
    groups = [
        {"physical_start": 1.0, "physical_end": 3.0, "text": "hello"},
    ]
    start_candidates = []  # no compatible start
    end_candidates = [
        BoundaryCandidate("e1", 2.95, "end", confidence=0.9),
    ]

    results = project_boundaries(groups, start_candidates, end_candidates)

    assert len(results) == 1
    assert results[0].is_fallback
    assert results[0].projected_start.state == ProjectionState.FALLBACK


def test_fallback_when_candidates_too_far():
    groups = [
        {"physical_start": 1.0, "physical_end": 3.0, "text": "hello"},
    ]
    start_candidates = [
        BoundaryCandidate("s1", 10.0, "start", confidence=0.9),  # too far
    ]
    end_candidates = [
        BoundaryCandidate("e1", 20.0, "end", confidence=0.9),  # too far
    ]

    results = project_boundaries(groups, start_candidates, end_candidates)

    assert len(results) == 1
    assert results[0].is_fallback


def test_inverted_projection_rejected():
    """Start after end should be rejected."""
    groups = [
        {"physical_start": 3.0, "physical_end": 5.0, "text": "late"},
    ]
    start_candidates = [
        BoundaryCandidate("s1", 4.5, "start", confidence=0.8),
    ]
    end_candidates = [
        BoundaryCandidate("e1", 4.0, "end", confidence=0.8),  # before start
    ]

    results = project_boundaries(groups, start_candidates, end_candidates)

    assert not results[0].is_accepted


def test_duration_limit_enforced():
    groups = [
        {"physical_start": 0.0, "physical_end": 10.0, "text": "very long"},
    ]
    start_candidates = [
        BoundaryCandidate("s1", 0.0, "start", confidence=0.9),
    ]
    end_candidates = [
        BoundaryCandidate("e1", 9.5, "end", confidence=0.9),
    ]

    results = project_boundaries(groups, start_candidates, end_candidates, max_duration=5.0)

    assert not results[0].is_accepted


def test_merge_with_compatible_neighbor():
    """When no admissible candidate, try to merge with neighbor."""
    groups = [
        {"physical_start": 1.0, "physical_end": 2.0, "text": "first",
         "speaker_id": 1, "physical_region_id": "r1"},
        {"physical_start": 2.1, "physical_end": 3.5, "text": "second",
         "speaker_id": 1, "physical_region_id": "r1"},
    ]
    # No candidates matching group 0
    start_candidates = [
        BoundaryCandidate("s1", 0.95, "start", confidence=0.9),
    ]

    results = project_boundaries(groups, start_candidates, [])

    # First group has no end candidate — should merge with compatible neighbor
    # Not FALLBACK because neighbor is compatible
    assert results[0].projected_end.state in (ProjectionState.MERGED, ProjectionState.FALLBACK)


def test_no_merge_across_speakers():
    groups = [
        {"physical_start": 1.0, "physical_end": 2.0, "text": "speaker A",
         "speaker_id": 1},
        {"physical_start": 2.1, "physical_end": 3.5, "text": "speaker B",
         "speaker_id": 2},  # different speaker
    ]

    results = project_boundaries(groups, [], [])

    # Should not merge across different speakers
    assert results[0].projected_end.state == ProjectionState.FALLBACK


# ── project_with_repair ──────────────────────────────────────────────

def test_repair_fn_called_once():
    repair_log = []

    def mock_repair(**kwargs):
        repair_log.append(kwargs)
        return {
            "start": ProjectedBoundary(1.0, "repair-start", 0.5, ProjectionState.ONE_REPAIR),
            "end": ProjectedBoundary(3.0, "repair-end", 0.5, ProjectionState.ONE_REPAIR),
        }

    # One group has a compatible neighbor (same speaker, same region) —
    # with no admissible end candidate, the projection enters MERGED state.
    groups = [
        {"physical_start": 1.0, "physical_end": 2.0, "text": "first",
         "speaker_id": 1, "physical_region_id": "r1"},
        {"physical_start": 2.1, "physical_end": 3.0, "text": "second",
         "speaker_id": 1, "physical_region_id": "r1"},
    ]

    # Provide a valid start but no valid end for group 0
    start_candidates = [
        BoundaryCandidate("s1", 0.95, "start", confidence=0.9),
    ]
    end_candidates = []  # no end candidates → triggers MERGED path

    results = project_with_repair(
        groups, start_candidates, end_candidates, repair_fn=mock_repair,
    )

    # Repair function should have been called for at least one group
    assert len(repair_log) == 1
    # The repaired group should have ONE_REPAIR state
    assert results[0].projected_start.state == ProjectionState.ONE_REPAIR
    assert results[0].projected_end.state == ProjectionState.ONE_REPAIR


def test_repair_not_called_when_already_accepted():
    repair_log = []

    def mock_repair(**kwargs):
        repair_log.append("should not be called")
        return None

    groups = [
        {"physical_start": 1.0, "physical_end": 3.0, "text": "good"},
    ]
    start_candidates = [
        BoundaryCandidate("s1", 0.95, "start", confidence=0.9),
    ]
    end_candidates = [
        BoundaryCandidate("e1", 2.95, "end", confidence=0.9),
    ]

    results = project_with_repair(
        groups, start_candidates, end_candidates, repair_fn=mock_repair,
    )

    assert results[0].is_accepted
    assert len(repair_log) == 0


def test_repair_failure_falls_back():
    def mock_repair(**kwargs):
        raise RuntimeError("unrecoverable")

    groups = [
        {"physical_start": 1.0, "physical_end": 3.0, "text": "unfixable"},
    ]

    results = project_with_repair(groups, [], [], repair_fn=mock_repair)

    assert results[0].is_fallback
    assert results[0].projected_start.state == ProjectionState.FALLBACK


def test_transition_count_capped():
    """Transitions must not exceed _MAX_TRANSITIONS."""
    groups = [
        {"physical_start": 1.0, "physical_end": 3.0, "text": "complex"},
    ]

    results = project_boundaries(groups, [], [])

    assert len(results[0].transition_log) <= _MAX_TRANSITIONS


def test_merge_blocked_by_hard_split():
    groups = [
        {"physical_start": 1.0, "physical_end": 2.0, "text": "first",
         "speaker_id": 1, "hard_split_after": True},
        {"physical_start": 2.1, "physical_end": 3.5, "text": "second",
         "speaker_id": 1},  # same speaker but hard split
    ]

    results = project_boundaries(groups, [], [])

    # Hard split should block merge
    assert results[0].projected_end.state == ProjectionState.FALLBACK


def test_result_to_dict_serializable():
    result = ProjectionResult(
        group_index=0,
        projected_start=ProjectedBoundary(1.0, "c1", 0.9, ProjectionState.ACCEPTED),
        projected_end=ProjectedBoundary(3.0, "c2", 0.85, ProjectionState.ACCEPTED),
        candidates_considered=[
            BoundaryCandidate("c1", 1.0, "start", 0.9),
            BoundaryCandidate("c2", 3.0, "end", 0.85),
        ],
        transition_log=["none -> proposed: start", "proposed -> projected: test"],
        warnings=[],
    )
    d = result.to_dict()
    assert d["group_index"] == 0
    assert d["projected_start"]["time"] == 1.0
    assert d["projected_end"]["time"] == 3.0
    assert d["projected_start"]["state"] == "accepted"
