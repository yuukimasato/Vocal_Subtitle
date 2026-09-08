from types import SimpleNamespace

from vocal_subtitle.asr.evidence import CandidateEvidence, DecisionEvidenceBundle, EvidenceWord
from vocal_subtitle.asr.evidence_decision import EvidenceDecisionEngine
from vocal_subtitle.asr.risk_scoring import RiskAssessment


def _candidate(identifier, text, start, end, *, source="segmented", words=()):
    return CandidateEvidence(
        id=identifier,
        source=source,
        text=text,
        start=start,
        end=end,
        words=tuple(words),
        confidence=0.2 if source == "segmented" else 0.9,
    )


def _high(identifier):
    return RiskAssessment(
        candidate_id=identifier,
        score=0.9,
        level="critical",
        evidence_codes=("high_cps",),
        factors={},
        review_required=True,
    )


def test_secondary_drop_requires_independent_sources_and_physical_blank():
    candidate = _candidate("main", "training phrase", 1.0, 1.3)
    timeline = SimpleNamespace(
        speech_evidence_spans=[SimpleNamespace(start=2.0, end=2.3)],
        duration=3.0,
        physical_clips=(),
    )
    bundles = [
        DecisionEvidenceBundle(
            source="sed",
            status="ok",
            candidate_ids=(candidate.id,),
            evidence={"label": "music", "score": 0.95},
        ),
        DecisionEvidenceBundle(
            source="semantic_review",
            status="ok",
            candidate_ids=(candidate.id,),
            evidence={"risk": "hallucination"},
        ),
    ]

    decision = EvidenceDecisionEngine().decide_bundle(
        [candidate], assessments=[_high(candidate.id)],
        physical_timeline=timeline, secondary_bundles=bundles,
    )[0]

    assert decision.decision == "drop"
    assert decision.physical_validation["status"] == "blank_allowed"
    assert decision.revision_trace[0]["sources"] == ["sed", "semantic_review"]


def test_secondary_split_requires_multiple_timed_parts():
    candidate = _candidate("main", "hello world", 1.0, 2.0)
    parts = [
        _candidate(
            "part-1", "hello", 1.0, 1.4, source="qwen",
            words=(EvidenceWord("w1", "hello", 1.0, 1.4, 0.9),),
        ),
        _candidate(
            "part-2", "world", 1.5, 2.0, source="qwen",
            words=(EvidenceWord("w2", "world", 1.5, 2.0, 0.9),),
        ),
    ]

    decisions = EvidenceDecisionEngine().decide_bundle(
        [candidate], review=parts,
        assessments=[_high(candidate.id)],
    )

    assert [item.decision for item in decisions] == ["split", "split"]
    assert [item.final_text for item in decisions] == ["hello", "world"]
