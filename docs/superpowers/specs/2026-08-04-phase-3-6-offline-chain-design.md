# Phase 3-6 Offline Subtitle Chain Design

Date: 2026-08-04

## Scope

Implement the remaining Phase 3-6 work from the 2026-08-04 offline subtitle
plan. The work covers diagnostic provenance, physical-boundary trace,
evidence schema convergence, and capability/noise/feedback governance. It
does not replace the primary ASR engine, relax matching thresholds, or apply
noise/profile suggestions by default.

## Approach

Use incremental contract convergence over the existing diagnostics dictionaries,
`CandidateEvidence`, `EvidenceDecision`, `SubtitleEvent.revision_trace`, and
run reports. Add stable identifiers and stage records without introducing a
central provenance graph or breaking legacy report fields.

## Data Contracts

Each candidate and final event may carry a `trace_context` containing
`source_id`, `offset_id`, `window_id`, `candidate_id`, `physical_span_ids`,
`decision_id`, and `final_event_ids`. Global evidence explicitly declares a
`candidate_role` of `global_signal` or `global_alternative`; legacy
`source=global` remains readable. Alternatives record admission rules,
rejection reasons, `alternative_for`, selection, and decision IDs.

Miss attribution follows this chain:

```text
expected event -> physical span -> candidate -> decision -> projected event
  -> post-process/final cue -> legacy/strict match
```

Every unmatched event receives one primary stage. Missing trace is reported as
`unknown_with_evidence_gap` and counted; it never becomes an implicit pass.

## Stage Behavior

- Phase 3 adds deterministic short-response, trailing-speech, and overlap
  ownership fixtures. Fixtures are diagnostic only and do not change the
  reference denominator.
- Phase 4 records boundary decisions and merge participation while preserving
  physical owner/bin, speaker, hard-split, and maximum-duration protections.
- Phase 5 normalizes IDs and diagnostics across skeleton, chunk, global,
  review, and recovery paths. Shadow projection remains observational;
  authoritative projection alone changes final events.
- Phase 6 records independent capability maturity, noise recommendations, and
  feedback-profile load status. Noise recommendations remain shadow-only.
  Profile failures fall back to base configuration and expose the fallback in
  the run report. Legacy `review_policy` remains compatible while policy
  responsibilities converge.

## Validation

Tests cover each attribution stage, fixture category, boundary matrix,
schema round-trip, global selection statistics, shadow/authoritative behavior,
maturity evidence, noise non-application, and profile fallback/hash behavior.
Existing reports without new fields remain readable and report missing values
as unavailable.
