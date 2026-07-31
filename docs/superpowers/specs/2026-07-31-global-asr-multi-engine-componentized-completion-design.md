# Global ASR Multi-Engine Completion Design

Date: 2026-07-31
Status: approved for implementation

## Scope

This completion pass closes the remaining implementation and validation work
for the componentized global ASR review path. Existing uncommitted changes are
treated as the current implementation baseline and are preserved.

## Design

The application global path remains the owner of route selection and legacy
physical allocation, while `GlobalTranscriber` owns bounded window execution.
For audio longer than `asr.global_asr.max_window_duration`, the runner passes
the physical timeline to `GlobalTranscriber`, which splits oversized clips,
projects each result to absolute time, deduplicates overlap, and returns
diagnostics. The segmented path remains the primary subtitle candidate and
global output remains evidence-only until `EvidenceDecision` resolves it.

Regression coverage will exercise both the component directly and the
application runner, including oversized physical clips, absolute timestamps,
overlap deduplication, failed windows, and global-to-segmented fallback.

Optional real runtimes are validated independently: Qwen3-ASR, Qwen3
ForcedAligner, Transformers SED, and the configured LLM client. Model paths
are explicit and downloads are opt-in; unavailable models produce structured
diagnostics without blocking the base pipeline.

The dependency conflict is resolved by isolating legacy `spleeter` from the
Python 3.11-compatible all-model extra. The lock is regenerated and checked
for Python 3.11 and 3.12 resolution; Spleeter remains installable through its
own explicitly incompatible extra.

The quality calibration command consumes `test/quality_manifest.yaml` and
existing benchmark outputs, records the selected thresholds and metrics, and
fails the quality gate when hallucination retention, false deletion, physical
overflow, unresolved rate, or P95 latency exceed configured limits. With no
gold subtitle annotations for a case, it reports the case as uncalibrated
rather than treating it as a pass.

## Deliverables

- long-audio GlobalTranscriber application and regression tests;
- real-runtime smoke/availability verification report;
- resolved `uv.lock` and documented extra matrix;
- reproducible risk-threshold and quality-gate calibration report;
- complete test and contract-check results;
- updated implementation plan with remaining external work explicitly listed.
