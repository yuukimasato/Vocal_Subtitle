# Global ASR Evidence Gap Completion Design

Date: 2026-08-01
Status: approved for implementation

## Goal

Close the implementation gaps found during review of the componentized global
ASR evidence path without enabling real optional models or changing the
default shadow rollout behavior.

## Changes

1. Global ASR failures in both `global` and `auto` routes record diagnostics and
   continue through the segmented subtitle path.
2. Global evidence is adapted from the global transcript/word IR when
   available. Legacy event adaptation preserves missing confidence and the
   original time source; it must not fabricate `1.0`.
3. The physical timeline produced during early detection is retained for the
   segmented evidence review path. Decision validation also checks word times
   against the audio and physical evidence bounds.
4. Risk scoring detects normalized repeated phrases separated by the configured
   gap. Repetition only raises risk and never independently drops a candidate.

## Compatibility and Safety

`shadow_mode=true` continues returning baseline events while recording
decisions. Non-shadow output continues to be produced only through
`EvidenceDecision`. Optional Qwen, ForcedAligner, SED and semantic adapters
remain opt-in and unavailable adapters do not block subtitles.

## Verification

Add focused tests for global failure fallback, missing confidence/time-source
preservation, segmented physical timeline injection, word-level physical
validation, and repeated phrase scoring. Run the full test suite and existing
component/import/public-compatibility checks.
