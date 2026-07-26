# Physical Coverage and Tail Recovery Design

## Goal

Prevent offline subtitle tasks from silently ending before the last audible
speech, and ensure that subtitle timestamps remain inside their owning physical
audio ranges after segmentation, editing, and export.

The implementation targets the existing global-ASR path and keeps the current
ASR model, physical timeline, allocation, and subtitle formats. It adds a
coverage contract around them instead of inventing text when ASR evidence is
missing.

## Scope

This change includes:

1. Physical-bin coverage diagnostics after global word allocation.
2. Local ASR recovery for uncovered physical ranges, with overlap deduplication.
3. Physical-boundary preservation through strict segmentation and final export.
4. WebUI preservation of provenance fields when rebuilding subtitle events.
5. Final subtitle-count and degraded-status reporting.
6. Regression tests for the supplied two-speaker reading fixture.

This change does not change the ASR model, diarization model, LLM prompts, or
the manually corrected reference subtitle.

## Data Flow

```text
audio duration
    |
physical timeline -> speech bins -> global ASR words
                                  |
                                  v
                         word allocation
                                  |
             uncovered-bin audit + tail recovery
                                  |
                         event construction
                                  |
             strict segmentation / merge / validation
                                  |
                  final serialization and UI result
```

### Coverage audit

The audit compares physical speech bins with allocated word intervals. A bin
is considered covered when at least one accepted word overlaps it by a positive
duration. Short bins below the configured recovery threshold are reported but
do not trigger a second ASR pass.

The diagnostics must include:

- `physical_bin_count`
- `covered_physical_bin_count`
- `uncovered_physical_bin_count`
- `uncovered_physical_bins`
- `transcript_end`
- `last_physical_speech_end`
- `tail_gap_seconds`
- `recovery_attempted`
- `recovery_status`

The existing `missing_word_ids` metric remains, but it is explicitly limited to
recognized words and is not used as a full-audio coverage signal.

### Tail and gap recovery

For every uncovered physical range that is long enough to contain speech, run
the configured global ASR engine on a bounded audio window. The window includes
a small context collar before and after the uncovered range, then the returned
timestamps are shifted to the absolute timeline.

Recovered words are merged with the original transcript using the existing
global overlap-deduplication rules. Recovery is accepted only if the resulting
transcript remains valid against the audio duration and its words can be
allocated to a physical clip.

The first implementation supports uncovered ranges through the end of the
audio, which fixes the observed tail loss. Interior uncovered ranges are also
diagnosed and use the same recovery path when they exceed the threshold.

If recovery fails or produces no usable words:

- do not extend the previous subtitle;
- do not synthesize placeholder text;
- keep the valid events already produced;
- mark the result `degraded`;
- expose the uncovered intervals in diagnostics and WebUI status.

## Boundary Invariants

Every exported event must satisfy:

```text
0 <= start < end <= audio_duration
start >= physical_start, when physical_start exists
end <= physical_end, when physical_end exists
```

When an event is split, its physical envelope is copied only for the selected
physical spans or bin. When events are merged, their physical envelope becomes
the union of their owned spans, and the merge is allowed only when the existing
speaker, warning, and physical-owner constraints allow it.

The final validator remains authoritative and runs after every event-changing
stage. Export and WebUI rebuild paths use the same validation contract and
preserve physical provenance instead of rebuilding events with only display
fields.

Timestamp formatting is treated as a final quantization step. After SRT/ASS
millisecond quantization, the output sequence is checked again for invalid
intervals and overlaps beyond the configured tolerance.

## WebUI Contract

The API event payload retains all provenance needed to reconstruct a
`SubtitleEvent`, including physical spans, physical envelope, source word IDs,
bin identifiers, time source, warnings, and hard-boundary flags.

The result summary reports final serialized subtitle count rather than only the
pre-export global event count. It also exposes coverage status so a task that
ends early cannot look fully successful.

## Error Handling

Recovery errors are non-fatal in automatic mode. They are captured in global
diagnostics with a stable category and do not hide the original valid output.
The task is successful only when all required physical speech bins are covered
or the configured policy explicitly permits reported uncovered ranges.

The recovery implementation must avoid logging audio content or credentials.

## Testing and Acceptance

Add focused tests for:

1. A physical tail with no allocated words triggers recovery diagnostics.
2. Successful recovery adds words/events without duplicate overlap.
3. Failed recovery produces degraded status without fabricated text.
4. Split and merged events remain inside their physical envelopes.
5. WebUI event reconstruction preserves physical metadata.
6. Exported timestamps stay within audio duration and do not overlap after
   millisecond quantization.

For `test/中文朗读测试-双人.wav`, the acceptance checks are:

- final subtitle coverage reaches at least `157.90s`;
- no required physical tail bin remains uncovered;
- no event exceeds its physical envelope or audio duration;
- no output overlap remains after formatting;
- WebUI count matches the final serialized event count.
