# ASR Language Context and LLM Configuration Fix

## Problem

The `181` run used `asr.language=null` with `language_mode=single`. In
skeleton-segmented mode, each 0.2-2 second skeleton segment is passed to the
ASR pipeline independently. The current ASR path detects language from that
short segment, so Chinese speech can be classified as English or Japanese and
produce hallucinated text such as `Thank you for watching` and
`字幕志愿者`.

The WebUI also exposes one LLM configuration, while the task currently stores
the key under `llm_optimize.api_key` but can omit the URL and leaves
`merge_decision` credentials empty. The task may therefore run without either
LLM text optimization or cloud merge decisions even though the UI shows a
configured key.

## Goals

1. Detect automatic language once from the complete audio for offline runs.
2. Reuse that task-level language for all short ASR segments in `single` mode.
3. Preserve explicit `asr.language` as an authoritative language override.
4. Allow per-segment language fallback only when `language_mode=mixed` is
   explicitly selected.
5. Make WebUI LLM URL/key overrides consistent with the configuration shown in
   the UI and make the task result clearly distinguish disabled, unavailable,
   and completed LLM optimization.
6. Keep speaker diarization independent from ASR text generation.

## Non-goals

- Do not add a hard-coded blacklist for phrases such as `Thank you for
  watching`.
- Do not change speaker clustering algorithms or relabel speakers as part of
  this fix.
- Do not automatically enable paid LLM calls merely because a key exists.
- Do not change the default language policy for genuinely mixed-language
  projects.

## Design

### 1. Task-level language context

Add a small pipeline-level language preparation step with this precedence:

1. `asr.language` is set: use it directly and skip detection.
2. `language_mode=single` and language is unset: detect once from the
   complete audio, store the result on the current `Pipeline`, and reuse it
   for every segment.
3. `language_mode=mixed`: use the complete-audio result as the default hint,
   but permit the existing per-segment confidence fallback to re-detect a
   segment when both language probability and recognition confidence satisfy
   the configured thresholds.

The segment ASR method will accept the prepared language context rather than
implicitly detecting language from its local audio. Skeleton, normal offline,
macro-chunk, and streaming paths must keep their current mode-specific
behavior; the task-level context applies to the offline skeleton path that
caused the observed failure.

The complete-audio detection must happen before skeleton iteration. A missing
or low-confidence detection is still recorded in diagnostics, but `single`
mode must not silently replace it with a new detection from a shorter segment.
Users can avoid auto detection entirely by setting `language: zh` in the
profile or UI.

### 2. LLM configuration flow

Keep `llm_optimize` and `merge_decision` as separate runtime controls, but
make the WebUI submit one consistent URL/key pair to the selected LLM
features. The UI will read the unified saved settings first, retain backward
compatibility with the legacy local-storage keys, and include the configured
base URL even when the user did not manually edit the prefilled provider URL.

The pipeline will continue to require the explicit `llm_optimize.enabled`
toggle before text optimization. Missing credentials or an unavailable API
will remain non-fatal and preserve ASR text. The existing rule fallback for
merge decisions remains active. Logs and result metadata will continue to
identify whether LLM optimization was skipped, degraded, or completed.

### 3. Regression and observability

Add focused tests for:

- explicit language bypassing language detection;
- automatic single-language mode detecting from full audio only once;
- mixed mode retaining the opt-in fallback behavior;
- skeleton processing passing the prepared language into each ASR call;
- WebUI override collection including the configured LLM URL/key;
- the `181`-style short English misclassification not being introduced by
  task-level language preparation.

The regression assertions will not require a live LLM API. They will inspect
the effective configuration and use mocked ASR/LLM boundaries. The real `181`
run will be performed separately after unit tests, with the expected result:

- no `Thank you for watching`, `字幕志愿者`, or similar phantom entries;
- all output events retaining a valid speaker assignment when diarization
  provides one;
- `llm_subtitle_path` present only when the optimization toggle is enabled and
  the optimizer actually completes.

## Error handling

- Language detection errors remain explicit in logs and diagnostics.
- A failed LLM call never replaces valid ASR text with an unvalidated result.
- API keys are never printed in logs or user-facing diagnostics.
- Existing unrelated dirty-worktree changes must remain untouched.

## Acceptance criteria

1. A default-profile `181` run does not perform language detection separately
   on each skeleton segment.
2. The `10.73-11.43s` segment is recognized using the task language context
   rather than being locked to English from its 0.7 second local audio.
3. The generated clean subtitle no longer contains the phantom entries seen
   in `test/181/181项目识别.ass`.
4. Explicit `zh` and explicit `mixed` behavior are covered by tests.
5. LLM URL/key state submitted by the WebUI is visible in the effective task
   configuration without enabling LLM calls implicitly.

