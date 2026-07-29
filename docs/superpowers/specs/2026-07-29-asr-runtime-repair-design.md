# ASR Runtime Repair Design

## Goal

Restore successful WebUI subtitle generation for engines that can recognize audio, make whisper.cpp runnable in the approved local environment, and preserve explicit task failures when an engine dependency or model is unavailable.

## Scope

- Fix `Pipeline._finalize_events()` so it can read the instance subtitle configuration.
- Keep finalization as the single source for final event splitting, `subtitle_count`, diagnostics, and exported subtitle files.
- Install whisper.cpp locally and configure the existing `whisper_cpp_bin` / `whisper_cpp_model_path` settings without changing ASR text or merge policies.
- Add focused regression coverage for finalization and runtime dependency propagation.
- Verify through WebUI browser runs before expanding to the complete audio matrix.

## Design

### Pipeline finalization

Convert `_finalize_events()` from a static method to an instance method. It will use `self.config.subtitle` to build `FinalizeConfig`, call `finalize_subtitle_events()`, update `stats.subtitle_count` and `stats.quality_diagnostics`, then return the finalized events. Existing call sites remain unchanged because they already invoke it through a `Pipeline` instance.

No changes are made to the finalization rules themselves. The fix only restores the configuration dependency that the current implementation already intended to use.

### whisper.cpp runtime

Use a project-local whisper.cpp binary and model so the application does not depend on an undocumented system installation. Configure the selected binary and model through the existing ASR configuration fields or environment variables. `WhisperCppEngine.load_model()` remains the single preflight boundary:

- missing executable raises `ASRDependencyError`;
- missing ggml/gguf model raises `ASRModelError`;
- valid executable and model proceed to transcription.

The WebUI task must expose these exceptions as `failed`, with the actionable error text. No empty successful task is allowed for a detected speech input.

### Testing

- Unit test that a pipeline finalization call uses subtitle limits from the instance configuration and updates statistics.
- Preserve and run existing whisper.cpp executable/model checks.
- Run focused pipeline and WebUI tests for invalid ASR results and error propagation.
- Start the WebUI through the browser and test a short Chinese audio with faster-whisper, whisper.cpp, and FunASR.
- Confirm successful runs expose non-empty events and export paths; confirm unavailable dependencies show failed status.

## Acceptance criteria

1. The representative faster-whisper and FunASR browser tasks complete without the `name 'self' is not defined` error.
2. whisper.cpp either completes with non-empty output using the installed binary/model or fails with a specific dependency/model error.
3. Finalized events, statistics, WebUI preview, and exported SRT/VTT/ASS use the same finalized event list.
4. Existing unrelated user changes remain untouched.

## Risks and rollback

The binary/model download is external and can be large; it is kept outside source files and can be removed independently. Code rollback is limited to the targeted pipeline method and its tests. No subtitle thresholds or recognition behavior are changed.
