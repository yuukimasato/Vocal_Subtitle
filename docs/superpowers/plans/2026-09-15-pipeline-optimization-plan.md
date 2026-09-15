# Vocal Subtitle Pipeline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce pipeline coupling and failure ambiguity while preserving the existing `Pipeline.run()` API, subtitle timing behavior, cache correctness, and quality baselines.

**Architecture:** Introduce explicit run context, stage-level diagnostics, and one ASR execution plan incrementally behind the existing mixin API. Centralize final timeline validation and make cache identity content/model/version based. WebUI and legacy cleanup follow only after compatibility gates remain green.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, pytest via `uv run`, SQLite, diskcache, existing ASR/VAD/separation adapters.

**Spec:** [流水线优化方案-基于main代码分析-2026-09-15.md](../../流水线优化方案-基于main代码分析-2026-09-15.md)

## Global Constraints

- Protect the existing working tree before edits; do not reset, clean, checkout, or overwrite user changes.
- Preserve the public `Pipeline.run()` signature and current CLI/WebUI contracts.
- Run tests with `uv run pytest`; direct `pytest` is unavailable in the current environment.
- Do not change model defaults, profile semantics, subtitle formats, or fallback policy without a regression test.
- Every task ends with a focused test run and a small commit; never combine unrelated refactors.
- A delegated worker must inspect the current tree first, preserve other workers' edits, and report changed files and test output.

## Baseline and handoff protocol

Before Task 1, the coordinator records:

```bash
git status --short --branch
git diff --stat
python -m compileall -q vocal_subtitle llm_subtitle_optimizer
uv run pytest -q --disable-warnings
python scripts/check_import_boundaries.py
python scripts/check_module_size.py --baseline
python scripts/check_api_contract.py
python scripts/check_cli_contract.py
```

If the full suite cannot run, save the exact command, exit code, and first failure to `artifacts/baseline-2026-09-15.txt`; do not alter the baseline to make it pass. Save a patch of the pre-existing dirty tree outside git before implementation:

```bash
git diff > /tmp/vocal-subtitle-preexisting.diff
git ls-files --others --exclude-standard > /tmp/vocal-subtitle-untracked.txt
```

### Task 1: Add regression and measurement harness

**Files:** Create `tests/test_pipeline_baseline_contract.py` and `tests/test_quality_report_contract.py`.

- [ ] Add tests asserting `Pipeline.run` accepts its current parameters, default profile values remain stable, and final exported events have no adjacent overlap.
- [ ] Add a fixture capturing stage names, elapsed time, fallback category, and quality status without loading optional models.
- [ ] Run focused tests and record the full-suite baseline before changing production code.
- [ ] Commit `test: lock pipeline compatibility baseline`.

Completion: focused tests pass and the baseline artifact contains command, commit, dirty-tree note, and results.

### Task 2: Introduce explicit run context and stage protocol

**Files:** Create `vocal_subtitle/application/run_context.py`, `vocal_subtitle/application/stage_protocol.py`, `tests/test_run_context.py`; modify `run_lifecycle.py` and `pipeline.py`.

**Interfaces:** `RunContext`, `PipelineStage.execute(context) -> RunContext`, and `RunContext.add_diagnostic(stage, payload)`.

- [ ] Test context defaults, diagnostic merging, cancellation propagation, and isolation between two contexts.
- [ ] Implement dataclasses with nullable intermediates and existing `PipelineStats`.
- [ ] Adapt preflight/report setup to consume context while leaving public behavior unchanged.
- [ ] Run focused context and pipeline compatibility tests.
- [ ] Commit `refactor: introduce explicit pipeline run context`.

Completion: the new context is used by production code, tests prove no state leaks, and public API tests pass.

### Task 3: Extract lifecycle stages incrementally

**Files:** Create `vocal_subtitle/application/stages/{preflight_stage,audio_stage,asr_stage,mapping_stage}.py` and `tests/test_pipeline_stages.py`; modify `run_lifecycle.py`.

- [ ] Extract preflight and audio preparation, preserving status behavior.
- [ ] Extract ASR invocation with transcript, events, route, and diagnostics.
- [ ] Extract mapping/postprocess handoff without changing event ordering or timing.
- [ ] Keep `run()` under 200 lines during this task; do not remove mixins yet.
- [ ] Run stage tests plus pipeline, global ASR, and offline production tests.
- [ ] Commit `refactor: extract pipeline lifecycle stages`.

Completion: each stage is testable with synthetic context and compatibility results match event/status counts.

### Task 4: Unify ASR execution planning and window concurrency

**Files:** Create `vocal_subtitle/asr/execution_plan.py` and `tests/test_asr/test_execution_plan.py`; modify `application/asr_path.py`, `asr/boundary_reasr.py`, `asr/window_execution.py`, and `tests/test_window_execution.py`.

**Interfaces:** `ASRExecutionPlan`, `build_execution_plan(config, route)`, and the existing coordinator as the sole window executor.

- [ ] Test explicit engine, auto route, FunASR fallback, global failure, and segmented degraded transitions.
- [ ] Move route/fallback selection into the plan while preserving diagnostics fields.
- [ ] Replace boundary re-ASR's local thread pool with the coordinator and preserve ordering.
- [ ] Add active worker, timeout, cancellation, and degraded counts.
- [ ] Run ASR routing, boundary, global path, and window tests.
- [ ] Commit `refactor: unify asr execution planning and concurrency`.

Completion: every window ASR path uses one coordinator and state transitions are explicit and tested.

### Task 5: Make timeline arbitration single-owner

**Files:** Create `vocal_subtitle/mapping/timeline_result.py` and `tests/test_mapping/test_timeline_invariant.py`; modify time mapper, display timeline, acoustic validator, lifecycle, and final validator modules.

- [ ] Test equal-boundary neighbors, extension clamping, genuine overlap export, and revision trace preservation.
- [ ] Define immutable final events and validation diagnostics.
- [ ] Move repairs before finalization; make export validation read-only.
- [ ] Preserve `enforce_non_overlap` as a compatibility wrapper with repair counts.
- [ ] Run mapping, physical timeline, acoustic validator, and overlap export tests.
- [ ] Commit `refactor: centralize subtitle timeline arbitration`.

Completion: no post-export stage mutates timing and forbidden overlap count is zero.

### Task 6: Add structured errors and quality reports

**Files:** Create `vocal_subtitle/contracts/errors.py`, `vocal_subtitle/quality/stage_report.py`, and two focused test files; modify lifecycle, ASR failure translation, and run reporting.

- [ ] Test dependency, decode, ASR, timeline, and recoverable errors and serialized statuses.
- [ ] Replace core broad catches with domain-specific catches; unexpected defects retain tracebacks.
- [ ] Aggregate stage reports into existing `PipelineStats.quality_diagnostics` without breaking JSON.
- [ ] Run error, reporting, preflight, and WebUI lifecycle tests.
- [ ] Commit `feat: add structured pipeline errors and stage quality reports`.

Completion: expected failures have stable categories and every completed task has stage quality data.

### Task 7: Correct cache identity and audio reuse

**Files:** Create `vocal_subtitle/utils/cache_key.py`, `vocal_subtitle/utils/audio_buffer.py`, `tests/test_cache_identity.py`, and `tests/test_audio_buffer.py`; modify cache manager, file hasher, and run context/audio stage.

- [ ] Test same-content different-path hits, changed-content misses, model/version invalidation, and bounded cleanup.
- [ ] Implement keys from content hash, normalized config, engine/model identity, stage version, and schema version.
- [ ] Add read-only audio slices and migrate the extracted audio stage.
- [ ] Run cache, audio utility, pipeline cache, and offline production tests.
- [ ] Commit `refactor: make cache identity content and version based`.

Completion: cache behavior is content-correct, version-aware, observable, and production reuses the audio buffer.

### Task 8: Separate WebUI task orchestration

**Files:** Create `vocal_subtitle/webui/task_executor.py`, `vocal_subtitle/webui/task_events.py`, and `tests/test_webui_task_executor.py`; modify `pipeline_tasks.py` and `websocket.py`.

- [ ] Test event ordering for start, progress, degradation, failure, cancellation, and completion.
- [ ] Move background execution and event publication behind `TaskExecutor`, retaining payloads and serialized fields.
- [ ] Run WebUI, task lifecycle, websocket, and upload-learning tests.
- [ ] Commit `refactor: separate webui task execution and events`.

Completion: WebUI routes do not depend on Pipeline private state and event-order tests pass.

### Task 9: Remove legacy paths only after usage evidence

**Files:** Modify ASR path, chunk runner, lifecycle, and reporting modules; create `tests/test_legacy_path_usage.py`.

- [ ] Add counters and diagnostics for legacy entry points.
- [ ] Run compatibility and representative golden tests to establish callers.
- [ ] Remove only paths with zero required callers; retain adapters for externally used names.
- [ ] Run the full validation matrix and update module-size output.
- [ ] Commit `refactor: retire unused legacy pipeline paths`.

Completion: every removed path has zero callers, remaining callers have adapters, and all baseline gates pass.

## Final validation and handoff

- [ ] Run `uv run pytest -q --disable-warnings`.
- [ ] Run compileall, import-boundary, module-size, API-contract, CLI-contract, and relevant golden quality scripts.
- [ ] Compare test counts, quality metrics, timing, fallback counts, and cache behavior with the saved baseline.
- [ ] Review `git diff --stat`, `git status`, and every changed file for accidental edits to pre-existing work.
- [ ] Create a final integration commit only after all task commits and gates pass.
