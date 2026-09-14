"""Pipeline 兼容性基线契约(2026-09-15 重构计划 Task 1)。

锁定重构期间不得改变的公共行为:
- ``Pipeline.run`` 当前参数与返回契约;
- default profile 的关键默认值;
- 最终导出事件相邻无重叠(最终时间轴不变量)。
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vocal_subtitle.config import ConfigLoader
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.pipeline_context import NoiseProfile, PipelineContext
from vocal_subtitle.utils.audio_utils import AudioUtils

RUN_PARAMS = (
    "input_path",
    "output_path",
    "output_format",
    "progress_callback",
    "skip_separation",
    "task_id",
    "session_dir",
    "feedback_reference",
)

RUN_RESULT_KEYS = {"subtitle_path", "stats", "events", "from_cache"}

# 重构期间必须保持稳定的 default profile 关键值。
STABLE_DEFAULTS = {
    ("asr", "engine"): "auto",
    ("asr", "model"): "large-v3",
    ("asr", "global_asr", "routing"): "segmented",
    ("asr", "global_asr", "alignment_enabled"): True,
    ("evidence_review", "enabled"): True,
    ("evidence_review", "context_reasr_enabled"): False,
    ("evidence_review", "global_alternative_enabled"): False,
    ("acoustic_validation", "enabled"): True,
    ("acoustic_validation", "timeline_arbitration"): False,
    ("acoustic_validation", "max_snap_distance"): 0.15,
    ("diarization", "word_split_on_turn"): False,
    ("llm_optimize", "enabled"): False,
}


def test_run_signature_keeps_current_parameters():
    signature = inspect.signature(Pipeline.run)

    positional = [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    assert tuple(positional) == RUN_PARAMS
    assert any(
        param.kind is param.VAR_KEYWORD
        for param in signature.parameters.values()
    ), "run() 必须保留 **overrides 兼容入口"


def test_default_profile_values_remain_stable():
    config = ConfigLoader().load_profile("default")

    for path, expected in STABLE_DEFAULTS.items():
        node = config
        for key in path:
            node = getattr(node, key)
        assert node == expected, f"default.{'.'.join(path)} 应保持 {expected!r}"


def _prepare_run(monkeypatch, tmp_path):
    config = ConfigLoader().load_profile("default")
    config.asr.engine = "faster-whisper"
    config.cache.enabled = False
    config.cache.full_pipeline_cache = False
    config.macro_chunking.enabled = False
    config.acoustic_validation.skeleton_mode = True
    config.acoustic_validation.enabled = False
    config.diarization.enabled = False
    config.merge_decision.llm_tier = "rule_only"
    config.llm_optimize.enabled = False

    pipeline = Pipeline(config)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        AudioUtils,
        "load_audio",
        staticmethod(lambda _: (np.zeros(16000, dtype=np.float32), 16000)),
    )
    pipeline._run_early_detection = lambda *args: (
        PipelineContext(
            audio_path=input_path,
            audio=np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
        ),
        [],
        None,
        NoiseProfile(0.0, 0.0, False),
    )
    pipeline._build_physical_shadow = lambda *args: SimpleNamespace(
        status="ok",
        diagnostics={},
        statistics={},
    )
    pipeline._post_process_events = lambda events, *args, **kwargs: events
    pipeline._finalize_events = lambda events, stats, duration: events
    pipeline._get_subtitle_builder = lambda: object()
    pipeline._export_subtitles_multi_format = (
        lambda builder, events, output_path, output_format, session_dir, label:
        {output_format: str(output_path)}
    )
    return pipeline, input_path


def test_run_returns_established_result_contract(monkeypatch, tmp_path):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path)

    def run_segmented(**kwargs):
        return [_event("segmented", 0.1, 0.5)], 1, None

    pipeline._process_skeleton_segmented = run_segmented
    pipeline._prepare_asr_route = lambda *a, **k: SimpleNamespace(
        requested_engine="faster-whisper",
        selected_engine="faster-whisper",
        final_engine="faster-whisper",
        detected_language=None,
        language_probability=0.0,
        route_version="asr-route-v1",
        quality_gate_version="asr-quality-v1",
        to_dict=lambda: {},
    )
    pipeline._run_offline_production_review = (
        lambda events, **kwargs: events
    )

    result = pipeline.run(
        input_path, output_path=tmp_path / "out.srt", skip_separation=True,
    )

    assert RUN_RESULT_KEYS.issubset(result.keys())
    assert result["from_cache"] is False
    assert [event.text for event in result["events"]] == ["segmented"]


def _event(text, start, end):
    return SubtitleEvent(
        index=1,
        start=start,
        end=end,
        text=text,
        physical_start=start,
        physical_end=end,
    )


def test_final_events_have_no_adjacent_overlap():
    from vocal_subtitle.mapping.final_validator import enforce_non_overlap

    events = [
        _event("one", 0.0, 1.0),
        _event("two", 0.8, 1.5),   # 与前一条重叠 → 必须被修复
        _event("three", 1.5, 2.0),
    ]

    repaired, diagnostics = enforce_non_overlap(events, source="baseline_contract")

    assert diagnostics["overlap_count"] >= 0
    ordered = sorted(repaired, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        assert current.start >= previous.start
        assert current.end <= current.start + 1e-9 or current.start >= previous.end - 1e-9
