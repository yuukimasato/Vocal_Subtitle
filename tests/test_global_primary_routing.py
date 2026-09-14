"""global-primary 实验路由的门禁、解析与回退行为。

- ``routing="global_primary"`` 时全局窗口作为主候选,但必须通过可行性门禁
  并走统一的 review/decision 出口;
- global 失败或覆盖不足时记录 ``global_primary_fallback_reason`` 并回退
  segmented 路径;
- 默认 ``routing="segmented"`` 行为保持不变。
"""

from types import SimpleNamespace

import numpy as np

from vocal_subtitle.application.global_primary import (
    evaluate_global_primary_suitability,
)
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.pipeline_context import NoiseProfile, PipelineContext
from vocal_subtitle.physical.ir import GlobalWord
from vocal_subtitle.physical.timeline import PhysicalTimeline
from vocal_subtitle.utils.audio_utils import AudioUtils


def _full_speech_timeline():
    timeline = PhysicalTimeline.from_duration(1.0)
    timeline.add_evidence(0.0, 1.0, "ffmpeg_skeleton")
    return timeline


def _event(text="global"):
    return SubtitleEvent(
        index=1,
        start=0.1,
        end=0.4,
        text=text,
        source_word_ids=["word-000000"],
        physical_start=0.1,
        physical_end=0.4,
    )


def _global_word(text="hello", start=0.1, end=0.9):
    return GlobalWord(
        id=f"gw:{text}:{start}",
        text=text,
        raw_start=start,
        raw_end=end,
        source_window_id="global",
        segment_id="seg:global",
    )


def _transcript(words=None):
    return SimpleNamespace(
        status="ok",
        words=words if words is not None else [_global_word()],
        segments=[SimpleNamespace(text="hello", start=0.1, end=0.9)],
    )


def _prepare_run(monkeypatch, tmp_path, routing):
    config = PipelineConfig()
    config.asr.engine = "faster-whisper"
    config.asr.global_asr.enabled = True
    config.asr.global_asr.routing = routing
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
        physical_timeline=_full_speech_timeline(),
    )
    pipeline._post_process_events = lambda events, *args, **kwargs: events
    pipeline._finalize_events = lambda events, stats, duration: events
    pipeline._get_subtitle_builder = lambda: object()
    pipeline._export_subtitles_multi_format = (
        lambda builder, events, output_path, output_format, session_dir, label:
        {output_format: str(output_path)}
    )
    return pipeline, input_path


# ---------------------------------------------------------------------------
# 路由解析
# ---------------------------------------------------------------------------


def test_default_routing_stays_segmented():
    config = PipelineConfig()
    pipeline = Pipeline(config)

    assert config.asr.global_asr.routing == "segmented"
    assert pipeline._resolve_asr_path() == "segmented"


def test_global_primary_routing_is_resolved_from_config():
    config = PipelineConfig()
    config.asr.global_asr.routing = "global_primary"
    pipeline = Pipeline(config)

    assert pipeline._resolve_asr_path() == "global_primary"


def test_legacy_global_routing_still_resolves():
    config = PipelineConfig()
    config.asr.global_asr.routing = "global"
    pipeline = Pipeline(config)

    assert pipeline._resolve_asr_path() == "global"


# ---------------------------------------------------------------------------
# 可行性门禁
# ---------------------------------------------------------------------------


def test_gate_passes_full_coverage_transcript():
    timeline = PhysicalTimeline.from_duration(1.0)
    timeline.add_evidence(0.1, 0.9, "ffmpeg_skeleton")

    result = evaluate_global_primary_suitability(
        _transcript(), audio_duration=1.0, physical_timeline=timeline
    )

    assert result.passed is True
    assert result.reason is None


def test_gate_rejects_empty_transcript():
    result = evaluate_global_primary_suitability(
        _transcript(words=[]), audio_duration=1.0
    )

    assert result.passed is False
    assert result.reason == "empty_transcript"


def test_gate_rejects_invalid_word_time_range():
    result = evaluate_global_primary_suitability(
        _transcript(words=[_global_word(start=0.1, end=5.0)]),
        audio_duration=1.0,
    )

    assert result.passed is False
    assert result.reason == "invalid_time_range"


def test_gate_rejects_insufficient_speech_coverage():
    timeline = PhysicalTimeline.from_duration(1.0)
    timeline.add_evidence(0.9, 1.0, "ffmpeg_skeleton")

    result = evaluate_global_primary_suitability(
        _transcript(words=[_global_word(start=0.0, end=0.05)]),
        audio_duration=1.0,
        physical_timeline=timeline,
    )

    assert result.passed is False
    assert result.reason == "insufficient_speech_coverage"


def test_gate_rejects_abnormal_text_density():
    timeline = PhysicalTimeline.from_duration(3.0)
    timeline.add_evidence(0.0, 3.0, "ffmpeg_skeleton")
    words = [_global_word(text="a", start=0.1, end=2.9)]

    result = evaluate_global_primary_suitability(
        _transcript(words=words),
        audio_duration=3.0,
        physical_timeline=timeline,
    )

    assert result.passed is False
    assert result.reason == "abnormal_text_density"


# ---------------------------------------------------------------------------
# 端到端路由:主候选选择与回退
# ---------------------------------------------------------------------------


def test_global_primary_success_selects_global_candidate(monkeypatch, tmp_path):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path, "global_primary")
    calls = {"global": 0, "segmented": 0}

    def run_legacy(**kwargs):
        calls["global"] += 1
        return [_event()], {}, _transcript()

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path_legacy = run_legacy
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(
        input_path, output_path=tmp_path / "out.srt", skip_separation=True
    )

    assert calls == {"global": 1, "segmented": 0}
    assert [event.text for event in result["events"]] == ["global"]
    assert result["stats"].asr_path == "global"


def test_global_primary_failure_falls_back_to_segmented(monkeypatch, tmp_path):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path, "global_primary")
    calls = {"global": 0, "segmented": 0}

    def run_legacy(**kwargs):
        calls["global"] += 1
        raise RuntimeError("model crashed")

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path_legacy = run_legacy
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(
        input_path, output_path=tmp_path / "out.srt", skip_separation=True
    )

    assert calls["global"] >= 1
    assert calls["segmented"] == 1
    assert [event.text for event in result["events"]] == ["segmented"]
    assert result["stats"].asr_path == "legacy_degraded"


def test_global_primary_insufficient_coverage_records_fallback_reason(
    monkeypatch, tmp_path
):
    pipeline, input_path = _prepare_run(monkeypatch, tmp_path, "global_primary")
    calls = {"global": 0, "segmented": 0}

    def run_legacy(**kwargs):
        calls["global"] += 1
        # 词时间集中在骨架语音区之外 → 覆盖不足。
        return (
            [_event()],
            {},
            _transcript(words=[_global_word(start=0.0, end=0.05)]),
        )

    def run_skeleton(**kwargs):
        calls["segmented"] += 1
        pipeline._global_review_timeline = PhysicalTimeline.from_duration(1.0)
        return [_event("segmented")], 1, None

    pipeline._run_global_transcription_path_legacy = run_legacy
    pipeline._process_skeleton_segmented = run_skeleton

    result = pipeline.run(
        input_path, output_path=tmp_path / "out.srt", skip_separation=True
    )

    assert calls["global"] >= 1
    assert calls["segmented"] == 1
    diagnostics = result["stats"].global_diagnostics
    assert (
        diagnostics["global_primary_fallback_reason"]
        == "insufficient_speech_coverage"
    )
    assert [event.text for event in result["events"]] == ["segmented"]
