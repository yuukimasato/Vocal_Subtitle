"""[层1] 说话人身份主干（early_turns / word_split_on_turn）测试。

覆盖设计文档 §7 P1/P2：
- 全局 turns 前置（run_early_global_pass 的门控/成功/不可用）
- "得了吧"核心场景：A 长话轮 + B 短反应语、中间无静音间隙 →
  词级后切分，B 的词被切出并标 B（合成 turns，mock 全局 pass）
- early_turns=false 回归等价、全局失败回退后处理 fusion
- tiny-fragment 同/异说话人合并行为、单说话人短路
- reconcile_regions 接线（骨架区间 × turns 求交 + 同说话人合并）
"""

from dataclasses import dataclass, replace

import numpy as np

from vocal_subtitle.asr.base import WordTimestamp
from vocal_subtitle.config import PipelineConfig
from vocal_subtitle.diarization import speaker_fusion
from vocal_subtitle.diarization.base import DiarizationResult, SpeakerTurn
from vocal_subtitle.diarization.early_turns import (
    EarlyTurnsState,
    assign_event_speakers,
    dominant_speaker_at,
    run_early_global_pass,
    spans_from_skeleton,
)
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.pipeline import Pipeline
from vocal_subtitle.pipeline_context import PipelineContext
from vocal_subtitle.vad.base import SpeechSegment


def _ok_state(turns, *, model="pyannote/community-1"):
    speakers = {turn.speaker_id for turn in turns}
    return EarlyTurnsState(
        turns=list(turns),
        speaker_count=len(speakers),
        backend="pyannote",
        status="ok",
        model_ref=model,
        attempted=True,
        single_speaker=len(speakers) <= 1,
        diagnostics={"global_status": "ok", "global_turn_count": len(turns)},
    )


def _postprocess_config(**diar_overrides):
    config = PipelineConfig()
    config.diarization.enabled = True
    config.diarization.early_turns = True
    config.diarization.word_split_on_turn = True
    config.merge_decision.llm_tier = "rule_only"
    config.acoustic_validation.enabled = False
    config.speaker_role.enabled = False
    for key, value in diar_overrides.items():
        setattr(config.diarization, key, value)
    return config


def _zh_pipeline(config=None):
    """带已解析语言（中文标签）的管线实例。"""
    pipeline = Pipeline(config or PipelineConfig())
    pipeline._resolved_language = "zh"
    return pipeline


def _run_postprocess(pipeline, events, config):
    from vocal_subtitle.application.pipeline_result import PipelineStats

    pipeline.config = config
    stats = PipelineStats(input_path=None, duration_seconds=4.0)
    events = pipeline._post_process_events(
        events,
        vocals_path=None,
        audio=np.ones(4 * 16000, dtype=np.float32),
        sample_rate=16000,
        stats=stats,
    )
    return events, stats


# ---------------------------------------------------------------------------
# "得了吧"核心场景（P2 词级切分）
# ---------------------------------------------------------------------------


def test_reaction_words_are_split_out_and_labeled_second_speaker():
    """A 长话轮 + B 短反应语、无静音间隙 → B 的词被切出并标 B。"""
    pipeline = _zh_pipeline()
    pipeline._early_turns_state = _ok_state(
        [
            SpeakerTurn(0.0, 2.0, 0),
            SpeakerTurn(2.0, 4.0, 1),
        ]
    )
    event = SubtitleEvent(
        1,
        0.0,
        4.0,
        "你怎么又迟到了得了吧",
        words=[
            WordTimestamp("你怎么又迟到了", 0.2, 1.8),
            WordTimestamp("得了吧", 2.1, 3.6),
        ],
    )

    events, stats = _run_postprocess(pipeline, [event], _postprocess_config())

    assert [e.text for e in events] == ["你怎么又迟到了", "得了吧"]
    assert [e.speaker_id for e in events] == [0, 1]
    assert [e.speaker_label for e in events] == ["说话人A", "说话人B"]
    assert [e.speaker_source for e in events] == ["global", "global"]
    # 切分点吸附到词间隙：两段边界来自词级时间戳
    assert events[0].start == 0.2 and events[0].end == 1.8
    assert events[1].start == 2.1 and events[1].end == 3.6
    # stats 契约字段由早前 pass 的结果填充
    assert stats.speaker_count == 2
    assert stats.diarization_backend == "pyannote"
    assert stats.diarization_status == "ok"
    assert stats.local_speaker_split_count == 1
    assert stats.unknown_speaker_count == 0
    assert stats.quality_diagnostics["global_turn_count"] == 2
    assert stats.quality_diagnostics["early_turns_status"] == "ok"
    assert stats.quality_diagnostics["event_clustering"] == "retired_by_early_turns"


def test_overlapped_turn_region_is_marked_not_forced(monkeypatch):
    """两人重叠区的 turn 标 overlapped → 事件诚实标注，不强判归属。"""
    turns = [
        SpeakerTurn(0.0, 2.0, 0),
        replace(SpeakerTurn(1.5, 3.0, 1), overlapped=True),
    ]
    event = SubtitleEvent(
        1,
        0.0,
        3.0,
        "甲乙",
        words=[WordTimestamp("甲", 0.2, 1.0), WordTimestamp("乙", 1.8, 2.8)],
    )

    events, diag = assign_event_speakers(
        [event],
        turns,
        word_split=True,
        language="zh",
    )

    overlapped = [e for e in events if e.genuine_overlap]
    assert overlapped and all(e.speaker_id == 1 for e in overlapped)
    assert diag["overlapped_count"] >= 1


# ---------------------------------------------------------------------------
# run_early_global_pass 门控与回退
# ---------------------------------------------------------------------------


def test_early_pass_disabled_by_config_never_runs_global_pass(monkeypatch):
    config = PipelineConfig()
    assert config.diarization.early_turns is False

    def _boom(*args, **kwargs):
        raise AssertionError("global pass must not run when early_turns=false")

    monkeypatch.setattr(speaker_fusion, "_run_global_pass", _boom)

    state = run_early_global_pass(np.zeros(1600, dtype=np.float32), 16000, config)

    assert state.status == "disabled"
    assert not state.active
    assert state.turns == []


def test_early_pass_success_normalizes_turns(monkeypatch):
    config = PipelineConfig()
    config.diarization.early_turns = True
    global_result = DiarizationResult(
        turns=[SpeakerTurn(-1.0, 1.0, 0), SpeakerTurn(1.0, 9.0, 1)],
        exclusive_turns=[SpeakerTurn(-1.0, 1.0, 0), SpeakerTurn(1.0, 9.0, 1)],
        speaker_count=2,
        backend="pyannote-community-1",
        status="ok",
    )
    monkeypatch.setattr(
        speaker_fusion,
        "_run_global_pass",
        lambda audio, sample_rate, config: (
            global_result,
            "pyannote/community-1",
            "ok",
        ),
    )

    state = run_early_global_pass(
        np.zeros(4 * 16000, dtype=np.float32),
        16000,
        config,
        duration=4.0,
    )

    assert state.active
    assert state.attempted
    assert state.speaker_count == 2
    assert not state.single_speaker
    # turns 归一到全局时间轴（裁掉负时间与超时长部分）
    assert state.turns[0].start == 0.0
    assert state.turns[1].end == 4.0
    assert state.diagnostics["global_turn_count"] == 2


def test_early_pass_unavailable_falls_back_to_postprocess_fusion(monkeypatch):
    """pyannote 不可用 → 状态非 ok，后处理 fusion 照跑（现状行为）。"""
    pipeline = Pipeline(PipelineConfig())
    pipeline._early_turns_state = EarlyTurnsState(
        status="unavailable",
        attempted=True,
        diagnostics={"global_status": "unavailable"},
    )
    config = _postprocess_config()

    fusion_result = speaker_fusion.SpeakerFusionResult(
        events=[SubtitleEvent(1, 0.0, 1.0, "甲", speaker_id=3, speaker_label="X")],
        speaker_count=1,
        backend="embedding",
        status="ok",
        diagnostics={"embedding_silhouette": 0.42},
    )
    called = {}

    def _fake_fusion(*args, **kwargs):
        called["fusion"] = True
        return fusion_result

    monkeypatch.setattr(speaker_fusion, "run_speaker_fusion", _fake_fusion)

    events, stats = _run_postprocess(
        pipeline,
        [SubtitleEvent(1, 0.0, 1.0, "甲")],
        config,
    )

    assert called.get("fusion") is True
    assert stats.speaker_count == 1
    assert stats.diarization_backend == "embedding"
    assert stats.diarization_silhouette == 0.42
    assert stats.quality_diagnostics["embedding_silhouette"] == 0.42


def test_early_turns_false_keeps_postprocess_fusion_path(monkeypatch):
    """回归等价：early_turns=false 时后处理事件级聚类照常执行。"""
    pipeline = Pipeline(PipelineConfig())
    assert pipeline.config.diarization.early_turns is False
    pipeline._early_turns_state = EarlyTurnsState(
        status="disabled",
        diagnostics={"reason": "early_turns_disabled"},
    )

    fusion_result = speaker_fusion.SpeakerFusionResult(
        events=[SubtitleEvent(1, 0.0, 1.0, "甲", speaker_id=0)],
        speaker_count=1,
        backend="agglomerative",
        status="ok",
        diagnostics={"global_turn_count": 0},
    )
    called = {}
    monkeypatch.setattr(
        speaker_fusion,
        "run_speaker_fusion",
        lambda *args, **kwargs: called.update(fusion=True) or fusion_result,
    )

    events, stats = _run_postprocess(
        pipeline,
        [SubtitleEvent(1, 0.0, 1.0, "甲")],
        pipeline.config,
    )

    assert called.get("fusion") is True
    assert stats.diarization_backend == "agglomerative"
    assert "early_turns_status" not in stats.quality_diagnostics


def test_early_labeling_error_falls_back_to_fusion(monkeypatch):
    """early_turns 标签注入异常 → 完整回退到后处理 fusion。"""
    pipeline = Pipeline(PipelineConfig())
    pipeline._early_turns_state = _ok_state([SpeakerTurn(0.0, 4.0, 0)])

    fusion_result = speaker_fusion.SpeakerFusionResult(
        events=[SubtitleEvent(1, 0.0, 1.0, "甲")],
        speaker_count=0,
        backend="unknown",
        status="degraded",
    )
    monkeypatch.setattr(
        "vocal_subtitle.diarization.speaker_fusion.run_speaker_fusion",
        lambda *args, **kwargs: fusion_result,
    )
    config = _postprocess_config()
    # 传 None turns 触发注入异常 → 回退
    pipeline._early_turns_state.turns = None

    events, stats = _run_postprocess(
        pipeline,
        [SubtitleEvent(1, 0.0, 1.0, "甲")],
        config,
    )

    assert stats.diarization_backend == "unknown"
    assert stats.diarization_status == "degraded"
    assert "early_turns_fallback_reason" in stats.quality_diagnostics


# ---------------------------------------------------------------------------
# 单说话人短路
# ---------------------------------------------------------------------------


def test_single_speaker_shortcut_skips_splitting_and_multi_speaker_path():
    """TTS/口播素材：turns ≤1 个说话人时跳过切分，整链继承单一标签。"""
    pipeline = _zh_pipeline()
    pipeline._early_turns_state = _ok_state([SpeakerTurn(0.0, 8.0, 0)])
    config = _postprocess_config()  # word_split_on_turn=True 也不得切分
    event = SubtitleEvent(
        1,
        0.0,
        8.0,
        "第一句第二句",
        words=[WordTimestamp("第一句", 0.5, 3.0), WordTimestamp("第二句", 4.0, 7.0)],
    )

    events, stats = _run_postprocess(pipeline, [event], config)

    assert len(events) == 1
    assert events[0].speaker_id == 0
    assert events[0].speaker_label == "说话人A"
    assert stats.local_speaker_split_count == 0
    assert stats.quality_diagnostics["single_speaker_shortcut"] is True
    assert stats.quality_diagnostics["early_turn_word_split"] is False


def test_word_split_still_applies_when_shortcut_disabled():
    """single_speaker_shortcut=false → 即使单人也强制走全路径配置语义。"""
    pipeline = Pipeline(PipelineConfig())
    pipeline._early_turns_state = _ok_state(
        [
            SpeakerTurn(0.0, 2.0, 0),
            SpeakerTurn(2.0, 4.0, 1),
        ]
    )
    config = _postprocess_config(single_speaker_shortcut=False)
    event = SubtitleEvent(
        1,
        0.0,
        4.0,
        "甲乙",
        words=[WordTimestamp("甲", 0.2, 1.8), WordTimestamp("乙", 2.1, 3.8)],
    )

    events, stats = _run_postprocess(pipeline, [event], config)

    assert len(events) == 2
    assert [e.speaker_id for e in events] == [0, 1]
    assert stats.quality_diagnostics["single_speaker_shortcut"] is False


# ---------------------------------------------------------------------------
# P1：无词级时间戳 fallback 与 word_split=false 主说话人继承
# ---------------------------------------------------------------------------


def test_no_word_timestamps_fallback_uses_split_event_intervals():
    """无词级时间戳：保留整段单事件并标记降级,不压进首个说话人片段。

    高精度方案 Task 5(2026-09-14)定案:旧 fallback 会把事件钳制到
    第一个说话人片段(整句文本压进小片段),改为整段保留 +
    ``speaker_split_degraded`` 降级标记,归属主说话人。
    """
    turns = [SpeakerTurn(0.0, 2.0, 0), SpeakerTurn(2.0, 4.0, 1)]
    event = SubtitleEvent(1, 0.0, 4.0, "整段文本")

    events, diag = assign_event_speakers(
        [event],
        turns,
        word_split=True,
        language="zh",
    )

    assert len(events) == 1
    assert events[0].speaker_id == 0
    assert events[0].text == "整段文本"
    # 事件不再被钳制到第一个说话人片段(Task 5)
    assert events[0].start == 0.0 and events[0].end == 4.0
    assert events[0].speaker_split_degraded is True
    assert diag["speaker_split_degraded_count"] == 1


def test_word_split_off_inherits_dominant_speaker_without_splitting():
    """word_split=false：事件不切分，整事件继承主说话人。"""
    turns = [SpeakerTurn(0.0, 2.0, 0), SpeakerTurn(2.0, 4.0, 1)]
    event = SubtitleEvent(
        1,
        0.0,
        4.0,
        "甲乙",
        words=[WordTimestamp("甲", 0.2, 1.8), WordTimestamp("乙", 2.1, 3.8)],
    )

    events, diag = assign_event_speakers([event], turns, word_split=False)

    assert len(events) == 1
    assert (events[0].start, events[0].end) == (0.0, 4.0)
    assert events[0].speaker_id == 0  # 主覆盖说话人（前半段等长→先到者）
    assert diag["local_split_count"] == 0


def test_ids_are_compacted_and_labels_stable():
    """全局 ID 压缩契约与 run_speaker_fusion 一致（0..n-1 + 稳定标签）。"""
    turns = [SpeakerTurn(0.0, 1.0, 7), SpeakerTurn(1.0, 2.0, 3)]
    events = [
        SubtitleEvent(1, 0.0, 1.0, "甲"),
        SubtitleEvent(2, 1.0, 2.0, "乙"),
    ]

    events, diag = assign_event_speakers(events, turns, language="zh")

    # 全局 ID 7 与 3 按出现顺序压缩为 0..n-1（与 run_speaker_fusion 一致）
    assert [e.speaker_id for e in events] == [1, 0]
    assert [e.speaker_label for e in events] == ["说话人B", "说话人A"]
    assert diag["speaker_count"] == 2


# ---------------------------------------------------------------------------
# P1：reconcile_regions 接线（骨架 × turns）
# ---------------------------------------------------------------------------


def test_spans_from_skeleton_splits_and_merges_same_speaker():
    turns = [
        SpeakerTurn(0.0, 1.0, 0),
        SpeakerTurn(1.0, 2.0, 1),
        SpeakerTurn(2.0, 3.0, 1),
        SpeakerTurn(3.0, 4.0, 0),
    ]

    spans = spans_from_skeleton(
        [(0.0, 4.0)],
        turns,
        duration=4.0,
        boundary_collar_ms=0,
    )

    assert [(s.start, s.end, s.speaker_id) for s in spans] == [
        (0.0, 1.0, 0),
        (1.0, 3.0, 1),  # 相邻同 speaker 合并
        (3.0, 4.0, 0),
    ]


def test_spans_from_skeleton_without_turn_coverage_keeps_unknown():
    turns = [SpeakerTurn(5.0, 6.0, 0)]

    spans = spans_from_skeleton(
        [(0.0, 1.0)],
        turns,
        duration=6.0,
    )

    assert len(spans) == 1
    assert spans[0].speaker_id is None
    assert spans[0].speaker_source == "unknown"


def test_attach_early_turns_context_and_segment_speaker_lookup():
    pipeline = Pipeline(PipelineConfig())
    pipeline._early_turns_state = _ok_state(
        [
            SpeakerTurn(0.0, 2.0, 0),
            SpeakerTurn(2.0, 4.0, 1),
        ]
    )
    pipeline._early_turn_spans = spans_from_skeleton(
        [(0.0, 4.0)],
        pipeline._early_turns_state.turns,
        duration=4.0,
        boundary_collar_ms=0,
    )

    ctx = PipelineContext(audio_path=None, audio=None, sample_rate=16000)
    pipeline._attach_early_turns_context(ctx, time_offset=0.0)
    assert len(ctx.early_turns) == 2
    assert len(ctx.early_turn_spans) == 2
    assert ctx.early_turns_window_offset == 0.0
    assert any("global_turn_count=2" in d for d in ctx.diagnostics)

    segments = [
        SpeechSegment(0.2, 1.8),  # 说话人 A
        SpeechSegment(2.1, 3.6),  # 说话人 B
    ]
    ids = pipeline._early_speaker_ids_for_segments(segments, ctx, duration=4.0)
    assert ids == [0, 1]

    # 窗口带偏移时按全局坐标查询（局部 + offset）
    ctx_shifted = PipelineContext(audio_path=None, audio=None, sample_rate=16000)
    pipeline._attach_early_turns_context(ctx_shifted, time_offset=10.0)
    ids_shifted = pipeline._early_speaker_ids_for_segments(
        segments,
        ctx_shifted,
        duration=4.0,
    )
    assert ids_shifted == [None, None]  # 10s 之后无 turns 覆盖

    # 无 spans 时回退 turns 主说话人
    ctx_turns_only = PipelineContext(audio_path=None, audio=None, sample_rate=16000)
    ctx_turns_only.early_turns = list(pipeline._early_turns_state.turns)
    ctx_turns_only.early_turns_window_offset = 0.0
    ids_turns = pipeline._early_speaker_ids_for_segments(
        segments,
        ctx_turns_only,
        duration=4.0,
    )
    assert ids_turns == [0, 1]


# ---------------------------------------------------------------------------
# P1：tiny-fragment 合并的说话人安全检查
# ---------------------------------------------------------------------------


@dataclass
class _StubText:
    text: str


def _asr(text):
    return [_StubText(text)]


def test_tiny_fragment_same_speaker_merges():
    pipeline = Pipeline(PipelineConfig())
    segments = [SpeechSegment(0.0, 0.4), SpeechSegment(0.4, 2.0)]
    speaker_ids = [0, 0]

    merged_segs, merged_asr, merged_ids = pipeline._filter_tiny_fragments(
        segments,
        [_asr("1."), _asr("正文内容足够长")],
        speaker_ids,
    )

    assert len(merged_segs) == 1
    assert merged_asr[0][0].text == "1."
    assert merged_asr[0][1].text == "正文内容足够长"


def test_tiny_fragment_different_speakers_kept_separate():
    """early_turns 生效：不同说话人的碎片保留独立段（死分支复活）。"""
    pipeline = Pipeline(PipelineConfig())
    segments = [
        SpeechSegment(0.0, 0.4),
        SpeechSegment(0.4, 2.0),
        SpeechSegment(2.0, 4.0),
    ]
    speaker_ids = [1, 0, 0]  # "得了吧"（B）与后文（A）不同人

    merged_segs, _, merged_ids = pipeline._filter_tiny_fragments(
        segments,
        [_asr("得了吧"), _asr("甲段内容足够长"), _asr("乙段内容也够长")],
        speaker_ids,
    )

    assert len(merged_segs) == 3
    assert merged_ids == [1, 0, 0]


def test_tiny_fragment_mixed_speaker_info_keeps_separate():
    """一侧 turn 未覆盖（None）→ 与已知说话人保守不合并。"""
    pipeline = Pipeline(PipelineConfig())
    segments = [SpeechSegment(0.0, 0.4), SpeechSegment(0.4, 2.0)]

    merged_segs, _, merged_ids = pipeline._filter_tiny_fragments(
        segments,
        [_asr("1."), _asr("正文内容足够长")],
        [0, None],
    )

    assert len(merged_segs) == 2
    assert merged_ids == [0, None]


def test_tiny_fragment_two_uncovered_segments_merge():
    """两侧都无 turn 覆盖（None == None）→ 无冲突，安全合并。"""
    pipeline = Pipeline(PipelineConfig())
    segments = [SpeechSegment(0.0, 0.4), SpeechSegment(0.4, 2.0)]

    merged_segs, _, merged_ids = pipeline._filter_tiny_fragments(
        segments,
        [_asr("1."), _asr("正文内容足够长")],
        [None, None],
    )

    assert len(merged_segs) == 1
    assert merged_ids == [None]


def test_tiny_fragment_without_speaker_info_keeps_legacy_behavior():
    """speaker_ids 为空（early_turns 关闭）→ 无信息安全合并（现状行为）。"""
    pipeline = Pipeline(PipelineConfig())
    segments = [SpeechSegment(0.0, 0.4), SpeechSegment(0.4, 2.0)]

    merged_segs, _, merged_ids = pipeline._filter_tiny_fragments(
        segments,
        [_asr("1."), _asr("正文内容足够长")],
        [],
    )

    assert len(merged_segs) == 1
    assert merged_ids == []


# ---------------------------------------------------------------------------
# P1：speaker_offset 补丁在 early_turns 生效时跳过
# ---------------------------------------------------------------------------


def test_early_turns_active_gate_controls_offset_patch():
    pipeline = Pipeline(PipelineConfig())
    assert pipeline._early_turns_active() is False  # 未运行/关闭

    pipeline._early_turns_state = EarlyTurnsState(status="failed", attempted=True)
    assert pipeline._early_turns_active() is False

    pipeline._early_turns_state = _ok_state([SpeakerTurn(0.0, 4.0, 0)])
    assert pipeline._early_turns_active() is True


def test_dominant_speaker_helpers():
    turns = [SpeakerTurn(0.0, 1.0, 0), SpeakerTurn(1.5, 3.0, 1)]
    assert dominant_speaker_at(turns, 0.0, 1.0) == 0
    assert dominant_speaker_at(turns, 1.6, 2.6) == 1
    assert dominant_speaker_at(turns, 1.0, 1.5) is None  # turn 间隙


def test_pyannote_turn_overlapped_propagates_from_global_pass(monkeypatch):
    """全局 pass 的 overlapped turn 归一后保留标记（供 P2 标注）。"""
    config = PipelineConfig()
    config.diarization.early_turns = True
    global_result = DiarizationResult(
        turns=[
            SpeakerTurn(0.0, 2.0, 0),
            replace(SpeakerTurn(1.5, 3.0, 1), overlapped=True),
        ],
        exclusive_turns=[SpeakerTurn(0.0, 2.0, 0)],
        speaker_count=2,
        status="ok",
    )
    monkeypatch.setattr(
        speaker_fusion,
        "_run_global_pass",
        lambda audio, sample_rate, cfg: (global_result, "pyannote/community-1", "ok"),
    )

    state = run_early_global_pass(
        np.zeros(4 * 16000, dtype=np.float32),
        16000,
        config,
        duration=4.0,
    )

    assert state.active
    assert any(turn.overlapped for turn in state.turns)
