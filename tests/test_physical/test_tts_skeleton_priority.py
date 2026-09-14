"""骨架优先边界策略(高精度方案 Task 8 / 优化方案 §10)。

TTS/干净单人播报场景:骨架段即 cue 的硬物理范围——
- cue 起点不早于骨架起点,终点不晚于骨架终点(向内 3ms 安全余量);
- 骨架间静音是硬边界,字幕不得跨越(跨段事件钳制回主骨架段);
- 完全落在骨架之外的事件保持原状并计数;
- 通用配置(不带 skeleton_priority)结果保持不变。
"""

from vocal_subtitle.acoustic.validator import AcousticValidator
from vocal_subtitle.config.models import AcousticValidationConfig
from vocal_subtitle.mapping.time_mapper import SubtitleEvent

SKELETON = [(1.0, 2.0), (3.0, 4.0)]
FFMPEG_RESULT = {"skeleton": SKELETON}


def _event(index, start, end, text="hello"):
    return SubtitleEvent(
        index=index,
        start=start,
        end=end,
        text=text,
        physical_start=start,
        physical_end=end,
    )


def _validator(**overrides):
    base = dict(
        enabled=True,
        unified_ffmpeg_pass=True,
        skeleton_priority=True,
        snap_end_margin=0.003,
        snap_start_margin=0.01,
    )
    base.update(overrides)
    return AcousticValidator(AcousticValidationConfig(**base))


def test_skeleton_priority_clamps_events_into_skeleton_segments():
    events = [
        _event(1, 1.05, 1.95),   # 段内,基本不动
        _event(2, 2.4, 3.5),     # 跨骨架间静音,钳制回主段 seg2
        _event(3, 3.6, 3.9),     # seg2 内
    ]

    validated, report = _validator().validate(
        events, ffmpeg_unified_result=FFMPEG_RESULT,
    )

    assert report["skeleton_priority"] is True
    e1, e2, e3 = validated
    # 每条 cue 都在自己主骨架段的物理范围内。
    for event in validated:
        seg = next(
            (s for s in SKELETON if s[0] <= event.start and event.end <= s[1] + 0.003),
            None,
        )
        assert seg is not None, f"event {event.start}-{event.end} outside any skeleton"
    # 跨静音事件被拉回主段,不再跨越 [2.0, 3.0] 的静音。
    assert e2.start >= 3.0
    assert e2.end <= 4.0 + 0.003
    # 相邻 cue 不重叠(同段内顺序保持)。
    assert e1.end <= e2.start + 1e-9 or e2.end <= e3.start + 1e-9
    assert e2.end <= e3.start + 1e-9
    # 诊断字段齐全:e2 原范围跨越骨架间静音被钳回主段;
    # e2/e3 同属 seg2,构成一对上游微停顿拆分(TTS 模式保留不合并)。
    assert report["cross_skeleton_merge_count"] == 1
    assert report["micro_pause_split_count"] == 1
    assert "skeleton_start_delta_ms" in report
    assert "skeleton_end_delta_ms" in report


def test_skeleton_priority_pulls_start_to_skeleton_onset():
    # ASR 漏掉起爆音:cue 起点比骨架段晚 300ms → 拉回骨架起点。
    events = [_event(1, 1.3, 1.9)]

    validated, report = _validator().validate(
        events, ffmpeg_unified_result=FFMPEG_RESULT,
    )

    assert validated[0].start == 1.0
    assert validated[0].end <= 2.0
    assert report["skeleton_start_delta_ms"] >= 0.0


def test_skeleton_priority_leaves_uncovered_events_untouched():
    events = [_event(1, 5.0, 5.5)]

    validated, report = _validator().validate(
        events, ffmpeg_unified_result=FFMPEG_RESULT,
    )

    assert validated[0].start == 5.0
    assert validated[0].end == 5.5
    assert report["skeleton_uncovered_count"] == 1


def test_default_config_behavior_unchanged_without_skeleton_priority():
    events = [_event(1, 2.4, 3.5)]

    # 不带 skeleton_priority:跨静音事件不会被强制钳回单个骨架段。
    validator = AcousticValidator(AcousticValidationConfig(
        enabled=True,
        unified_ffmpeg_pass=True,
        skeleton_priority=False,
        allow_end_extend=False,
        allow_end_shorten=False,
        allow_start_pull_earlier=False,
        max_snap_distance=0.0,
        max_start_snap_distance=0.0,
        confidence_threshold=1.1,
    ))
    validated, report = validator.validate(
        events, ffmpeg_unified_result=FFMPEG_RESULT,
    )

    assert report.get("skeleton_priority") is not True
    assert validated[0].start < 3.0
