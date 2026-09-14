"""统一词级边界裁决:时间来源优先级 + 分置信度吸附限幅。

优先级(优化方案 4.1):
    whisperx_alignment > faster_whisper_word > segment_boundary

吸附限幅(优化方案 4.2):
    高置信 ≤120ms;中置信 ≤200ms;低置信不自动移动仅标记;
    超过 200ms 必须通过局部 RMS、骨架连续、相邻事件和全局文本一致性确认,
    且不得跨越硬静音。
"""

import numpy as np

from vocal_subtitle.physical.boundary_arbiter import BoundaryDecision
from vocal_subtitle.physical.boundary_decision import (
    LARGE_SNAP_LIMIT,
    TIER_SNAP_LIMITS,
    WORD_TIME_SOURCE_PRECEDENCE,
    enforce_snap_policy,
    resolve_word_time_source,
    time_source_rank,
)
from vocal_subtitle.physical.ir import GlobalWord
from vocal_subtitle.physical.allocator import WordAllocation
from vocal_subtitle.physical.subtitle_bins import PhysicalSubtitleBin
from vocal_subtitle.physical.timeline import PhysicalTimeline
from vocal_subtitle.physical.word_alignment import align_words_to_physical


def _word(text="hello", start=1.0, end=1.5, confidence=0.95, metadata=None):
    return GlobalWord(
        id=f"gw:{text}:{start}",
        text=text,
        raw_start=start,
        raw_end=end,
        confidence=confidence,
        source_window_id="w",
        segment_id="s",
        metadata=metadata or {},
    )


def _alloc(word, bin_id="bin-1"):
    return WordAllocation(
        word=word,
        physical_spans=(),
        evidence_ids=("e1",),
        accepted=True,
        physical_bin_id=bin_id,
    )


def _decision(time, raw, accepted=True, confidence=0.9):
    return BoundaryDecision(
        accepted=accepted,
        boundary_time=time,
        boundary_type="start",
        confidence=confidence,
        reason_codes=("aligned_start",) if accepted else ("no_legal_start_candidate",),
    )


# ---------------------------------------------------------------------------
# 时间来源优先级
# ---------------------------------------------------------------------------


def test_time_source_precedence_order():
    assert WORD_TIME_SOURCE_PRECEDENCE == (
        "whisperx_alignment",
        "faster_whisper_word",
        "segment_boundary",
    )
    assert time_source_rank("whisperx_alignment") == 0
    assert time_source_rank("faster_whisper_word") == 1
    assert time_source_rank("segment_boundary") == 2


def test_resolve_word_time_source_reads_metadata():
    aligned = _word(metadata={"time_source": "whisperx_alignment"})
    native = _word(metadata={"time_source": "faster_whisper_word"})
    missing = _word()
    invalid = _word(metadata={"time_source": "made_up_source"})

    assert resolve_word_time_source(aligned) == "whisperx_alignment"
    assert resolve_word_time_source(native) == "faster_whisper_word"
    # 缺失或非法来源回退 segment_boundary,不伪造词级时间。
    assert resolve_word_time_source(missing) == "segment_boundary"
    assert resolve_word_time_source(invalid) == "segment_boundary"


# ---------------------------------------------------------------------------
# 分置信度吸附限幅
# ---------------------------------------------------------------------------


def test_tier_snap_limits_match_plan():
    assert TIER_SNAP_LIMITS["high"] <= 0.12
    assert TIER_SNAP_LIMITS["medium"] <= 0.20
    assert TIER_SNAP_LIMITS["low"] == 0.0
    assert LARGE_SNAP_LIMIT == 0.20


def test_high_confidence_boundary_within_120ms_is_accepted():
    decision = _decision(time=1.10, raw=1.0)

    result = enforce_snap_policy(decision, raw_time=1.0, tier="high")

    assert result.accepted is True
    assert result.boundary_time == 1.10
    assert any(item.get("reason") == "snap_policy_pass" for item in result.revision_trace)


def test_medium_confidence_snap_up_to_200ms_is_accepted():
    decision = _decision(time=1.18, raw=1.0)

    result = enforce_snap_policy(decision, raw_time=1.0, tier="medium")

    assert result.accepted is True
    assert result.boundary_time == 1.18


def test_high_confidence_move_over_120ms_is_clamped():
    decision = _decision(time=1.16, raw=1.0)

    result = enforce_snap_policy(decision, raw_time=1.0, tier="high")

    assert result.boundary_time == 1.0
    assert result.accepted is False
    assert "snap_limit_exceeded" in result.reason_codes


def test_low_confidence_boundary_is_never_moved_automatically():
    decision = _decision(time=1.05, raw=1.0, confidence=0.4)

    result = enforce_snap_policy(decision, raw_time=1.0, tier="low")

    assert result.boundary_time == 1.0
    assert result.accepted is False
    assert "snap_limit_exceeded" in result.reason_codes
    # 候选证据保留在 trace 中,仅诊断不静默丢弃。
    assert any(
        item.get("reason") == "snap_limit_exceeded" for item in result.revision_trace
    )


def test_large_snap_requires_all_escape_hatch_evidence():
    decision = _decision(time=1.35, raw=1.0)

    # 缺少任一确认 → 拒绝。
    partial = enforce_snap_policy(
        decision,
        raw_time=1.0,
        tier="medium",
        large_snap_evidence={
            "rms_confirmed": True,
            "skeleton_continuous": True,
            "adjacent_clear": True,
            "global_text_consistent": False,
        },
    )
    assert partial.boundary_time == 1.0

    # 全部确认 → 允许,但必须记录 trace。
    allowed = enforce_snap_policy(
        decision,
        raw_time=1.0,
        tier="medium",
        large_snap_evidence={
            "rms_confirmed": True,
            "skeleton_continuous": True,
            "adjacent_clear": True,
            "global_text_consistent": True,
        },
    )
    assert allowed.boundary_time == 1.35
    assert any(
        item.get("reason") == "large_snap_allowed" for item in allowed.revision_trace
    )


def test_snap_across_hard_silence_is_rejected():
    decision = _decision(time=0.85, raw=1.0)
    hard_silences = ((0.90, 0.95),)

    result = enforce_snap_policy(
        decision,
        raw_time=1.0,
        tier="high",
        hard_silences=hard_silences,
    )

    assert result.boundary_time == 1.0
    assert "cross_hard_silence" in result.reason_codes


def test_snap_policy_preserves_revision_trace_and_time_source():
    decision = _decision(time=1.10, raw=1.0)
    decision = BoundaryDecision(
        accepted=decision.accepted,
        boundary_time=decision.boundary_time,
        boundary_type=decision.boundary_type,
        confidence=decision.confidence,
        reason_codes=decision.reason_codes,
        time_source="whisperx_alignment",
        revision_trace=({"stage": "arbiter", "reason": "aligned_start"},),
    )

    result = enforce_snap_policy(decision, raw_time=1.0, tier="high")

    assert result.time_source == "whisperx_alignment"
    stages = [item.get("stage") for item in result.revision_trace]
    assert stages == ["arbiter", "snap_policy"]
    payload = result.to_dict()
    assert payload["time_source"] == "whisperx_alignment"
    assert BoundaryDecision.from_dict(payload).revision_trace == result.revision_trace


# ---------------------------------------------------------------------------
# align_words_to_physical 集成
# ---------------------------------------------------------------------------


def _bin(bin_id, start, end):
    return PhysicalSubtitleBin(
        id=bin_id,
        start=start,
        end=end,
        source="test",
    )


def test_alignment_stamps_word_time_source_and_runs_snap_policy():
    first = _word("one", start=0.5, end=0.8, confidence=0.95)
    # 第二个词原始起点与前一词重叠(0.75 < 0.8),asr 候选被单调性拒绝,
    # arbiter 选择合法的 bin_start(0.82),snap_policy 记录 30ms 吸附 trace。
    second = _word(
        "two",
        start=0.75,
        end=1.1,
        confidence=0.95,
        metadata={"time_source": "whisperx_alignment"},
    )
    bins = [_bin("bin-1", 0.4, 0.9), _bin("bin-2", 0.82, 1.2)]

    result = align_words_to_physical(
        [_alloc(first, "bin-1"), _alloc(second, "bin-2")], bins
    )

    aligned = result[1]
    assert aligned.time_source == "whisperx_alignment"
    assert aligned.start_boundary_decision.time_source == "whisperx_alignment"
    trace_reasons = [
        item.get("reason")
        for item in aligned.start_boundary_decision.revision_trace
    ]
    assert "snap_policy_pass" in trace_reasons


def test_adjacent_words_do_not_overlap_after_alignment():
    first = _word("one", start=0.5, end=0.8, confidence=0.95)
    second = _word("two", start=0.85, end=1.1, confidence=0.95)
    bins = [_bin("bin-1", 0.4, 0.9), _bin("bin-2", 0.82, 1.2)]

    result = align_words_to_physical(
        [_alloc(first, "bin-1"), _alloc(second, "bin-2")], bins
    )

    assert result[0].aligned_end <= result[1].aligned_start + 1e-9


def test_alignment_rejects_candidate_across_hard_silence_gap():
    # 语音证据 [1.3, 2.0];词起点 1.5,若吸附到 1.3 需跨越静音 [1.2, 1.45]。
    timeline = PhysicalTimeline.from_duration(2.5)
    timeline.add_evidence(1.3, 2.0, "ffmpeg_skeleton")
    word = _word(start=1.5, end=1.8, confidence=0.95)
    word_bin = _bin("bin-1", 1.2, 2.0)

    result = align_words_to_physical(
        [_alloc(word)],
        [word_bin],
        timeline=timeline,
        hard_silences=((1.2, 1.45),),
    )

    aligned = result[0]
    assert aligned.aligned_start >= 1.45
