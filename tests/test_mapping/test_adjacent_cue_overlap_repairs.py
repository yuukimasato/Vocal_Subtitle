"""相邻字幕时间重叠修复的回归测试（2026-09-14 定案）。

重叠 bug 因果链（181人声.srt 实测 1611ms / 2214ms 两处重叠）：
1. H2 源头：`_time_utterances` 词时间戳路径无包络钳制，合并事件
   词表跨句非单调时子句 start 倒置进上一子句；
2. final_validator 把重叠裁成相等边界（next.start == prev.end）；
3. H1 致命一步：显示能量对齐的 end 延长带 `next.start > event.end`
   守卫，恰好跳过相等边界上的让界钳制，end 直接越过下一事件起点。

修复：H2 包络+时序钳制、H1 无条件让界、导出前 enforce_non_overlap
最后防线、H3 同说话人跨 cue 告警。
"""

import pytest

from vocal_subtitle.acoustic.validator import (
    AcousticValidationConfig,
    AcousticValidator,
)
from vocal_subtitle.asr.base import WordTimestamp
from vocal_subtitle.mapping.display_timeline import map_to_display_timeline
from vocal_subtitle.mapping.final_validator import enforce_non_overlap
from vocal_subtitle.mapping.subtitle_builder import SubtitleBuilder
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


def _event(index, start, end, text, **kwargs):
    return SubtitleEvent(index=index, start=start, end=end, text=text, **kwargs)


# ── H1：end 延长必须无条件让界 ───────────────────────────────────────


class TestEndExtensionYieldsAtEqualBoundary:
    """骨架段内 end 延长在相等边界（next.start == end）上不得越界。"""

    def test_equal_boundary_blocks_extension(self):
        validator = AcousticValidator(AcousticValidationConfig())
        events = [
            _event(1, 10.0, 15.0, "第一句"),
            _event(2, 15.0, 20.0, "第二句"),
        ]
        result, report = validator._physical_snap_validation(events, [(10.0, 21.0)])
        # 旧行为：守卫跳过钳制，end 延长到 20.99，与下一句重叠 5.99s
        assert result[0].end <= result[1].start
        # 第一句在相等边界上不得延长（第二句无后继，延长到骨架末端
        # 属于正常截尾修复，不计入本断言）
        assert result[0].end == pytest.approx(15.0)

    def test_gap_boundary_still_extends(self):
        """正常间隙（next.start > end）的截尾修复能力不受影响。"""
        validator = AcousticValidator(AcousticValidationConfig())
        events = [
            _event(1, 10.0, 15.0, "第一句"),
            _event(2, 15.5, 20.0, "第二句"),
        ]
        result, report = validator._physical_snap_validation(events, [(10.0, 21.0)])
        assert report["ends_extended"] >= 1
        assert result[0].end <= result[1].start
        assert result[0].end == pytest.approx(15.48, abs=0.01)


# ── H2：子句词时间戳路径的包络与时序钳制 ────────────────────────────


class TestUtteranceWordTimingClamp:
    """词时间戳非单调（合并事件跨句边界）时子句时序不得倒置。"""

    def test_non_monotonic_words_stay_ordered(self):
        event = _event(
            0,
            10.0,
            16.0,
            "我答应过师傅. 我知道你心中不平.",
            words=[
                WordTimestamp(word="我答应过师傅.", start=0.0, end=3.0),
                # 第二句首词时间戳倒置：落在第一句内部（rel 1.6）
                WordTimestamp(word="我知道你心中不平.", start=1.6, end=5.8),
            ],
        )
        utterances = ["我答应过师傅.", "我知道你心中不平."]
        timed = SubtitleBuilder._time_utterances(
            utterances,
            event,
            6.0,
            SubtitleBuilder._count_display_chars(event.text),
        )
        assert len(timed) == 2
        # 旧行为：第二句 start=11.6 < 第一句 end=13.0（重叠 1.4s）
        assert timed[1]["start"] >= timed[0]["end"]
        for utt in timed:
            assert utt["start"] >= event.start
            assert utt["end"] <= event.end
            assert utt["end"] > utt["start"]

    def test_no_words_fallback_still_monotonic(self):
        event = _event(0, 10.0, 16.0, "我答应过师傅. 我知道你心中不平.")
        utterances = ["我答应过师傅.", "我知道你心中不平."]
        timed = SubtitleBuilder._time_utterances(
            utterances,
            event,
            6.0,
            SubtitleBuilder._count_display_chars(event.text),
        )
        assert timed[1]["start"] >= timed[0]["end"]


# ── 导出前最后防线：enforce_non_overlap ─────────────────────────────


class TestEnforceNonOverlap:
    def test_trims_previous_end_to_next_start(self):
        events = [
            _event(1, 10.0, 15.5, "甲"),
            _event(2, 13.0, 18.0, "乙"),
        ]
        result, diag = enforce_non_overlap(events, source="test")
        assert diag["overlap_count"] == 1
        assert len(diag["repairs"]) == 1
        assert diag["repairs"][0]["overlap_ms"] == pytest.approx(2500.0)
        assert result[0].end == pytest.approx(13.0)
        assert all(result[i + 1].start >= result[i].end for i in range(len(result) - 1))

    def test_drops_previous_when_trim_empties_it(self):
        events = [
            _event(1, 10.0, 13.0, "甲"),
            _event(2, 10.0, 12.0, "乙"),
        ]
        result, diag = enforce_non_overlap(events, source="test")
        assert diag["overlap_count"] == 1
        assert len(result) == 1
        assert result[0].text == "乙"

    def test_disjoint_events_untouched(self):
        events = [
            _event(1, 10.0, 12.0, "甲"),
            _event(2, 12.5, 15.0, "乙"),
        ]
        result, diag = enforce_non_overlap(events, source="test")
        assert diag["overlap_count"] == 0
        assert diag["repairs"] == []
        assert result[0].end == pytest.approx(12.0)
        assert result[1].end == pytest.approx(15.0)
        assert [e.index for e in result] == [1, 2]


# ── H3：同说话人跨 cue 重叠告警 ─────────────────────────────────────


class TestSameSpeakerDisplayOverlap:
    def _groups(self, second_speaker):
        return [
            {
                "physical_start": 1.0,
                "physical_end": 2.6,
                "text": "第一句",
                "index": 1,
                "speaker_id": 1,
            },
            {
                "physical_start": 2.0,
                "physical_end": 3.0,
                "text": "第二句",
                "index": 2,
                "speaker_id": 2 if second_speaker else 1,
            },
        ]

    def test_same_speaker_overlap_warns(self):
        cues = map_to_display_timeline(
            self._groups(second_speaker=False), audio_duration=10.0
        )
        assert "same_speaker_display_overlap" in cues[1].warnings
        assert "speaker_boundary_display_conflict" not in cues[1].warnings

    def test_different_speaker_warning_unchanged(self):
        cues = map_to_display_timeline(
            self._groups(second_speaker=True), audio_duration=10.0
        )
        assert "speaker_boundary_display_conflict" in cues[1].warnings
        assert "same_speaker_display_overlap" not in cues[1].warnings

    def test_disjoint_same_speaker_no_warning(self):
        groups = [
            {
                "physical_start": 1.0,
                "physical_end": 2.0,
                "text": "第一句",
                "index": 1,
                "speaker_id": 1,
            },
            {
                "physical_start": 2.5,
                "physical_end": 3.5,
                "text": "第二句",
                "index": 2,
                "speaker_id": 1,
            },
        ]
        cues = map_to_display_timeline(groups, audio_duration=10.0)
        assert cues[1].warnings == []
