"""编辑日志摄取（edit-journal-v1）测试：解析校验、去重、重放还原、V1 统计、
编辑器偏好推导、事件级归因与 V3 触发机制。"""

import json
from pathlib import Path

import pytest

from vocal_subtitle.feedback.aligner import parse_subtitle_file
from vocal_subtitle.feedback.diff_analyzer import analyze_journal_events
from vocal_subtitle.feedback.journal_ingest import (
    JournalIngestError,
    check_v3_trigger,
    derive_editor_preferences,
    find_original_subtitle,
    journal_statistics,
    load_journal_files,
    parse_journal_text,
    replay_journal,
)


def _event(session_id="s-1", seq=0, command="updateCueTimes", diff=None, **extra):
    event = {
        "schema": "edit-journal-v1",
        "type": "event",
        "session_id": session_id,
        "seq": seq,
        "ts": "2026-09-10T00:00:00Z",
        "actor": "human",
        "command": command,
        "diff": diff if diff is not None else [],
        "context": extra.pop("context", {}),
    }
    event.update(extra)
    return event


def _ndjson(records) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"


# ---------------------------------------------------------------------------
# 解析与校验
# ---------------------------------------------------------------------------

class TestParse:
    def test_header_events_rejected(self):
        text = _ndjson([
            {"schema": "edit-journal-v1", "type": "header", "session_id": "s-1"},
            _event(seq=0),
            {"schema": "wrong", "type": "event"},
            "not json",
            {"schema": "edit-journal-v1", "type": "event"},  # 缺必填字段
        ])
        header, events, rejected = parse_journal_text(text)
        assert header["session_id"] == "s-1"
        assert len(events) == 1
        assert rejected == 3

    def test_load_files_dedupes_across_files(self, tmp_path):
        record = _event(seq=3)
        (tmp_path / "a.journal.jsonl").write_text(_ndjson([record]), encoding="utf-8")
        (tmp_path / "b.journal.jsonl").write_text(_ndjson([record, _event(seq=4)]), encoding="utf-8")
        files = load_journal_files([tmp_path / "a.journal.jsonl", tmp_path / "b.journal.jsonl"])
        assert files[0].events[0]["seq"] == 3
        assert [e["seq"] for e in files[1].events] == [4]  # 跨文件去重

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(JournalIngestError):
            load_journal_files([tmp_path / "nope.jsonl"])


# ---------------------------------------------------------------------------
# 重放（原始字幕 + 日志 = 最终字幕）
# ---------------------------------------------------------------------------

@pytest.fixture
def original_srt(tmp_path):
    srt = tmp_path / "lesson01.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n第一句\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n第二句\n\n"
        "3\n00:00:08,000 --> 00:00:10,000\n第三句\n",
        encoding="utf-8",
    )
    return srt


class TestReplay:
    def test_modify_remove_add_roundtrip(self, original_srt):
        original = parse_subtitle_file(original_srt)
        events = [
            # 修改第一句时间
            _event(seq=0, diff=[{"op": "modify", "id": "cue-a", "changes": [
                {"field": "start", "before": 1.0, "after": 0.9},
                {"field": "end", "before": 3.0, "after": 3.2},
            ]}]),
            # 第二句与第三句合并（改第二句 end + 删第三句）
            _event(seq=1, command="mergeWithNext", diff=[
                {"op": "modify", "id": "cue-b", "changes": [{"field": "end", "before": 6.0, "after": 6.5}]},
                {"op": "remove", "id": "cue-c", "index": 2, "cue": {"start": 8.0, "end": 10.0, "text": "第三句"}},
            ]),
            # 新增一行
            _event(seq=2, command="insertAtTime", diff=[
                {"op": "add", "id": "cue-new", "index": 2, "cue": {"start": 11.0, "end": 12.5, "text": "新增"}},
            ]),
        ]
        result = replay_journal(original, events)
        texts = [e.text for e in result.final_events]
        assert texts == ["第一句", "第二句", "新增"]
        first, second, third = result.final_events
        assert (first.start, first.end) == (0.9, 3.2)
        assert (second.start, second.end) == (4.0, 6.5)
        assert (third.start, third.end) == (11.0, 12.5)
        assert [e.index for e in result.final_events] == [1, 2, 3]  # 重编号
        assert result.exact is True
        assert result.matched_ids == 3

    def test_untouched_cues_preserved(self, original_srt):
        original = parse_subtitle_file(original_srt)
        result = replay_journal(original, [])
        assert [e.text for e in result.final_events] == ["第一句", "第二句", "第三句"]
        assert result.exact is True

    def test_empty_events_list_rejected(self, original_srt):
        original = parse_subtitle_file(original_srt)
        result = replay_journal(original, [_event(seq=0, diff=[{"op": "modify", "id": "ghost", "changes": [
            {"field": "start", "before": 99.0, "after": 98.0}]}])])
        # 对不回任何原始事件 → 不精确（警告信号，不阻塞）
        assert result.exact is False


# ---------------------------------------------------------------------------
# V1 统计与偏好
# ---------------------------------------------------------------------------

def _stats_events():
    return [
        _event(seq=0, diff=[{"op": "modify", "id": "a", "changes": [
            {"field": "start", "before": 1.0, "after": 0.9}]}],
            context={"cue": {"gap_after": 1.5, "cps": 2.0}, "audio": {"nearest_speech_boundary": {"t": 0.92, "dist": 0.02}}}),
        _event(seq=1, diff=[{"op": "modify", "id": "b", "changes": [
            {"field": "end", "before": 3.0, "after": 3.3}]}],
            context={"cue": {"gap_after": 2.5, "cps": 3.0}}, provenance={"source_stage": "asr"}),
        _event(seq=2, command="removeCues", diff=[
            {"op": "remove", "id": "c", "index": 0, "cue": {"start": 8.0, "end": 9.0, "text": "幻觉行"}}]),
    ]


class TestStatistics:
    def test_summary(self):
        header = {"schema": "edit-journal-v1", "type": "header", "session_id": "s-1"}
        _, events, _ = parse_journal_text(_ndjson([header, *_stats_events()]))
        journal_file = type("JF", (), {"path": Path("x"), "header": header, "events": events, "skipped": 0})()
        stats = journal_statistics([journal_file])
        assert stats.event_count == 3
        assert stats.session_count == 1
        assert stats.commands["updateCueTimes"] == 2
        assert stats.structural["remove"] == 1
        assert stats.start_delta_median == pytest.approx(-0.1)
        assert stats.end_delta_median == pytest.approx(0.3)
        assert stats.gap_after_median == pytest.approx(2.0)
        # 线性插值 P90：2.0 + (3.0 - 2.0) * 0.9
        assert stats.cps_p90 == pytest.approx(2.9)
        assert stats.boundary_snap_rate == pytest.approx(1.0)
        assert stats.provenance_coverage == pytest.approx(1 / 3)


class TestEditorPreferences:
    def test_systematic_shift_only_above_threshold(self):
        stats = type("S", (), {
            "start_delta_median": -0.1, "end_delta_median": 0.005,
            "gap_after_median": 0.8, "cps_p90": 4.5, "boundary_snap_rate": 0.9,
        })()
        prefs = derive_editor_preferences(stats)
        assert prefs["editor.offset_start_ms"] == -100.0
        assert "editor.offset_end_ms" not in prefs  # 5ms 不足以体现系统性偏移
        assert prefs["editor.gap_preference_ms"] == 800.0
        assert prefs["editor.cps_ceiling"] == 4.5
        assert prefs["editor.boundary_snap_to_speech"] == 1.0

    def test_no_preferences_without_signal(self):
        stats = type("S", (), {
            "start_delta_median": None, "end_delta_median": None,
            "gap_after_median": None, "cps_p90": None, "boundary_snap_rate": None,
        })()
        assert derive_editor_preferences(stats) == {}


# ---------------------------------------------------------------------------
# 事件级归因
# ---------------------------------------------------------------------------

class TestJournalAttribution:
    def test_padding_direction(self):
        events = [
            _event(seq=0, diff=[{"op": "modify", "id": "a", "changes": [
                {"field": "start", "before": 1.0, "after": 0.85}]}]),   # 开始提前 → padding_min ↑
            _event(seq=1, diff=[{"op": "modify", "id": "b", "changes": [
                {"field": "end", "before": 3.0, "after": 2.9}]}]),      # 结束收紧 → padding_max ↓
        ]
        attribution = analyze_journal_events(events)
        assert attribution["merging.padding_min"].direction == "increase"
        assert attribution["merging.padding_max"].direction == "decrease"

    def test_merge_signal(self):
        events = [
            _event(seq=i, command="mergeWithNext") for i in range(4)
        ] + [_event(seq=9, command="updateCueTimes")]
        attribution = analyze_journal_events(events)
        assert attribution["merge_decision.fast_merge_max_gap"].direction == "increase"

    def test_insufficient_behavior_no_attribution(self):
        assert analyze_journal_events([]) == {}
        tiny = [_event(seq=0, diff=[{"op": "modify", "id": "a", "changes": [
            {"field": "start", "before": 1.0, "after": 1.005}]}])]
        assert analyze_journal_events(tiny) == {}  # 5ms 偏移不归因


# ---------------------------------------------------------------------------
# V3 触发机制（D16）
# ---------------------------------------------------------------------------

class TestV3Trigger:
    def test_no_thresholds_configured_skips(self):
        ok, message = check_v3_trigger(100, 0.9, 0.0)
        assert ok is False
        assert "未配置" in message

    def test_thresholds_enforced(self):
        ok, _ = check_v3_trigger(120, 0.9, 0.05, min_samples=100, min_coverage=0.8, max_conflict_rate=0.1)
        assert ok is True
        not_ok, _ = check_v3_trigger(50, 0.9, 0.05, min_samples=100)
        assert not_ok is False
        conflict, _ = check_v3_trigger(120, 0.9, 0.2, max_conflict_rate=0.1)
        assert conflict is False


# ---------------------------------------------------------------------------
# 配套工具
# ---------------------------------------------------------------------------

class TestCompanions:
    def test_find_original_subtitle(self, original_srt):
        journal = original_srt.parent / "lesson01.journal.jsonl"
        journal.write_text("", encoding="utf-8")
        assert find_original_subtitle(journal) == original_srt

    def test_find_original_subtitle_missing(self, tmp_path):
        (tmp_path / "x.journal.jsonl").write_text("", encoding="utf-8")
        assert find_original_subtitle(tmp_path / "x.journal.jsonl") is None
