"""时间轴单一 Owner 不变量(2026-09-15 重构计划 Task 5)。

- ``validate_timeline`` 为只读校验:相邻相等边界合法、越界延长必须报告;
- ``genuine_overlap`` 事件允许重叠(真实重叠对白),不计违规;
- 修复仍由 ``enforce_non_overlap`` 单一 Owner 完成,每次修复有 trace;
- 校验不修改事件(只读),导出阶段只做验证。
"""

from vocal_subtitle.mapping.final_validator import enforce_non_overlap
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.mapping.timeline_result import (
    TimelineIssue,
    validate_timeline,
)


def _event(start, end, text="x", genuine_overlap=False):
    event = SubtitleEvent(
        index=1,
        start=start,
        end=end,
        text=text,
        physical_start=start,
        physical_end=end,
    )
    if genuine_overlap:
        event.genuine_overlap = True
    return event


def test_equal_boundary_neighbors_are_valid():
    events = [_event(0.0, 1.0), _event(1.0, 2.0), _event(2.0, 3.0)]

    report = validate_timeline(events)

    assert report.ok is True
    assert report.issues == ()


def test_extension_beyond_next_start_is_flagged():
    events = [_event(0.0, 1.5), _event(1.0, 2.0)]

    report = validate_timeline(events)

    assert report.ok is False
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.kind == "overlap"
    assert issue.first_index == 0
    assert issue.second_index == 1
    assert issue.overlap_seconds > 0


def test_genuine_overlap_events_are_allowed():
    events = [_event(0.0, 1.5), _event(1.0, 2.0, genuine_overlap=True)]

    report = validate_timeline(events)

    assert report.ok is True
    assert report.issues == ()


def test_validation_is_read_only():
    events = [_event(0.0, 1.5), _event(1.0, 2.0)]
    before = [(item.start, item.end, item.text) for item in events]

    report = validate_timeline(events)

    after = [(item.start, item.end, item.text) for item in events]
    assert before == after
    assert report.ok is False  # 只报告,不修复


def test_enforce_non_overlap_repairs_and_records_trace():
    events = [_event(0.0, 1.5), _event(1.0, 2.0)]

    repaired, diagnostics = enforce_non_overlap(events, source="timeline_invariant")

    assert diagnostics["overlap_count"] >= 1
    ordered = sorted(repaired, key=lambda item: (item.start, item.end))
    for previous, current in zip(ordered, ordered[1:]):
        assert current.start >= previous.end - 1e-9
    # 修复后只读校验必须通过(单一 Owner 修复 + 只读验证闭环)。
    assert validate_timeline(repaired).ok is True


def test_repair_preserves_revision_trace():
    events = [_event(0.0, 1.5), _event(1.0, 2.0)]
    events[0].revision_trace.append({"stage": "finalize", "note": "kept"})

    repaired, _diagnostics = enforce_non_overlap(events, source="trace_check")

    traces = [
        item
        for event in repaired
        for item in (getattr(event, "revision_trace", ()) or ())
    ]
    assert any(item.get("stage") == "finalize" for item in traces), (
        "修复不得丢弃既有 revision trace"
    )


def test_timeline_issue_is_frozen_value():
    issue = TimelineIssue(
        kind="overlap", first_index=0, second_index=1, overlap_seconds=0.5,
    )

    assert issue.overlap_seconds == 0.5
    assert issue.to_dict()["kind"] == "overlap"
