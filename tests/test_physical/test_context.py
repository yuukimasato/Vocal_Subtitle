import pytest

from vocal_subtitle.physical.context import build_context_windows
from vocal_subtitle.physical.timeline import PhysicalTimeline


def test_one_context_window_per_clip_with_owner_and_stable_id():
    timeline = PhysicalTimeline(10.0)
    timeline.add_clip(0.0, 4.0, clip_id="clip-a")
    timeline.add_clip(6.0, 10.0, clip_id="clip-b")

    windows = build_context_windows(timeline, 1.0, 1.0)

    assert [window.id for window in windows] == [
        "ctx:clip-a:l1000:r1000",
        "ctx:clip-b:l1000:r1000",
    ]
    assert (windows[0].start, windows[0].end) == (0.0, 5.0)
    assert (windows[1].start, windows[1].end) == (5.0, 10.0)
    assert windows[0].physical_clip_id == "clip-a"
    assert windows[1].physical_clip_id == "clip-b"


def test_context_can_overlap_but_does_not_change_timeline():
    timeline = PhysicalTimeline(12.0)
    timeline.add_clip(2.0, 5.0, clip_id="a")
    timeline.add_clip(7.0, 10.0, clip_id="b")
    before = timeline.to_dict()

    windows = build_context_windows(timeline, 3.0, 3.0, id_prefix="asr")

    assert (windows[0].start, windows[0].end) == (0.0, 8.0)
    assert (windows[1].start, windows[1].end) == (4.0, 12.0)
    assert timeline.to_dict() == before


@pytest.mark.parametrize("left,right", [(-1.0, 1.0), (1.0, -1.0), (float("inf"), 1.0)])
def test_invalid_context_values_are_rejected(left, right):
    timeline = PhysicalTimeline.from_duration(5.0)
    with pytest.raises(ValueError):
        build_context_windows(timeline, left, right)
