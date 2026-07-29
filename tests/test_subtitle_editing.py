import pytest

from vocal_subtitle.webui.subtitle_editing import (
    SubtitleBatchEditError,
    apply_batch_edit,
)


def _events():
    return [
        {"index": 1, "start": 0.0, "end": 1.0, "text": "第一句", "speaker_id": 0, "speaker_label": "主持人"},
        {"index": 2, "start": 1.2, "end": 2.0, "text": "第二句", "speaker_id": None, "speaker_label": None},
        {"index": 3, "start": 2.2, "end": 3.0, "text": "第三句", "speaker_id": 1, "speaker_label": "嘉宾"},
        {"index": 4, "start": 3.2, "end": 4.0, "text": "第四句", "speaker_id": 0, "speaker_label": "主持人"},
    ]


def test_speaker_batch_allows_non_contiguous_selection_and_keeps_final_text():
    events = _events()
    result = apply_batch_edit(
        events,
        action="speaker",
        indexes=[1, 4],
        speaker_id=1,
        speaker_label="嘉宾",
    )

    assert [event["speaker_id"] for event in result] == [1, None, 1, 1]
    assert [event["text"] for event in result] == ["第一句", "第二句", "第三句", "第四句"]
    assert events[0]["speaker_id"] == 0


def test_new_speaker_gets_the_smallest_unused_id():
    result = apply_batch_edit(
        _events(),
        action="speaker",
        indexes=[2],
        speaker_label="新说话人",
    )

    assert result[1]["speaker_id"] == 2
    assert result[1]["speaker_label"] == "新说话人"


def test_merging_uses_real_newline_and_reindexes_final_events():
    result = apply_batch_edit(_events(), action="merge", indexes=[2, 3], separator="newline")

    assert len(result) == 3
    assert result[1]["index"] == 2
    assert result[1]["start"] == 1.2
    assert result[1]["end"] == 3.0
    assert result[1]["text"] == "第二句\n第三句"
    assert result[1]["original_text"] is None


def test_merging_can_use_spaces():
    result = apply_batch_edit(_events(), action="merge", indexes=[2, 3], separator="space")

    assert result[1]["text"] == "第二句 第三句"


@pytest.mark.parametrize("indexes", ([1], [1, 3]))
def test_merging_rejects_less_than_two_or_non_contiguous_selection(indexes):
    with pytest.raises(SubtitleBatchEditError):
        apply_batch_edit(_events(), action="merge", indexes=indexes)


def test_empty_speaker_is_rejected():
    with pytest.raises(SubtitleBatchEditError):
        apply_batch_edit(_events(), action="speaker", indexes=[1], speaker_label="  ")
