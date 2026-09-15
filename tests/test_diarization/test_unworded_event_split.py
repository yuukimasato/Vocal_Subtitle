"""无词级时间戳事件的 speaker 切分安全(高精度方案 Task 5 / 方案 4.4)。

- 有可靠词级时间时,跨 turn 事件按词归属切分为多个 speaker;
- 无词级时间戳(或词时间非法)时,保留整段单事件,标记
  ``speaker_split_degraded`` 并归属主说话人——绝不把整句文本压进
  首个说话人片段,也不伪造词级时间。
"""

from vocal_subtitle.diarization.base import SpeakerTurn
from vocal_subtitle.diarization.early_turns import assign_event_speakers
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


def _turn(speaker_id, start, end):
    return SpeakerTurn(speaker_id=speaker_id, start=start, end=end)


def _word(text, start, end):
    return type("W", (), {"word": text, "start": start, "end": end})()


def _worded_event():
    return SubtitleEvent(
        index=1,
        start=1.0,
        end=3.0,
        text="hello world",
        words=[_word("hello", 1.0, 1.9), _word("world", 2.1, 3.0)],
    )


def test_worded_event_splits_into_two_speakers():
    event = _worded_event()
    turns = [_turn(0, 0.0, 2.0), _turn(1, 2.0, 4.0)]

    output, diagnostics = assign_event_speakers(
        [event],
        turns,
        word_split=True,
        min_part_duration=0.05,
    )

    assert len(output) == 2
    speakers = {item.speaker_id for item in output}
    assert speakers == {0, 1}
    assert diagnostics["local_split_count"] == 1
    assert all(not item.speaker_split_degraded for item in output)
    # 每个片段文本来自各自的词,不出现整句重复。
    assert [item.text for item in output] == ["hello", "world"]


def test_unworded_event_stays_whole_and_marks_degraded():
    event = SubtitleEvent(
        index=1,
        start=1.0,
        end=3.0,
        text="这是一句很长的话不应该被压进小片段",
        words=[],
    )
    # 第二个说话人片段只有 60ms —— 旧 fallback 会把整句压进这里。
    turns = [_turn(0, 0.0, 2.9), _turn(1, 2.9, 2.96), _turn(0, 2.96, 4.0)]

    output, diagnostics = assign_event_speakers(
        [event],
        turns,
        word_split=True,
        min_part_duration=0.05,
    )

    assert len(output) == 1
    kept = output[0]
    assert kept.text == "这是一句很长的话不应该被压进小片段"
    assert kept.speaker_split_degraded is True
    assert kept.time_source == "segment_boundary"
    # 归属主说话人(区间重叠最长的 turn),而不是第一个片段的 speaker。
    assert kept.speaker_id == 0
    assert diagnostics["speaker_split_degraded_count"] == 1


def test_invalid_word_times_are_not_fabricated_into_splits():
    event = SubtitleEvent(
        index=1,
        start=1.0,
        end=3.0,
        text="bad word times",
        words=[
            _word("bad", 1.5, 1.2),  # 倒序
            _word("times", None, 2.0),  # 缺失
        ],
    )
    turns = [_turn(0, 0.0, 2.0), _turn(1, 2.0, 4.0)]

    output, diagnostics = assign_event_speakers(
        [event],
        turns,
        word_split=True,
        min_part_duration=0.05,
    )

    assert len(output) == 1
    kept = output[0]
    assert kept.text == "bad word times"
    assert kept.speaker_split_degraded is True
    assert kept.words[0].start == 1.5  # 原始词数据不被改写或伪造
    assert kept.words[1].start is None
    assert diagnostics["speaker_split_degraded_count"] == 1


def test_word_split_disabled_keeps_event_whole_by_config():
    event = _worded_event()
    turns = [_turn(0, 0.0, 2.0), _turn(1, 2.0, 4.0)]

    output, _diagnostics = assign_event_speakers([event], turns, word_split=False)

    assert len(output) == 1
    assert output[0].speaker_split_degraded is False
