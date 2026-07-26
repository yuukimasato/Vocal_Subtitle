from vocal_subtitle.asr.base import WordTimestamp
from vocal_subtitle.mapping.event_constraints import can_merge_events
from vocal_subtitle.mapping.final_validator import validate_events
from vocal_subtitle.mapping.llm_guard import validate_llm_text
from vocal_subtitle.mapping.strict_segmenter import (
    StrictSegmentationConfig,
    normalize_event_text,
    repair_short_unknown_fragments,
    repair_unknown_runs,
    segment_events,
)
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.merging.llm_merge_engine import (
    LLMMergeEngine,
    MergeDecisionConfig,
    apply_frame_seamless_stitching,
)


def _event(
    index,
    text,
    start,
    end,
    words,
    *,
    clip="clip-a",
    speaker_id=0,
    warning=None,
):
    return SubtitleEvent(
        index=index,
        start=start,
        end=end,
        text=text,
        words=words,
        speaker_id=speaker_id,
        physical_start=0.0,
        physical_end=3.0,
        physical_spans=[
            {"physical_clip_id": clip, "start": start, "end": end}
        ],
        source_word_ids=[f"w{index}-{offset}" for offset, _ in enumerate(words)],
        alignment_warning=warning,
    )


def test_strict_segmenter_splits_on_sentence_end_at_word_boundary():
    event = _event(
        1,
        "Hello world. Next",
        0.0,
        2.0,
        [
            WordTimestamp("Hello", 0.1, 0.4),
            WordTimestamp("world.", 0.5, 0.9),
            WordTimestamp("Next", 1.4, 1.8),
        ],
    )

    result = segment_events(
        [event],
        config=StrictSegmentationConfig(max_duration=5.0),
    )

    assert [item.text for item in result.events] == ["Hello world.", "Next"]
    assert result.events[1].hard_split_before is True
    assert len(result.events[0].words) == 2
    assert len(result.events[1].words) == 1


def test_event_normalization_moves_leading_punctuation_to_previous_event():
    previous = _event(
        1,
        "差点摔我一跤",
        0.0,
        1.0,
        [WordTimestamp("差点摔我一跤", 0.0, 1.0)],
    )
    current = _event(
        2,
        "。 哈",
        1.02,
        1.4,
        [WordTimestamp("。 哈", 0.0, 0.38)],
    )

    result = normalize_event_text([previous, current])

    assert [item.text for item in result] == ["差点摔我一跤。", "哈"]


def test_event_normalization_joins_cjk_tail_fragment():
    previous = _event(
        1,
        "你中间歇了八",
        0.0,
        1.0,
        [WordTimestamp("你中间歇了八", 0.0, 1.0)],
    )
    current = _event(
        2,
        "回",
        1.04,
        1.2,
        [WordTimestamp("回", 0.0, 0.16)],
    )

    result = normalize_event_text([previous, current])

    assert [item.text for item in result] == ["你中间歇了八回"]
    assert result[0].end == 1.2


def test_unknown_tail_fragment_uses_three_way_speaker_consensus():
    previous = _event(
        1,
        "你中间歇了八",
        0.0,
        1.0,
        [WordTimestamp("你中间歇了八", 0.0, 1.0)],
        speaker_id=0,
    )
    current = _event(
        2,
        "回",
        1.04,
        1.2,
        [WordTimestamp("回", 0.0, 0.16)],
        speaker_id=None,
    )
    following = _event(
        3,
        "光吃煎饼",
        1.24,
        2.0,
        [WordTimestamp("光吃煎饼", 0.0, 0.76)],
        speaker_id=0,
    )

    result = repair_short_unknown_fragments([previous, current, following])

    assert [item.text for item in result] == ["你中间歇了八回", "光吃煎饼"]
    assert result[0].speaker_id == 0
    assert result[0].end == 1.2


def test_unknown_interjection_is_not_repaired_as_tail_fragment():
    previous = _event(
        1,
        "还带点金边,",
        0.0,
        1.0,
        [WordTimestamp("还带点金边,", 0.0, 1.0)],
        speaker_id=0,
    )
    current = _event(
        2,
        "咦,",
        1.04,
        1.2,
        [WordTimestamp("咦,", 0.0, 0.16)],
        speaker_id=None,
    )
    following = _event(
        3,
        "底下那层黑云",
        1.24,
        2.0,
        [WordTimestamp("底下那层黑云", 0.0, 0.76)],
        speaker_id=0,
    )

    result = repair_short_unknown_fragments([previous, current, following])

    assert [item.text for item in result] == ["还带点金边,", "咦,", "底下那层黑云"]
    assert result[1].speaker_id is None


def test_unknown_tail_with_different_neighbor_is_not_repaired():
    previous = _event(
        1,
        "你中间歇了八",
        0.0,
        1.0,
        [WordTimestamp("你中间歇了八", 0.0, 1.0)],
        speaker_id=0,
    )
    current = _event(
        2,
        "回",
        1.04,
        1.2,
        [WordTimestamp("回", 0.0, 0.16)],
        speaker_id=None,
    )
    following = _event(
        3,
        "你说得对",
        1.24,
        2.0,
        [WordTimestamp("你说得对", 0.0, 0.76)],
        speaker_id=1,
    )

    result = repair_short_unknown_fragments([previous, current, following])

    assert [item.text for item in result] == ["你中间歇了八", "回", "你说得对"]


def test_two_character_unknown_phrase_is_not_repaired_by_default():
    previous = _event(
        1,
        "我觉得",
        0.0,
        1.0,
        [WordTimestamp("我觉得", 0.0, 1.0)],
        speaker_id=0,
    )
    current = _event(
        2,
        "不是",
        1.04,
        1.2,
        [WordTimestamp("不是", 0.0, 0.16)],
        speaker_id=None,
    )
    following = _event(
        3,
        "这个意思",
        1.24,
        2.0,
        [WordTimestamp("这个意思", 0.0, 0.76)],
        speaker_id=0,
    )

    result = repair_short_unknown_fragments([previous, current, following])

    assert [item.text for item in result] == ["我觉得", "不是", "这个意思"]


def test_contiguous_unknown_run_inherits_same_neighbor_speaker_without_merging():
    previous = _event(
        1, "前", 0.0, 0.8, [WordTimestamp("前", 0.0, 0.8)], speaker_id=0
    )
    unknown_one = _event(
        2, "中", 0.82, 0.94, [WordTimestamp("中", 0.0, 0.12)], speaker_id=None
    )
    unknown_two = _event(
        3, "间", 0.96, 1.08, [WordTimestamp("间", 0.0, 0.12)], speaker_id=None
    )
    following = _event(
        4, "后", 1.10, 1.8, [WordTimestamp("后", 0.0, 0.7)], speaker_id=0
    )
    previous.speaker_label = "Speaker A"

    result, diagnostics = repair_unknown_runs(
        [previous, unknown_one, unknown_two, following]
    )

    assert [item.speaker_id for item in result] == [0, 0, 0, 0]
    assert [item.text for item in result] == ["前", "中", "间", "后"]
    assert diagnostics["repaired_run_count"] == 1
    assert diagnostics["repaired_event_count"] == 2
    assert unknown_one.speaker_source == "unknown_run_inheritance"
    assert unknown_two.speaker_repair_reason == "bounded_unknown_run_same_neighbor"


def test_unknown_run_is_not_repaired_across_physical_clip_boundary():
    previous = _event(
        1, "前", 0.0, 0.8, [WordTimestamp("前", 0.0, 0.8)], speaker_id=0
    )
    unknown = _event(
        2, "中", 0.82, 0.94, [WordTimestamp("中", 0.0, 0.12)],
        clip="clip-b", speaker_id=None,
    )
    following = _event(
        3, "后", 0.96, 1.7, [WordTimestamp("后", 0.0, 0.7)],
        clip="clip-a", speaker_id=0,
    )

    result, diagnostics = repair_unknown_runs([previous, unknown, following])

    assert result[1].speaker_id is None
    assert diagnostics["blocked_reasons"]["physical_owner_conflict"] == 1


def test_event_normalization_drops_punctuation_only_event():
    previous = _event(
        1,
        "你好",
        0.0,
        0.5,
        [WordTimestamp("你好", 0.0, 0.5)],
    )
    punctuation = _event(
        2,
        ",,,",
        0.55,
        0.7,
        [WordTimestamp(",,,", 0.0, 0.15)],
    )

    result = normalize_event_text([previous, punctuation])

    assert [item.text for item in result] == ["你好,,,"]


def test_strict_segmenter_splits_long_event_without_splitting_word():
    event = _event(
        1,
        "one two three four",
        0.0,
        2.0,
        [
            WordTimestamp("one", 0.1, 0.3),
            WordTimestamp("two", 0.4, 0.6),
            WordTimestamp("three", 0.8, 1.0),
            WordTimestamp("four", 1.3, 1.5),
        ],
    )

    result = segment_events(
        [event],
        config=StrictSegmentationConfig(
            max_duration=1.0,
            max_chars_latin=42,
        ),
    )

    assert len(result.events) == 2
    assert [word.word for word in result.events[0].words] == [
        "one",
        "two",
        "three",
    ]
    assert [word.word for word in result.events[1].words] == ["four"]


def test_strict_segmentation_is_idempotent_for_piece_relative_words():
    event = _event(
        1,
        "one two three four",
        0.0,
        2.0,
        [
            WordTimestamp("one", 0.1, 0.3),
            WordTimestamp("two", 0.4, 0.6),
            WordTimestamp("three", 0.8, 1.0),
            WordTimestamp("four", 1.3, 1.5),
        ],
    )
    config = StrictSegmentationConfig(max_duration=1.0, max_chars_latin=42)

    first = segment_events([event], config=config).events
    second = segment_events(first, config=config).events

    assert [(item.text, item.start, item.end) for item in second] == [
        (item.text, item.start, item.end) for item in first
    ]
    assert [word.start for word in second[1].words] == [0.0]


def test_latin_soft_punctuation_splits_clause_but_keeps_list_together():
    words = [
        WordTimestamp("It's", 0.0, 0.2),
        WordTimestamp("got", 0.3, 0.5),
        WordTimestamp("lots", 0.6, 0.8),
        WordTimestamp("of", 0.9, 1.0),
        WordTimestamp("glass", 1.1, 1.4),
        WordTimestamp("by", 1.5, 1.7),
        WordTimestamp("fold", 1.8, 2.1),
        WordTimestamp("doors,", 2.2, 2.6),
        WordTimestamp("lovely", 2.9, 3.2),
        WordTimestamp("roof", 3.3, 3.6),
        WordTimestamp("lantern,", 3.7, 4.1),
        WordTimestamp("but", 4.4, 4.6),
        WordTimestamp("for", 4.7, 4.9),
        WordTimestamp("the", 5.0, 5.1),
        WordTimestamp("winter.", 5.2, 5.6),
    ]
    event = _event(
        1,
        "It's got lots of glass by fold doors, lovely roof lantern, but for the winter.",
        0.0,
        6.0,
        words,
    )

    result = segment_events([event]).events

    assert [item.text for item in result] == [
        "It's got lots of glass by fold doors, lovely roof lantern,",
        "but for the winter.",
    ]


def test_soft_punctuation_does_not_change_cjk_segmentation():
    words = [
        WordTimestamp("中文", 0.0, 0.4),
        WordTimestamp("内容，", 0.4, 0.8),
        WordTimestamp("后续", 0.9, 1.2),
        WordTimestamp("内容", 1.2, 1.5),
    ]
    event = _event(1, "中文内容，后续内容", 0.0, 2.0, words)

    result = segment_events([event]).events

    assert len(result) == 1


def test_event_constraints_block_physical_and_hard_boundaries():
    left = _event(
        1,
        "left",
        0.0,
        0.4,
        [WordTimestamp("left", 0.1, 0.3)],
    )
    right = _event(
        2,
        "right",
        0.45,
        0.8,
        [WordTimestamp("right", 0.05, 0.2)],
        clip="clip-b",
    )
    assert can_merge_events(left, right, max_gap=0.2) is False

    right.physical_spans = left.physical_spans
    right.hard_split_before = True
    assert can_merge_events(left, right, max_gap=0.2) is False


def test_final_validator_clamps_envelope_and_rejects_outside_event():
    inside = _event(
        1,
        "inside",
        0.1,
        1.2,
        [WordTimestamp("inside", 0.1, 0.4)],
    )
    inside.physical_start = 0.2
    inside.physical_end = 1.0
    outside = _event(
        2,
        "outside",
        2.5,
        2.8,
        [WordTimestamp("outside", 0.1, 0.2)],
    )
    outside.physical_start = 0.0
    outside.physical_end = 2.0

    result = validate_events([inside, outside], audio_duration=3.0)

    assert len(result.events) == 1
    assert result.events[0].start == 0.2
    assert result.events[0].end == 1.0
    assert result.diagnostics["removed_count"] == 1


def test_frame_stitching_respects_physical_owner():
    left = _event(
        1,
        "left",
        0.0,
        0.4,
        [WordTimestamp("left", 0.1, 0.3)],
    )
    right = _event(
        2,
        "right",
        0.43,
        0.8,
        [WordTimestamp("right", 0.05, 0.2)],
        clip="clip-b",
    )

    apply_frame_seamless_stitching([left, right], max_stitch_gap=0.1)

    assert left.end == 0.4


def test_llm_fast_merge_respects_physical_owner():
    engine = LLMMergeEngine(
        MergeDecisionConfig(fast_merge_max_gap=0.2, llm_tier="rule_only")
    )
    fragments = [
        {
            "id": 1,
            "start": 0.0,
            "end": 0.4,
            "text": "left",
            "speaker": "A",
            "physical_spans": [
                {"physical_clip_id": "clip-a", "start": 0.0, "end": 0.4}
            ],
            "gap_to_next_sec": 0.03,
        },
        {
            "id": 2,
            "start": 0.43,
            "end": 0.8,
            "text": "right",
            "speaker": "A",
            "physical_spans": [
                {"physical_clip_id": "clip-b", "start": 0.43, "end": 0.8}
            ],
            "gap_to_next_sec": None,
        },
    ]

    assert len(engine._apply_fast_merges(fragments)) == 2


def test_llm_text_guard_rejects_translation_and_cross_event_copy():
    accepted, reason = validate_llm_text(
        "Hello world",
        "你好世界",
    )
    assert accepted is False
    assert reason == "script_changed"

    accepted, reason = validate_llm_text(
        "Hello",
        "Hello world",
        peer_texts=["world"],
        min_similarity=0.5,
        max_length_ratio=3.0,
    )
    assert accepted is False
    assert reason == "cross_event_text"
