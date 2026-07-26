"""全局 speaker turn 与物理语音区间融合测试。"""

from dataclasses import dataclass

from vocal_subtitle.diarization.base import SpeakerTurn
from vocal_subtitle.diarization.turn_reconciler import (
    merge_same_speaker_spans,
    reconcile_regions,
    normalize_turns,
    shift_turns,
    split_event_intervals,
)


@dataclass
class Region:
    start: float
    end: float


def test_one_physical_region_is_split_at_speaker_change():
    spans = reconcile_regions(
        [Region(0.0, 4.0)],
        [
            SpeakerTurn(0.0, 2.0, 0),
            SpeakerTurn(2.0, 4.0, 1),
        ],
        boundary_collar_ms=0,
    )

    assert [(s.start, s.end, s.speaker_id) for s in spans] == [
        (0.0, 2.0, 0),
        (2.0, 4.0, 1),
    ]


def test_global_identity_is_stable_for_complex_alternation():
    turns = [
        SpeakerTurn(0.0, 1.0, 0),
        SpeakerTurn(1.0, 2.0, 1),
        SpeakerTurn(2.0, 3.0, 0),
        SpeakerTurn(3.0, 4.0, 0),
        SpeakerTurn(4.0, 5.0, 1),
        SpeakerTurn(5.0, 6.0, 2),
        SpeakerTurn(6.0, 7.0, 1),
    ]
    spans = reconcile_regions(
        [Region(0.0, 7.0)], turns, boundary_collar_ms=0,
    )

    assert [s.speaker_id for s in spans] == [0, 1, 0, 0, 1, 2, 1]


def test_gap_does_not_create_or_change_identity():
    spans = [
        *reconcile_regions(
            [Region(0.0, 1.0)], [SpeakerTurn(0.0, 1.0, 0)],
            boundary_collar_ms=0,
        ),
        *reconcile_regions(
            [Region(3.0, 4.0)], [SpeakerTurn(3.0, 4.0, 0)],
            boundary_collar_ms=0,
        ),
    ]

    merged = merge_same_speaker_spans(spans, max_gap=3.0)
    assert len(merged) == 1
    assert merged[0].speaker_id == 0


def test_different_speakers_never_merge_even_without_gap():
    spans = [
        *reconcile_regions(
            [Region(0.0, 1.0)], [SpeakerTurn(0.0, 1.0, 0)],
            boundary_collar_ms=0,
        ),
        *reconcile_regions(
            [Region(1.0, 2.0)], [SpeakerTurn(1.0, 2.0, 1)],
            boundary_collar_ms=0,
        ),
    ]

    assert len(merge_same_speaker_spans(spans, max_gap=10.0)) == 2


def test_uncovered_physical_audio_is_explicitly_unknown():
    spans = reconcile_regions(
        [Region(0.0, 3.0)],
        [SpeakerTurn(1.0, 2.0, 0)],
        boundary_collar_ms=0,
    )

    assert [(s.start, s.end, s.speaker_id) for s in spans] == [
        (0.0, 1.0, None),
        (1.0, 2.0, 0),
        (2.0, 3.0, None),
    ]


def test_shift_turns_uses_local_chunk_coordinates():
    shifted = shift_turns(
        [SpeakerTurn(10.0, 11.0, 2)], offset=10.0, duration=2.0,
    )
    assert [(turn.start, turn.end, turn.speaker_id) for turn in shifted] == [
        (0.0, 1.0, 2),
    ]


def test_split_event_intervals_keeps_unknown_gap():
    pieces = split_event_intervals(
        0.0,
        4.0,
        [SpeakerTurn(0.0, 1.5, 0), SpeakerTurn(2.0, 4.0, 1)],
    )
    assert pieces == [
        (0.0, 1.5, 0),
        (1.5, 2.0, None),
        (2.0, 4.0, 1),
    ]


def test_normalize_turns_clips_invalid_and_short_turns():
    turns = normalize_turns(
        [
            SpeakerTurn(-1.0, 0.1, 0),
            SpeakerTurn(0.5, 2.0, 1),
            SpeakerTurn(1.0, 1.0, 2),
            SpeakerTurn(0.0, 0.1, 3),
        ],
        duration=1.5,
        min_duration=0.2,
    )

    assert [(turn.start, turn.end, turn.speaker_id) for turn in turns] == [
        (0.5, 1.5, 1),
    ]
