"""Tests for constrained LLM semantic sliding window."""

import pytest

from vocal_subtitle.mapping.semantic_fragments import PhysicalFragment
from vocal_subtitle.merging.semantic_window import (
    SemanticWindowInput,
    SemanticWindowOutput,
    build_semantic_windows,
    merge_windows,
    validate_window_output,
)


def _make_fragment(
    frag_id: str,
    word_ids: list,
    physical_start: float,
    physical_end: float,
    candidate_speaker: int | None = None,
    speaker_status: str = "unknown",
    hard_split_before: bool = False,
    hard_split_reason: str = "",
    pause_class: str = "",
) -> PhysicalFragment:
    return PhysicalFragment(
        id=frag_id,
        word_ids=list(word_ids),
        physical_start=physical_start,
        physical_end=physical_end,
        candidate_speaker=candidate_speaker,
        speaker_status=speaker_status,
        pause_class=pause_class,
        hard_split_before=hard_split_before,
        hard_split_reason=hard_split_reason,
    )


class TestBuildWindows:
    def test_empty_fragments(self):
        assert build_semantic_windows([]) == []

    def test_single_fragment_window(self):
        frag = _make_fragment("f-1", ["w1", "w2"], 0.0, 2.0)
        windows = build_semantic_windows([frag])

        assert len(windows) == 1
        assert windows[0].fragment_ids == ["f-1"]
        assert windows[0].word_ids == ["w1", "w2"]

    def test_hard_split_creates_new_window(self):
        f1 = _make_fragment("f-1", ["w1"], 0.0, 1.0)
        f2 = _make_fragment("f-2", ["w2"], 1.5, 2.5, hard_split_before=True, hard_split_reason="LongPause")
        f3 = _make_fragment("f-3", ["w3"], 2.6, 3.6)

        windows = build_semantic_windows([f1, f2, f3])

        # f1 goes in one window, f2+f3 in another (f2 has hard_split_before)
        # With overlap, f3 may appear in a third window — accept 2 or 3
        assert len(windows) >= 2
        assert windows[0].fragment_ids[0] == "f-1"
        # f-2 starts at index 0 of the second window
        assert "f-2" in windows[1].fragment_ids

    def test_soft_limit_splits(self):
        frags = [_make_fragment(f"f-{i}", [f"w{i}"], float(i), float(i + 1)) for i in range(15)]

        windows = build_semantic_windows(frags, max_fragments_per_window=5)

        assert len(windows) >= 3  # 15/5 = 3 windows minimum

    def test_window_input_serialization(self):
        win = SemanticWindowInput(
            window_id="sw-0001",
            fragment_ids=["f-1"],
            word_ids=["w1", "w2"],
            physical_range_start=0.0,
            physical_range_end=2.0,
            language="en",
            fragments=[],
        )

        payload = win.to_dict()
        assert payload["window_id"] == "sw-0001"
        assert payload["fragment_ids"] == ["f-1"]


class TestValidateOutput:
    def test_valid_output_passes(self):
        frag = _make_fragment("f-1", ["w1", "w2"], 0.0, 2.0)
        window = SemanticWindowInput(
            window_id="sw-1",
            fragment_ids=["f-1"],
            word_ids=["w1", "w2"],
            fragments=[{
                "fragment_id": "f-1",
                "word_ids": ["w1", "w2"],
                "candidate_speaker": None,
                "speaker_status": "unknown",
                "speaker_confidence": 0.0,
                "pause_class": "",
                "hard_split_before": False,
                "hard_split_reason": "",
                "genuine_overlap": False,
                "language": None,
            }],
        )
        output = SemanticWindowOutput(
            window_id="sw-1",
            groups=[["w1", "w2"]],
            normalized_text={},
            reason="test",
        )

        valid, errors = validate_window_output(output, window)

        assert valid is True
        assert len(errors) == 0

    def test_unknown_id_rejected(self):
        frag = _make_fragment("f-1", ["w1"], 0.0, 1.0)
        window = SemanticWindowInput(
            window_id="sw-1",
            fragment_ids=["f-1"],
            word_ids=["w1"],
            fragments=[],
        )
        output = SemanticWindowOutput(
            window_id="sw-1",
            groups=[["w1", "w-ghost"]],  # w-ghost doesn't exist
        )

        valid, errors = validate_window_output(output, window)

        assert valid is False
        assert any("unknown id" in e for e in errors)

    def test_duplicate_id_rejected(self):
        frag = _make_fragment("f-1", ["w1"], 0.0, 1.0)
        window = SemanticWindowInput(
            window_id="sw-1",
            fragment_ids=["f-1"],
            word_ids=["w1"],
            fragments=[],
        )
        output = SemanticWindowOutput(
            window_id="sw-1",
            groups=[["w1"], ["w1"]],  # w1 appears twice
        )

        valid, errors = validate_window_output(output, window)

        assert valid is False
        assert any("duplicate" in e for e in errors)


class TestMergeWindows:
    def test_merge_deduplicates_across_windows(self):
        w1 = SemanticWindowOutput(
            window_id="sw-1",
            groups=[["w1", "w2"]],
        )
        w2 = SemanticWindowOutput(
            window_id="sw-2",
            groups=[["w2", "w3"]],  # w2 overlaps with w1
        )

        merged = merge_windows([w1, w2])

        assert len(merged.groups) == 2
        # w2 should only appear in the first group
        all_ids = [fid for group in merged.groups for fid in group]
        assert all_ids == ["w1", "w2", "w3"]

    def test_empty_merge(self):
        merged = merge_windows([])
        assert merged.window_id == "empty"
