"""Tests for the real-material quality benchmark without loading ASR models."""

from pathlib import Path

import pytest

from scripts.run_quality_benchmark import _reference_content_coverage, load_manifest
from scripts.compare_timeline import TimelineEvent, _ass_time_to_seconds, match_events


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0:00:00.00", 0.0),
        ("0:00:05.01", 5.01),
        ("0:00:05.80", 5.80),
        ("0:00:05.999", 5.999),
        ("1:02:03", 3723.0),
    ],
)
def test_ass_time_parser_preserves_decimal_fraction(value, expected):
    assert _ass_time_to_seconds(value) == pytest.approx(expected)


def test_event_matching_does_not_cascade_after_one_missing_event():
    auto = [
        TimelineEvent(1, 0.0, 1.0, "第一句"),
        TimelineEvent(2, 2.0, 3.0, "完全不同的内容"),
        TimelineEvent(3, 4.0, 5.0, "第三句"),
    ]
    ground_truth = [
        TimelineEvent(1, 0.0, 1.0, "第一句"),
        TimelineEvent(2, 2.0, 3.0, "唉"),
        TimelineEvent(3, 4.0, 5.0, "第三句"),
    ]

    matches = match_events(auto, ground_truth)

    assert [(left.text, right.text) for left, right in matches] == [
        ("第一句", "第一句"),
        ("第三句", "第三句"),
    ]


def test_reference_content_coverage_allows_extra_filler_words():
    ground_truth = [TimelineEvent(1, 0.0, 1.0, "你好世界")]
    auto = [TimelineEvent(1, 0.0, 0.5, "嗯,你好"), TimelineEvent(2, 0.5, 1.2, "世界")]

    assert _reference_content_coverage(auto, ground_truth) == 1.0


def test_reference_content_coverage_detects_missing_content():
    ground_truth = [TimelineEvent(1, 0.0, 1.0, "你好世界")]
    auto = [TimelineEvent(1, 0.0, 0.5, "你好")]

    assert _reference_content_coverage(auto, ground_truth) == 0.5


def test_quality_manifest_contains_existing_fixture_pairs():
    root = Path(__file__).resolve().parents[1]
    scenes = load_manifest(root / "test/quality_manifest.yaml", root)

    assert len(scenes) == 8
    assert {scene["category"] for scene in scenes} == {
        "single_speaker",
        "multi_speaker",
    }
    assert all(scene["audio_path"].is_file() for scene in scenes)
    assert sum(scene["ground_truth_path"] is not None for scene in scenes) == 6
    assert all(
        scene["ground_truth_path"] is None
        or scene["ground_truth_path"].is_file()
        for scene in scenes
    )
    assert all(scene.get("language") in {"zh", "en"} for scene in scenes)
    assert all(scene.get("speaker_count", 0) >= 1 for scene in scenes)
    assert any(
        "overlap_or_interruption" in scene.get("tags", []) for scene in scenes
    )
    assert {"中文朗读双人", "巧乐兹"} <= {scene["name"] for scene in scenes}


def test_quality_manifest_rejects_duplicate_names(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "scenes:\n"
        "  - name: duplicate\n"
        "    audio: test/181人声.wav\n"
        "    ground_truth: test/181-人工修正.ass\n"
        "    category: single_speaker\n"
        "  - name: duplicate\n"
        "    audio: test/181人声.wav\n"
        "    ground_truth: test/181-人工修正.ass\n"
        "    category: single_speaker\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(manifest, Path(__file__).resolve().parents[1])
