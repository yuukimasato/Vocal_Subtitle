import pytest

from vocal_subtitle.physical.timeline import PhysicalTimeline


def test_default_clip_and_evidence_are_absolute_and_serializable():
    timeline = PhysicalTimeline.from_duration(10.0)
    evidence = timeline.add_evidence(1.0, 3.0, "silero", confidence=0.8)

    assert timeline.physical_clips[0].id == "clip-000001"
    assert evidence.physical_clip_id == "clip-000001"
    restored = PhysicalTimeline.from_dict(timeline.to_dict())
    assert restored.to_dict() == timeline.to_dict()


def test_evidence_is_clipped_to_owner_clip_without_changing_clip():
    timeline = PhysicalTimeline(10.0)
    clip = timeline.add_clip(2.0, 5.0)
    evidence = timeline.add_evidence(1.0, 6.0, "ffmpeg", physical_clip_id=clip.id)

    assert (evidence.start, evidence.end) == (2.0, 5.0)
    assert evidence.metadata["clipped"] is True
    assert (clip.start, clip.end) == (2.0, 5.0)


def test_overlapping_clips_and_ambiguous_evidence_are_rejected():
    timeline = PhysicalTimeline(10.0)
    timeline.add_clip(0.0, 4.0, clip_id="a")
    with pytest.raises(ValueError):
        timeline.add_clip(3.0, 5.0, clip_id="b")

    timeline.add_clip(5.0, 8.0, clip_id="b")
    with pytest.raises(ValueError):
        timeline.add_evidence(3.5, 5.5, "vad")


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError):
        PhysicalTimeline.from_dict({"schema_version": "other", "duration": 1.0})
