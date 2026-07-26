"""pyannote backend contract tests without loading the optional model."""

from pathlib import Path

from vocal_subtitle.diarization.pyannote_engine import PyannoteDiarizationEngine


class _Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _Annotation:
    def itertracks(self, yield_label=False):
        assert yield_label is True
        yield _Segment(0.0, 1.0), 0, "SPEAKER_01"
        yield _Segment(1.0, 2.0), 0, "SPEAKER_00"


def test_annotation_tracks_are_converted_without_model_import():
    result = PyannoteDiarizationEngine._annotation_to_turns(_Annotation())

    assert result == [
        (0.0, 1.0, "SPEAKER_01"),
        (1.0, 2.0, "SPEAKER_00"),
    ]


def test_overlap_duration_only_counts_different_speakers():
    turns = [
        (0.0, 2.0, "A"),
        (1.0, 3.0, "B"),
        (1.5, 2.5, "A"),
    ]

    assert PyannoteDiarizationEngine._overlap_duration(turns) == 1.5


def test_local_model_config_prefers_huggingface_snapshot(tmp_path, monkeypatch):
    model_dir = (
        tmp_path
        / "hub"
        / "models--pyannote--speaker-diarization-community-1"
    )
    revision = "local-revision"
    snapshot = model_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (model_dir / "refs").mkdir()
    (model_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    (snapshot / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    engine = PyannoteDiarizationEngine()

    assert engine._local_model_config() == snapshot / "config.yaml"


def test_load_model_uses_local_snapshot_before_huggingface(monkeypatch, tmp_path):
    model_dir = (
        tmp_path
        / "hub"
        / "models--pyannote--speaker-diarization-community-1"
    )
    revision = "local-revision"
    snapshot = model_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (model_dir / "refs").mkdir()
    (model_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    config_path = snapshot / "config.yaml"
    config_path.write_text("pipeline: {}\n", encoding="utf-8")

    import pyannote.audio

    calls = []

    def fake_from_pretrained(model_ref, **kwargs):
        calls.append(model_ref)
        return object()

    monkeypatch.setattr(
        pyannote.audio.Pipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )

    engine = PyannoteDiarizationEngine(
        model_ref="pyannote/speaker-diarization-community-1",
        cache_dir=tmp_path,
        offline=True,
    )
    engine.load_model()

    assert calls == [str(config_path)]


def test_diarization_does_not_send_default_speaker_bounds(monkeypatch):
    captured = {}

    class FakePipeline:
        def __call__(self, source, **kwargs):
            captured.update(kwargs)
            return _Annotation()

    engine = PyannoteDiarizationEngine()
    engine._pipeline = FakePipeline()
    monkeypatch.setattr(engine, "load_model", lambda: None)

    engine.diarize(audio=[0.0, 0.0], sample_rate=1)

    assert captured == {}
