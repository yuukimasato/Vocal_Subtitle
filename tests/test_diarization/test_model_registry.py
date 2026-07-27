"""Speaker model registry tests are local-only."""

import os

from vocal_subtitle.diarization.model_registry import (
    is_model_cached,
    model_status,
)
from vocal_subtitle.diarization import model_registry
from vocal_subtitle.diarization.speaker_embedding import is_huggingface_model_cached


def test_speechbrain_cache_requires_weights_and_hparams(tmp_path):
    savedir = tmp_path / "speechbrain_spkrec-ecapa-voxceleb"
    savedir.mkdir()
    (savedir / "hyperparams.yaml").write_text("x: 1", encoding="utf-8")
    assert not is_model_cached("speechbrain-ecapa", tmp_path)
    (savedir / "embedding_model.ckpt").write_bytes(b"weights" * 100)
    assert is_model_cached("speechbrain-ecapa", tmp_path)


def test_global_model_status_is_non_networking(tmp_path):
    status = model_status("community-1", tmp_path)
    assert status["cached"] is False
    assert status["status"] == "not_cached"
    assert status["requires_token"] is True
    assert status["cache_integrity"] == "incomplete_or_missing"


def test_huggingface_cache_requires_model_file(tmp_path):
    snapshot = (
        tmp_path
        / "hub"
        / "models--example--model"
        / "snapshots"
        / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    assert not is_huggingface_model_cached("example/model", tmp_path)

    (snapshot / "pytorch_model.bin").write_bytes(b"weights")
    assert is_huggingface_model_cached("example/model", tmp_path)


def test_download_temporarily_enables_huggingface_network(monkeypatch, tmp_path):
    observed = {}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_registry, "is_model_cached", lambda *args, **kwargs: False)

    def fake_download(**kwargs):
        observed["offline"] = os.environ.get("HF_HUB_OFFLINE")
        observed["token"] = kwargs["token"]

    monkeypatch.setattr(model_registry, "_download_snapshot", fake_download)
    monkeypatch.setattr(
        model_registry,
        "model_status",
        lambda *args, **kwargs: {"cached": True, "status": "ready"},
    )
    model_registry.download_model("community-1", token="hf_test", cache_dir=tmp_path)

    assert observed == {"offline": "0", "token": "hf_test"}
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_download_uses_encrypted_token_when_request_token_is_missing(monkeypatch, tmp_path):
    observed = {}

    monkeypatch.setattr(model_registry, "is_model_cached", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        model_registry,
        "_download_snapshot",
        lambda **kwargs: observed.update(token=kwargs["token"]),
    )
    monkeypatch.setattr(
        model_registry,
        "model_status",
        lambda *args, **kwargs: {"cached": True, "status": "ready"},
    )
    monkeypatch.setattr(
        "vocal_subtitle.utils.hf_token_store.load_hf_token",
        lambda: "hf_saved",
    )

    model_registry.download_model("community-1", cache_dir=tmp_path)

    assert observed["token"] == "hf_saved"


def test_download_rejects_incomplete_snapshot(monkeypatch, tmp_path):
    snapshot = (
        tmp_path
        / "hub"
        / "models--pyannote--speaker-diarization-community-1"
        / "snapshots"
        / "revision"
    )

    def fake_download(**kwargs):
        snapshot.mkdir(parents=True)
        (snapshot / "config.yaml").write_text("model: incomplete", encoding="utf-8")

    monkeypatch.setattr(model_registry, "_download_snapshot", fake_download)

    try:
        model_registry.download_model("community-1", token="hf_test", cache_dir=tmp_path)
    except RuntimeError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete model download should fail")


def test_snapshot_download_honors_hf_endpoint(monkeypatch):
    observed = {}

    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: observed.update(kwargs),
    )

    model_registry._download_snapshot(repo_id="example/model", token="hf_test")

    assert observed["endpoint"] == "https://hf-mirror.example"
    assert observed["etag_timeout"] == 30
