from pathlib import Path

from vocal_subtitle.asr import funasr_manager as manager


def _cached_model(root: Path) -> Path:
    path = root / manager.DEFAULT_FUNASR_MODEL
    path.mkdir(parents=True)
    (path / "configuration.json").write_text("{}", encoding="utf-8")
    return path


def test_normalize_model_maps_generic_whisper_names():
    assert manager.normalize_model_id("large-v3") == manager.DEFAULT_FUNASR_MODEL
    assert manager.normalize_model_id("") == manager.DEFAULT_FUNASR_MODEL


def test_status_is_local_only_and_reports_cached_model(tmp_path, monkeypatch):
    cached = _cached_model(tmp_path)
    monkeypatch.setattr(manager, "funasr_package_installed", lambda: True)

    status = manager.funasr_status("large-v3", cache_dir=tmp_path)

    assert status["ready"] is True
    assert status["model"] == manager.DEFAULT_FUNASR_MODEL
    assert status["model_path"] == str(cached)


def test_prepare_uses_local_cache_without_download(tmp_path, monkeypatch):
    cached = _cached_model(tmp_path)
    monkeypatch.setattr(manager, "funasr_package_installed", lambda: True)
    monkeypatch.setattr(
        manager,
        "_download_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")),
    )

    result = manager.ensure_funasr_ready("large-v3", cache_dir=tmp_path)

    assert result["ready"] is True
    assert result["model_path"] == str(cached)


def test_prepare_installs_missing_package_then_downloads_model(tmp_path, monkeypatch):
    package_states = iter((False, True))
    monkeypatch.setattr(manager, "funasr_package_installed", lambda: next(package_states))
    installed = []
    monkeypatch.setattr(manager, "_install_funasr_package", lambda: installed.append(True))
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    (downloaded / "configuration.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(manager, "_download_model", lambda *args, **kwargs: downloaded)

    result = manager.ensure_funasr_ready("medium", cache_dir=tmp_path)

    assert installed == [True]
    assert result["ready"] is True
    assert result["model"] == manager.DEFAULT_FUNASR_MODEL
