"""模型加载器必须保持离线默认，避免语义降级阻塞主流程。"""

from vocal_subtitle.utils import model_loader


def test_missing_semantic_model_does_not_download_by_default(monkeypatch):
    monkeypatch.setattr(model_loader, "is_model_cached", lambda _name: False)

    def fail_if_downloaded(*_args, **_kwargs):
        raise AssertionError("offline default must not start a model download")

    monkeypatch.setattr(model_loader, "_download_model", fail_if_downloaded)
    assert model_loader.load_sentence_transformer("missing-model") is None


def test_semantic_model_download_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(model_loader, "is_model_cached", lambda _name: False)
    sentinel = object()
    monkeypatch.setattr(model_loader, "_download_model", lambda *_args: sentinel)

    assert model_loader.load_sentence_transformer("missing-model") is None
    assert (
        model_loader.load_sentence_transformer(
            "missing-model",
            allow_download=True,
        )
        is sentinel
    )


def test_semantic_model_download_env_flag_is_opt_in(monkeypatch):
    monkeypatch.setenv(model_loader.ALLOW_DOWNLOAD_ENV, "1")
    monkeypatch.setattr(model_loader, "is_model_cached", lambda _name: False)
    sentinel = object()
    monkeypatch.setattr(model_loader, "_download_model", lambda *_args: sentinel)

    assert model_loader.load_sentence_transformer("missing-model") is sentinel
