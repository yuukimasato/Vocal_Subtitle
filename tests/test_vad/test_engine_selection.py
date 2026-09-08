"""VAD 引擎选择：--no-torch / 无 GPU 环境下 silero 自动降级为 WebRTC"""

import logging

import pytest

from vocal_subtitle.application import pipeline_services
from vocal_subtitle.application.pipeline_services import PipelineServices
from vocal_subtitle.config import PipelineConfig


def _services(engine: str) -> PipelineServices:
    config = PipelineConfig()
    config.vad.engine = engine
    return PipelineServices(config)


def test_silero_falls_back_to_webrtc_without_torch(monkeypatch, caplog):
    """默认配置 silero + 无 torch（--no-torch 安装）不应直接 ImportError"""
    monkeypatch.setattr(pipeline_services, "_torch_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        engine = _services("silero").get_vad_engine()
    assert engine.name == "webrtc"
    assert any("降级" in record.getMessage() for record in caplog.records)


def test_silero_used_when_torch_available(monkeypatch):
    monkeypatch.setattr(pipeline_services, "_torch_available", lambda: True)
    assert _services("silero").get_vad_engine().name == "silero"


def test_explicit_webrtc_not_affected_by_torch_probe(monkeypatch):
    monkeypatch.setattr(pipeline_services, "_torch_available", lambda: False)
    assert _services("webrtc").get_vad_engine().name == "webrtc"


def test_unknown_engine_raises():
    with pytest.raises(ValueError, match="Unknown VAD engine"):
        _services("nope").get_vad_engine()
