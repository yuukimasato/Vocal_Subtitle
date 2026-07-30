"""测试 whisper.cpp 引擎"""

import pytest

import vocal_subtitle.asr.whisper_cpp_engine as whisper_cpp_module
from vocal_subtitle.asr.whisper_cpp_engine import WhisperCppEngine
from vocal_subtitle.asr.base import ASRDependencyError, ASRModelError


class TestWhisperCppEngine:
    """whisper.cpp 引擎单元测试"""

    def test_engine_name(self):
        engine = WhisperCppEngine()
        assert engine.name == "whisper-cpp"

    def test_model_name(self):
        engine = WhisperCppEngine(model="small")
        assert engine.model_name == "small"

    def test_default_parameters(self):
        engine = WhisperCppEngine()
        assert engine._n_threads == 4
        assert engine._language is None

    def test_custom_parameters(self):
        engine = WhisperCppEngine(
            model="medium",
            n_threads=8,
            language="zh",
            whisper_cpp_bin="/usr/local/bin/whisper-cli",
        )
        assert engine.model_name == "medium"
        assert engine._n_threads == 8
        assert engine._language == "zh"
        assert engine._bin == "/usr/local/bin/whisper-cli"

    def test_parse_timestamp(self):
        engine = WhisperCppEngine()
        result = engine._parse_timestamp("00:01:30,500")
        assert result == 90.5

    def test_parse_timestamp_zero(self):
        engine = WhisperCppEngine()
        result = engine._parse_timestamp("00:00:00,000")
        assert result == 0.0

    def test_parse_timestamp_hour(self):
        engine = WhisperCppEngine()
        result = engine._parse_timestamp("01:00:00,000")
        assert result == 3600.0

    def test_repr(self):
        engine = WhisperCppEngine(model="tiny")
        rep = repr(engine)
        assert "WhisperCppEngine" in rep
        assert "tiny" in rep

    def test_model_not_loaded_initially(self):
        engine = WhisperCppEngine()
        assert engine._model_path is None

    def test_load_model_reports_missing_binary(self, tmp_path):
        model = tmp_path / "ggml-small.bin"
        model.write_bytes(b"model")
        engine = WhisperCppEngine(
            model="small",
            whisper_cpp_bin=str(tmp_path / "missing-whisper-cli"),
            model_path=str(model),
        )

        with pytest.raises(ASRDependencyError, match="executable"):
            engine.load_model()

    def test_load_model_reports_missing_model(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        engine = WhisperCppEngine(
            model="small",
            whisper_cpp_bin=str(binary),
            model_path=str(tmp_path / "missing-model.bin"),
        )

        with pytest.raises(ASRModelError, match="model"):
            engine.load_model()

    def test_load_model_accepts_configured_binary_and_model(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        model = tmp_path / "ggml-small.bin"
        model.write_bytes(b"model")
        engine = WhisperCppEngine(
            model="small",
            whisper_cpp_bin=str(binary),
            model_path=str(model),
        )

        engine.load_model()
        assert engine._model_path == model

    def test_load_model_discovers_project_local_runtime(self, tmp_path, monkeypatch):
        local_bin = tmp_path / "cache" / "whisper_cpp" / "build" / "bin" / "whisper-cli"
        local_model = tmp_path / "cache" / "whisper_cpp" / "models" / "ggml-tiny.bin"
        local_bin.parent.mkdir(parents=True)
        local_model.parent.mkdir(parents=True)
        local_bin.write_text("#!/bin/sh\nexit 0\n")
        local_bin.chmod(0o755)
        local_model.write_bytes(b"model")
        monkeypatch.setattr(whisper_cpp_module, "_PROJECT_ROOT", tmp_path)

        engine = WhisperCppEngine(model="tiny")

        engine.load_model()

        assert engine._bin == str(local_bin)
        assert engine._model_path == local_model
