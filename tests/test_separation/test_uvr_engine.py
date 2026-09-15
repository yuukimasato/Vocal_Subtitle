"""测试 UVR 人声分离引擎"""

import numpy as np
import soundfile as sf

from vocal_subtitle.separation.base import LicenseInfo
from vocal_subtitle.separation.uvr_engine import UVREngine


class TestUVREngine:
    """UVR 引擎单元测试"""

    def test_engine_name(self):
        engine = UVREngine()
        assert engine.name == "uvr"

    def test_license_info(self):
        engine = UVREngine()
        info = engine.license_info
        assert isinstance(info, LicenseInfo)
        assert info.code_license == "MIT"
        assert info.model_license == "MIT"
        assert "audio-separator" in info.source_url.lower()

    def test_default_model_name(self):
        engine = UVREngine()
        assert "roformer" in engine.DEFAULT_MODEL.lower()

    def test_repr(self):
        engine = UVREngine()
        rep = repr(engine)
        assert "UVREngine" in rep
        assert "uvr" in rep


class TestShortAudioGuard:
    """短音频防护：audio-separator 0.30.2 的 roformer 路径在音频短于
    一个推理 chunk（8s@44.1k）时 overlap_add 负切片崩溃，引擎必须
    pad 到安全长度并在分离后把 stem 截回原始时长。"""

    def _write_wav(self, tmp_path, seconds, rate=24000, subtype="PCM_16"):
        frames = int(seconds * rate)
        data = (np.sin(np.linspace(0, 440 * 2 * np.pi, frames)) * 0.3).astype(
            np.float32
        )
        path = tmp_path / f"in_{seconds:.2f}s_{rate}.wav"
        sf.write(path, data, rate, subtype=subtype)
        return path

    def test_short_input_is_padded_to_safe_length(self, tmp_path):
        engine = UVREngine()
        src = self._write_wav(tmp_path, 5.84)  # 用户失败样本的画像:24kHz/5.84s

        padded, pad_seconds = engine._pad_short_input(src)

        assert pad_seconds > 0
        assert padded != src
        info = sf.info(str(padded))
        assert info.frames / info.samplerate >= 10.0
        orig, _ = sf.read(str(src), dtype="float32")
        head, _ = sf.read(str(padded), dtype="float32", always_2d=True)
        np.testing.assert_allclose(head[: len(orig), 0], orig, atol=1e-6)

    def test_sufficient_input_is_not_padded(self, tmp_path):
        engine = UVREngine()
        src = self._write_wav(tmp_path, 12.0, rate=44100)

        padded, pad_seconds = engine._pad_short_input(src)

        assert pad_seconds == 0.0
        assert padded == src

    def test_trim_stem_back_to_original_length(self, tmp_path):
        engine = UVREngine()
        stem = self._write_wav(tmp_path, 12.0, rate=44100)
        dst = tmp_path / "vocals.wav"

        engine._trim_audio_to_length(stem, dst, keep_seconds=5.84)

        info = sf.info(str(dst))
        assert abs(info.frames / info.samplerate - 5.84) < 0.01

    def test_trim_without_padding_copies_through(self, tmp_path):
        engine = UVREngine()
        src = self._write_wav(tmp_path, 12.0, rate=44100)
        dst = tmp_path / "vocals.wav"

        engine._trim_audio_to_length(src, dst, keep_seconds=None)

        assert sf.info(str(dst)).frames == sf.info(str(src)).frames
