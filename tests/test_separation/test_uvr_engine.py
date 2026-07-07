"""测试 UVR 人声分离引擎"""

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
