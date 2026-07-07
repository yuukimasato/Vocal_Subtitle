"""测试 Open-Unmix 人声分离引擎"""

from vocal_subtitle.separation.base import LicenseInfo
from vocal_subtitle.separation.openunmix_engine import OpenUnmixEngine


class TestOpenUnmixEngine:
    """Open-Unmix 引擎单元测试"""

    def test_engine_name(self):
        engine = OpenUnmixEngine()
        assert engine.name == "openunmix"

    def test_license_info(self):
        engine = OpenUnmixEngine()
        info = engine.license_info
        assert isinstance(info, LicenseInfo)
        assert info.code_license == "MIT"
        assert info.model_license == "MIT"
        assert "open-unmix" in info.source_url.lower()

    def test_default_model_name(self):
        engine = OpenUnmixEngine()
        assert engine.DEFAULT_MODEL == "umxhq"

    def test_repr(self):
        engine = OpenUnmixEngine()
        rep = repr(engine)
        assert "OpenUnmixEngine" in rep
        assert "openunmix" in rep
