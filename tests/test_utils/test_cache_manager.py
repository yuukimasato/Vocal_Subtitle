"""测试 CacheManager 缓存管理模块"""

import tempfile
from pathlib import Path

import pytest

from vocal_subtitle.utils.cache_manager import CacheManager


class TestCacheManager:
    """缓存管理器测试"""

    @pytest.fixture
    def cache(self, temp_dir: Path) -> CacheManager:
        """创建临时缓存实例"""
        return CacheManager(
            cache_dir=str(temp_dir / "cache"),
            ttl_separation=3600,
            ttl_transcription=7200,
        )

    def test_make_key_consistent(self):
        """相同输入产生相同键"""
        key1 = CacheManager.make_key(Path("/tmp/test.mp3"), engine="spleeter")
        key2 = CacheManager.make_key(Path("/tmp/test.mp3"), engine="spleeter")
        assert key1 == key2
        assert len(key1) == 64  # SHA256 哈希

    def test_make_key_different(self):
        """不同输入产生不同键"""
        key1 = CacheManager.make_key(Path("/tmp/test1.mp3"))
        key2 = CacheManager.make_key(Path("/tmp/test2.mp3"))
        assert key1 != key2

    def test_make_key_with_params(self):
        """参数影响缓存键"""
        key1 = CacheManager.make_key(Path("test.mp3"), model="large-v3")
        key2 = CacheManager.make_key(Path("test.mp3"), model="small")
        assert key1 != key2

    def test_set_and_get(self, cache: CacheManager):
        """设置和获取缓存"""
        key = "test_key"
        value = {"text": "hello", "segments": [1, 2, 3]}
        cache.set("transcription", key, value)
        result = cache.get("transcription", key)
        assert result == value

    def test_get_missing(self, cache: CacheManager):
        """获取不存在的键返回 None"""
        result = cache.get("transcription", "nonexistent")
        assert result is None

    def test_delete(self, cache: CacheManager):
        """删除缓存项"""
        key = "to_delete"
        cache.set("transcription", key, "data")
        assert cache.delete("transcription", key) is True
        assert cache.get("transcription", key) is None

    def test_delete_nonexistent(self, cache: CacheManager):
        """删除不存在的键"""
        assert cache.delete("transcription", "no_such_key") is False

    def test_clear_stage(self, cache: CacheManager):
        """清除阶段缓存"""
        cache.set("separation", "key1", "data1")
        cache.set("separation", "key2", "data2")
        cache.set("transcription", "key3", "data3")

        cache.clear_stage("separation")
        assert cache.get("separation", "key1") is None
        assert cache.get("separation", "key2") is None
        # 其他阶段不受影响
        assert cache.get("transcription", "key3") == "data3"

    def test_clear_all(self, cache: CacheManager):
        """清除所有缓存"""
        cache.set("separation", "key1", "data1")
        cache.set("transcription", "key2", "data2")

        cache.clear_all()
        assert cache.get("separation", "key1") is None
        assert cache.get("transcription", "key2") is None

    def test_cache_dir_property(self, cache: CacheManager):
        """缓存目录属性"""
        assert cache.cache_dir.exists()
        assert cache.cache_dir.is_dir()

    def test_default_ttl(self, cache: CacheManager):
        """默认 TTL"""
        assert cache._ttl_map["separation"] == 3600
        assert cache._ttl_map["transcription"] == 7200
        assert cache._ttl_map["vad"] == 3600
        assert cache._ttl_map["subtitle"] == 7200

    def test_custom_ttl(self, cache: CacheManager):
        """自定义 TTL 存储"""
        key = "ttl_test"
        cache.set("transcription", key, "data", ttl=10)
        result = cache.get("transcription", key)
        assert result == "data"

    def test_stage_isolation(self, cache: CacheManager):
        """不同阶段的键不应冲突"""
        key = "same_key"
        cache.set("separation", key, "sep_data")
        cache.set("transcription", key, "asr_data")

        assert cache.get("separation", key) == "sep_data"
        assert cache.get("transcription", key) == "asr_data"

    @pytest.mark.parametrize("stage", ["separation", "transcription", "vad", "subtitle"])
    def test_all_stages(self, cache: CacheManager, stage: str):
        """所有阶段都能正常存取"""
        key = f"test_{stage}"
        value = {stage: True}
        cache.set(stage, key, value)
        assert cache.get(stage, key) == value
