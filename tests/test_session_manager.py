"""SessionManager 单元测试"""

import json
import tempfile
from pathlib import Path

from vocal_subtitle.utils.session_manager import OUTPUT_NAMES, SessionManager


class TestSessionManager:
    """测试基于哈希的会话目录管理器"""

    def test_compute_session_key(self):
        """测试会话键计算"""
        mgr = SessionManager(Path(tempfile.mkdtemp()))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"test audio content " * 100)
            tmp_path = Path(f.name)

        try:
            key = mgr.compute_session_key(tmp_path)
            assert len(key) == 16, f"Session key should be 16 chars, got {len(key)}"
            # All hex chars
            assert all(c in "0123456789abcdef" for c in key), "Key should be hex"
        finally:
            tmp_path.unlink()

    def test_compute_session_key_from_bytes(self):
        """测试从字节数据计算会话键"""
        key = SessionManager.compute_session_key_from_bytes(b"hello world")
        assert len(key) == 16
        # Deterministic
        key2 = SessionManager.compute_session_key_from_bytes(b"hello world")
        assert key == key2
        # Different data → different key
        key3 = SessionManager.compute_session_key_from_bytes(b"hello world!")
        assert key != key3

    def test_session_dir_creation(self):
        """测试会话目录创建"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            session_dir = mgr.session_dir("a1b2c3d4e5f67890")
            assert str(session_dir).startswith(upload_dir)
            assert session_dir.name == "a1b2c3d4e5f67890"

    def test_exists(self):
        """测试会话目录存在性检查"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            assert not mgr.exists("a1b2c3d4e5f67890")
            mgr.session_dir("a1b2c3d4e5f67890").mkdir()
            assert mgr.exists("a1b2c3d4e5f67890")

    def test_get_or_create(self):
        """测试获取或创建会话目录"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            test_file = Path(upload_dir) / "test.wav"
            test_file.write_bytes(b"test content " * 100)

            # First call: creates directory
            session_dir, is_dup = mgr.get_or_create(test_file, "my audio.aac")
            assert session_dir.exists()
            assert not is_dup

            # Second call: detects duplicate
            session_dir2, is_dup2 = mgr.get_or_create(test_file, "my audio.aac")
            assert session_dir2 == session_dir
            assert is_dup2

    def test_write_and_read_metadata(self):
        """测试元数据写入和读取"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            session_dir = mgr.session_dir("a1b2c3d4e5f67890")
            session_dir.mkdir()

            mgr.write_metadata(
                session_dir,
                original_filename="test audio.aac",
                input_sha256="a" * 64,
                profile="podcast",
                config_hash="b" * 64,
                task_id="task001",
                outputs={"ASR-generated.srt": {"sha256": "c" * 64, "size": 1024}},
            )

            meta = mgr.read_metadata(session_dir)
            assert meta is not None
            assert meta["original_filename"] == "test audio.aac"
            assert meta["input_sha256"] == "a" * 64
            assert meta["profile"] == "podcast"
            assert meta["config_hash"] == "b" * 64
            assert meta["task_id"] == "task001"
            assert "ASR-generated.srt" in meta["outputs"]

    def test_read_metadata_nonexistent(self):
        """测试读取不存在的元数据"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            session_dir = mgr.session_dir("nonexistent")
            assert mgr.read_metadata(session_dir) is None

    def test_get_output_path(self):
        """测试标准化输出路径"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            session_dir = mgr.session_dir("a1b2c3d4e5f67890")

            asr_srt = mgr.get_output_path(session_dir, "asr_srt")
            assert asr_srt.name == "ASR-generated.srt"
            assert asr_srt.parent == session_dir

            vocals = mgr.get_output_path(session_dir, "vocals")
            assert vocals.name == "Human_Voice_Audio.wav"

            llm_ass = mgr.get_output_path(session_dir, "llm_ass")
            assert llm_ass.name == "LLM-optimized.ass"

    def test_output_names_coverage(self):
        """测试 OUTPUT_NAMES 包含所有必要键"""
        required = {
            "vocals", "accompaniment",
            "asr_srt", "asr_vtt", "asr_ass",
            "llm_srt", "llm_vtt", "llm_ass",
        }
        assert set(OUTPUT_NAMES.keys()) == required

    def test_metadata_merge_on_rewrite(self):
        """测试重写元数据时合并 outputs"""
        with tempfile.TemporaryDirectory() as upload_dir:
            mgr = SessionManager(Path(upload_dir))
            session_dir = mgr.session_dir("merge_test_key")
            session_dir.mkdir()

            # First write
            mgr.write_metadata(
                session_dir,
                original_filename="test.aac",
                input_sha256="a" * 64,
                profile="default",
                config_hash="b" * 64,
                outputs={"file1.srt": {"sha256": "x", "size": 100}},
            )

            # Second write — should merge outputs, preserve original timestamp
            mgr.write_metadata(
                session_dir,
                original_filename="",
                input_sha256="",
                profile="",
                config_hash="",
                outputs={"file2.srt": {"sha256": "y", "size": 200}},
            )

            meta = mgr.read_metadata(session_dir)
            assert meta is not None
            assert "file1.srt" in meta["outputs"]
            assert "file2.srt" in meta["outputs"]
            assert meta["original_filename"] == "test.aac"  # Preserved
            assert meta["input_sha256"] == "a" * 64  # Preserved
