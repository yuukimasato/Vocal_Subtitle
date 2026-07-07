"""声学特征提取器测试"""

import numpy as np
import pytest

from vocal_subtitle.diarization.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """声学特征提取器单元测试"""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(sample_rate=16000)

    @pytest.fixture
    def sine_220hz(self):
        """220Hz 正弦波（模拟低音说话人），0.5 秒"""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        return np.sin(2 * np.pi * 220 * t).astype(np.float32)

    @pytest.fixture
    def sine_440hz(self):
        """440Hz 正弦波（模拟高音说话人），0.5 秒"""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # ------------------------------------------------------------------
    # 基础功能
    # ------------------------------------------------------------------

    def test_extract_features_returns_array(self, extractor, sine_440hz):
        """特征提取返回 numpy 数组"""
        feats = extractor.extract_features(sine_440hz)
        assert isinstance(feats, np.ndarray)
        assert feats.ndim == 1
        assert len(feats) > 0

    def test_extract_features_no_nan(self, extractor, sine_440hz):
        """特征向量不含 NaN 或 Inf"""
        feats = extractor.extract_features(sine_440hz)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_extract_features_float64(self, extractor, sine_440hz):
        """特征向量为 float64 类型"""
        feats = extractor.extract_features(sine_440hz)
        assert feats.dtype == np.float64

    # ------------------------------------------------------------------
    # 批量提取
    # ------------------------------------------------------------------

    def test_extract_features_batch(self, extractor, sine_220hz, sine_440hz):
        """批量提取返回 (n_segments × n_features) 矩阵"""
        batch = [sine_220hz, sine_440hz, sine_220hz]
        matrix = extractor.extract_features_batch(batch)
        assert matrix.shape[0] == 3
        assert matrix.ndim == 2

    def test_extract_features_batch_empty(self, extractor):
        """空列表返回 (0, 0) 矩阵"""
        matrix = extractor.extract_features_batch([])
        assert matrix.shape == (0, 0)

    # ------------------------------------------------------------------
    # 短片段处理
    # ------------------------------------------------------------------

    def test_short_segment_padded(self, extractor):
        """过短片段（< 0.25s）用零填充"""
        short = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        feats = extractor.extract_features(short)
        assert not np.any(np.isnan(feats))
        assert len(feats) > 0

    # ------------------------------------------------------------------
    # 不同频率产生不同特征（核心说话人区分验证）
    # ------------------------------------------------------------------

    def test_different_frequencies_different_features(
        self, extractor, sine_220hz, sine_440hz
    ):
        """不同频率的音频应产生不同的特征向量"""
        feats_220 = extractor.extract_features(sine_220hz)
        feats_440 = extractor.extract_features(sine_440hz)
        # 特征向量不应相同
        diff = np.linalg.norm(feats_220 - feats_440)
        assert diff > 1e-6, (
            f"Feature vectors for 220Hz and 440Hz should differ, "
            f"but L2 norm of difference = {diff}"
        )

    def test_same_frequency_similar_features(self, extractor):
        """相同频率的音频应产生相似的特征向量"""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio_a = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_b = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8

        feats_a = extractor.extract_features(audio_a)
        feats_b = extractor.extract_features(audio_b)

        # 余弦相似度 ≥ 0.99（音量差异已被归一化）
        cos_sim = np.dot(feats_a, feats_b) / (
            np.linalg.norm(feats_a) * np.linalg.norm(feats_b) + 1e-10
        )
        assert cos_sim > 0.99, (
            f"Same frequency features should be nearly identical, "
            f"cosine similarity = {cos_sim}"
        )

    # ------------------------------------------------------------------
    # 音频类型无关性
    # ------------------------------------------------------------------

    def test_int16_audio_handled(self, extractor):
        """int16 音频自动转换为 float32"""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        feats = extractor.extract_features(audio)
        assert not np.any(np.isnan(feats))
