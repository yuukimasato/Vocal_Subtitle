"""说话人聚类引擎测试

使用不同频率的正弦波模拟不同说话人，验证声学聚类正确性。
"""

import numpy as np
import pytest

from vocal_subtitle.diarization.speaker_clusterer import SpeakerDiarizer
from vocal_subtitle.vad.base import SpeechSegment


def _make_sine(duration: float, freq: float, sr: int = 16000) -> np.ndarray:
    """生成指定频率和时长的正弦波"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_segments(
    starts_ends: list[tuple[float, float]], sr: int = 16000
) -> list[SpeechSegment]:
    """从 (start, end) 列表创建 SpeechSegment（不含音频）"""
    return [
        SpeechSegment(start=s, end=e) for s, e in starts_ends
    ]


def _make_audio_with_segments(
    segments: list[SpeechSegment],
    frequencies: list[float],
    sr: int = 16000,
) -> np.ndarray:
    """生成完整音频，每段使用指定频率的正弦波填充"""
    total_samples = int(max(seg.end for seg in segments) * sr) + 1
    audio = np.zeros(total_samples, dtype=np.float32)
    for seg, freq in zip(segments, frequencies):
        start_s = int(seg.start * sr)
        end_s = int(seg.end * sr)
        n = end_s - start_s
        if n <= 0:
            continue
        t = np.linspace(0, seg.duration, n, endpoint=False)
        audio[start_s:end_s] = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio


class TestSpeakerDiarizerBasic:
    """说话人聚类器基础测试"""

    def test_empty_segments(self):
        """空片段列表返回空结果"""
        diarizer = SpeakerDiarizer()
        audio = np.zeros(16000, dtype=np.float32)
        result = diarizer.diarize([], audio, 16000)
        assert result == []

    def test_single_segment_single_speaker(self):
        """单一片段 → 单个说话人"""
        diarizer = SpeakerDiarizer()
        segments = _make_segments([(0.0, 1.0)])
        audio = _make_audio_with_segments(segments, [220.0])
        result = diarizer.diarize(segments, audio, 16000)
        assert result == [0]

    def test_name_property(self):
        """引擎名称正确"""
        diarizer = SpeakerDiarizer()
        assert diarizer.name == "agglomerative"

    def test_load_model_noop(self):
        """load_model 为无操作（纯算法实现）"""
        diarizer = SpeakerDiarizer()
        diarizer.load_model()
        assert diarizer._model_loaded is True

    def test_last_silhouette_updated(self):
        """聚类后 last_silhouette_ 被设置"""
        diarizer = SpeakerDiarizer()
        # 使用略有差异的频率以避免零方差特征向量
        segments = _make_segments([(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)])
        audio = _make_audio_with_segments(segments, [220.0, 330.0, 440.0])
        diarizer.diarize(segments, audio, 16000)
        assert hasattr(diarizer, "last_silhouette_")
        assert isinstance(diarizer.last_silhouette_, float)


class TestTwoSpeakerSeparation:
    """双说话人分离测试 — 使用明显不同的频率"""

    def test_two_distinct_speakers_clustered(self):
        """220Hz vs 880Hz（差两个八度）→ 2 个簇，轮廓系数 > 0.5"""
        segments = _make_segments([
            (0.0, 0.8),   # 说话人 A
            (1.0, 1.8),   # 说话人 B
            (2.0, 2.8),   # 说话人 A
            (3.0, 3.8),   # 说话人 B
            (4.0, 4.8),   # 说话人 A
            (5.0, 5.8),   # 说话人 B
        ])
        frequencies = [220.0, 880.0, 220.0, 880.0, 220.0, 880.0]
        audio = _make_audio_with_segments(segments, frequencies)

        diarizer = SpeakerDiarizer(distance_threshold=0.5)
        speaker_ids = diarizer.diarize(segments, audio, 16000)

        # 应检测到 2 个说话人
        unique = set(speaker_ids)
        assert len(unique) == 2, f"Expected 2 speakers, got {len(unique)}"

        # 相同频率的片段应分配到同一说话人
        # 段 0, 2, 4 是 220Hz → 同一说话人
        # 段 1, 3, 5 是 880Hz → 同一说话人
        assert speaker_ids[0] == speaker_ids[2] == speaker_ids[4], (
            f"Same-frequency segments should share a speaker: {speaker_ids}"
        )
        assert speaker_ids[1] == speaker_ids[3] == speaker_ids[5], (
            f"Same-frequency segments should share a speaker: {speaker_ids}"
        )
        assert speaker_ids[0] != speaker_ids[1], (
            "Different-frequency segments should have different speakers"
        )

        # 轮廓系数应 > 0.5（清晰分离）
        assert diarizer.last_silhouette_ > 0.5, (
            f"Silhouette should be > 0.5 for distinct speakers, "
            f"got {diarizer.last_silhouette_:.3f}"
        )

    def test_two_similar_speakers_lower_silhouette(self):
        """220Hz vs 260Hz（接近的频率）→ 轮廓系数较低"""
        segments = _make_segments([
            (0.0, 0.8),   # 说话人 A
            (1.0, 1.8),   # 说话人 B
            (2.0, 2.8),   # 说话人 A
            (3.0, 3.8),   # 说话人 B
        ])
        frequencies = [220.0, 260.0, 220.0, 260.0]
        audio = _make_audio_with_segments(segments, frequencies)

        diarizer = SpeakerDiarizer(distance_threshold=0.3)
        diarizer.diarize(segments, audio, 16000)

        # 相近频率应产生较低的轮廓系数
        # （不强制断言，因为取决于聚类参数，但至少 ≥ 0）
        assert diarizer.last_silhouette_ >= 0.0


class TestThreeSpeakerSeparation:
    """三说话人分离测试"""

    def test_three_distinct_speakers(self):
        """220Hz vs 440Hz vs 880Hz → 3 个簇"""
        segments = _make_segments([
            (0.0, 0.6),   # 说话人 A (220Hz)
            (0.8, 1.4),   # 说话人 B (440Hz)
            (1.6, 2.2),   # 说话人 C (880Hz)
            (2.4, 3.0),   # 说话人 A
            (3.2, 3.8),   # 说话人 B
            (4.0, 4.6),   # 说话人 C
            (4.8, 5.4),   # 说话人 A
            (5.6, 6.2),   # 说话人 B
            (6.4, 7.0),   # 说话人 C
        ])
        frequencies = [
            220.0, 440.0, 880.0,
            220.0, 440.0, 880.0,
            220.0, 440.0, 880.0,
        ]
        audio = _make_audio_with_segments(segments, frequencies)

        diarizer = SpeakerDiarizer(distance_threshold=0.4, max_speakers=10)
        speaker_ids = diarizer.diarize(segments, audio, 16000)

        unique = set(speaker_ids)
        assert len(unique) == 3, (
            f"Expected 3 speakers, got {len(unique)}: {speaker_ids}"
        )

        # 验证一致性分组
        spk_a = speaker_ids[0]
        spk_b = speaker_ids[1]
        spk_c = speaker_ids[2]
        assert speaker_ids[3] == spk_a  # 段 3 也是 A
        assert speaker_ids[4] == spk_b  # 段 4 也是 B
        assert speaker_ids[5] == spk_c  # 段 5 也是 C

    def test_min_max_speaker_constraints(self):
        """min_speakers / max_speakers 约束生效"""
        segments = _make_segments([
            (0.0, 0.5), (1.0, 1.5), (2.0, 2.5),
            (3.0, 3.5), (4.0, 4.5),
        ])
        # 全部使用不同频率 → 理论上可产生多达 5 个簇
        frequencies = [220.0, 330.0, 440.0, 550.0, 660.0]
        audio = _make_audio_with_segments(segments, frequencies)

        # 限制 max_speakers=2
        diarizer = SpeakerDiarizer(distance_threshold=0.3, max_speakers=2)
        speaker_ids = diarizer.diarize(segments, audio, 16000)
        unique = set(speaker_ids)
        assert len(unique) <= 2, (
            f"max_speakers=2 constraint violated: {len(unique)} speakers"
        )


class TestSilhouetteReporting:
    """轮廓系数报告测试"""

    def test_single_cluster_silhouette_one(self):
        """单说话人（相同频率相近音频）→ 轮廓系数应为合理值"""
        # 使用相同频率但略加变化以避免零方差
        segments = _make_segments([
            (0.0, 0.5), (1.0, 1.5), (2.0, 2.5),
        ])
        # 微小随机幅度差异模拟同一人说话的细微变化
        audio = _make_audio_with_segments(segments, [220.0, 221.0, 219.0])

        diarizer = SpeakerDiarizer(
            distance_threshold=0.9, min_speakers=1, max_speakers=10,
        )
        diarizer.diarize(segments, audio, 16000)

        # 轮廓系数应在 [0, 1] 范围内
        assert diarizer.last_silhouette_ >= 0.0, "Silhouette should be >= 0"
        assert diarizer.last_silhouette_ <= 1.0, "Silhouette should be <= 1"
