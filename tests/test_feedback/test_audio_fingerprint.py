"""反馈学习模块 — 单元与集成测试

覆盖范围:
  - aligner: 1:1/N:1/1:N 匹配, 锚点切分, 对齐质量门控
  - diff_analyzer: 修改分类, 归因映射, 参数解耦
  - param_learner: EMA 更新, 学习率分级, 异常值过滤, 硬边界
  - user_profile: CRUD, 备份轮转, 回滚, 分级衰减
  - few_shot_builder: LRU 淘汰, 重复检测, 衰减淘汰
  - health_scorer: 4维加权, 自动回滚判断
  - impact_estimator: 影响预估方向正确性
  - conflict_detector: 震汤检测, 非震汤不误报
  - audio_fingerprint: 向量转换, 马氏距离, KNN 动态阈值
  - shadow_mode: 升级/丢弃决策

设计原则: 所有测试使用合成数据，不加载真实模型或音频文件。
"""

import json
import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 测试 fixtures — 合成 SubtitleEvent
# ---------------------------------------------------------------------------

# 使用简单的 dict-as-event 或动态创建 SubtitleEvent
# SubtitleEvent 来自 vocal_subtitle.mapping.time_mapper


def _make_event(index, start, end, text, speaker_label=None):
    """创建合成的 SubtitleEvent"""
    from vocal_subtitle.mapping.time_mapper import SubtitleEvent

    return SubtitleEvent(
        index=index,
        start=start,
        end=end,
        text=text,
        speaker_label=speaker_label,
    )


def _make_events(times_texts):
    """从 (start, end, text) 列表批量创建 SubtitleEvent"""
    events = []
    for i, (start, end, text) in enumerate(times_texts, 1):
        events.append(_make_event(i, start, end, text))
    return events


# ============================================================================
#  1. Aligner 测试
# ============================================================================


class TestAudioFingerprint:
    """音频指纹测试"""

    def test_vector_roundtrip(self):
        """向量序列化/反序列化一致"""
        from vocal_subtitle.feedback.audio_fingerprint import AudioFingerprint

        fp = AudioFingerprint(
            duration_seconds=123.45,
            sample_rate=16000,
            spectral_centroid_mean=2000.0,
            spectral_bandwidth_mean=1500.0,
            spectral_contrast_mean=[10.0, 20.0, 30.0, 25.0, 15.0, 5.0, 8.0],
            mfcc_means=[float(i) for i in range(13)],
            rms_mean=0.05,
            rms_std=0.02,
            zero_crossing_rate=0.1,
            speech_ratio=0.7,
            estimated_speaker_count=2,
            noise_floor_db=-50.0,
            snr_estimate=25.0,
        )

        vec = fp.to_vector()
        assert vec.shape == (48,)
        assert vec.dtype == np.float32

        restored = AudioFingerprint.from_vector(vec)
        assert restored.duration_seconds == pytest.approx(123.45)
        assert restored.speech_ratio == pytest.approx(0.7)
        assert restored.estimated_speaker_count == 2

    def test_mahalanobis_distance_same_vector(self):
        """相同向量 → 距离 ≈ 0"""
        from vocal_subtitle.feedback.audio_fingerprint import (
            AudioFingerprint,
            MahalanobisMatcher,
        )

        fp = AudioFingerprint(duration_seconds=60.0, sample_rate=16000)
        vec = fp.to_vector()

        matcher = MahalanobisMatcher()
        # 用 10 个相同向量拟合
        matcher.fit(np.array([vec] * 10))

        dist = matcher.mahalanobis_distance(vec, vec)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_mahalanobis_distance_different_vectors(self):
        """不同向量 → 距离 > 0"""
        from vocal_subtitle.feedback.audio_fingerprint import (
            AudioFingerprint,
            MahalanobisMatcher,
        )

        fp1 = AudioFingerprint(duration_seconds=60.0, snr_estimate=30.0)
        fp2 = AudioFingerprint(duration_seconds=600.0, snr_estimate=5.0)

        vec1, vec2 = fp1.to_vector(), fp2.to_vector()

        matcher = MahalanobisMatcher()
        matcher.fit(np.array([vec1, vec2] * 5))

        dist = matcher.mahalanobis_distance(vec1, vec2)
        assert dist > 0.5  # 差异应显著

    def test_similarity_decay(self):
        """马氏距离 → 相似度：距离越大相似度越低"""
        from vocal_subtitle.feedback.audio_fingerprint import MahalanobisMatcher

        matcher = MahalanobisMatcher()

        sim_near = matcher.to_similarity(0.1)
        sim_far = matcher.to_similarity(10.0)
        assert sim_near > sim_far
        assert 0 < sim_near <= 1.0
        assert 0 <= sim_far < 1.0

    def test_cosine_similarity_baseline(self):
        """余弦相似度作为备选方案"""
        from vocal_subtitle.feedback.audio_fingerprint import (
            AudioFingerprint,
            MahalanobisMatcher,
        )

        fp = AudioFingerprint(duration_seconds=60.0)
        vec = fp.to_vector()

        matcher = MahalanobisMatcher()
        sim = matcher.cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_sqlite_crud(self):
        """SQLite 数据库 CRUD 操作"""
        from vocal_subtitle.feedback.audio_fingerprint import AudioFingerprint, AudioFingerprinter

        db_path = Path(tempfile.gettempdir()) / "__test_fingerprints.db"
        try:
            fingerprinter = AudioFingerprinter(db_path=db_path)

            fp = AudioFingerprint(duration_seconds=100.0, snr_estimate=20.0)
            audio_hash = "test_hash_abc123"

            # Store
            row_id = fingerprinter.store("__test__", fp, audio_hash)
            assert row_id > 0

            # Get by hash
            result = fingerprinter.get_by_audio_hash(audio_hash, "__test__")
            assert result is not None
            assert result["audio_hash"] == audio_hash

            # List all
            all_fps = fingerprinter.list_all()
            assert len(all_fps) >= 1

            # Record feedback
            fh_id = fingerprinter.record_feedback(
                profile_id="__test__",
                audio_hash=audio_hash,
                alignment_coverage=0.95,
                diff_summary="test summary",
                adjustments={"merging.padding": [0.10, 0.14]},
                health_before=80.0,
                health_after=85.0,
                health_detail={"alignment_coverage": 90.0},
            )
            assert fh_id > 0

            # Get health trend
            trend = fingerprinter.get_health_trend("__test__")
            assert len(trend) >= 1

            # Delete
            fingerprinter.delete_by_id(row_id)
            result = fingerprinter.get_by_audio_hash(audio_hash, "__test__")
            assert result is None

        finally:
            # Cleanup
            if db_path.exists():
                db_path.unlink(missing_ok=True)
            for ext in ("-wal", "-shm"):
                p = Path(str(db_path) + ext)
                p.unlink(missing_ok=True)


# ============================================================================
# 10. ShadowMode 测试
# ============================================================================

