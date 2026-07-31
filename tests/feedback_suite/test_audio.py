"""Feedback tests: test_audio."""

from .common import *

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

class TestShadowMode:
    """影子模式评估器测试"""

    def test_should_upgrade_all_conditions_met(self):
        """全部条件满足 → 建议升级"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(
            min_shadow_runs=10,
            upgrade_threshold=0.05,
        )

        for i in range(10):
            evaluator.add_run(ShadowRunResult(
                timestamp=f"2026-07-{(i + 1):02d}T10:00:00",
                health_current=70.0,
                health_shadow=75.0,  # 平均好 5 分 (+7.1%)
                health_detail_current={"alignment_coverage": 70.0, "semantic_similarity": 70.0,
                                        "time_iou": 70.0, "structure_consistency": 70.0},
                health_detail_shadow={"alignment_coverage": 75.0, "semantic_similarity": 75.0,
                                      "time_iou": 75.0, "structure_consistency": 75.0},
            ))

        result = evaluator.should_upgrade()
        assert result.should_upgrade
        assert result.recommendation == "upgrade"

    def test_insufficient_runs(self):
        """运行次数不足 → 继续收集"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(min_shadow_runs=10)
        evaluator.add_run(ShadowRunResult(
            health_current=70.0, health_shadow=80.0,
        ))

        result = evaluator.should_upgrade()
        assert not result.should_upgrade
        assert result.recommendation == "continue"

    def test_insufficient_improvement(self):
        """提升不足阈值 → 丢弃"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(
            min_shadow_runs=3,
            upgrade_threshold=0.05,
        )

        for _ in range(3):
            evaluator.add_run(ShadowRunResult(
                health_current=70.0,
                health_shadow=70.5,  # 仅好 0.5 (+0.7%)
                health_detail_current={"alignment_coverage": 70.0, "semantic_similarity": 70.0,
                                        "time_iou": 70.0, "structure_consistency": 70.0},
                health_detail_shadow={"alignment_coverage": 70.5, "semantic_similarity": 70.5,
                                      "time_iou": 70.5, "structure_consistency": 70.5},
            ))

        result = evaluator.should_upgrade()
        assert result.recommendation == "discard"

    def test_dimension_degradation(self):
        """有子项退化 → 丢弃"""
        from vocal_subtitle.feedback.shadow_mode import (
            ShadowModeEvaluator,
            ShadowRunResult,
        )

        evaluator = ShadowModeEvaluator(
            min_shadow_runs=3,
            upgrade_threshold=0.05,
            max_dim_degradation=0.10,
        )

        for _ in range(3):
            evaluator.add_run(ShadowRunResult(
                health_current=70.0,
                health_shadow=80.0,  # 整体高 14%
                health_detail_current={"alignment_coverage": 70.0, "semantic_similarity": 70.0,
                                        "time_iou": 70.0, "structure_consistency": 70.0},
                health_detail_shadow={"alignment_coverage": 95.0, "semantic_similarity": 95.0,
                                      "time_iou": 30.0,   # ★ 严重退化 -57%
                                      "structure_consistency": 95.0},
            ))

        result = evaluator.should_upgrade()
        assert result.recommendation == "discard"
        assert len(result.degraded_dims) >= 1


# ============================================================================
# 11. 集成测试
# ============================================================================


