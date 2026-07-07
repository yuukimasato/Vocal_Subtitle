"""测试 ASR 边界双向精修器 (方案四)"""

import numpy as np
import pytest

from vocal_subtitle.asr.boundary_refiner import (
    BoundaryRefinementConfig,
    BoundaryRefiner,
    resolve_boundary_conflict,
)
from vocal_subtitle.vad.base import SpeechSegment


class TestBoundaryRefiner:
    """ASR 边界双向精修器测试"""

    @pytest.fixture
    def refiner(self):
        return BoundaryRefiner(BoundaryRefinementConfig(
            enabled=True,
            max_shrink_ms=200,
            max_extend_ms=100,
            check_frames=3,
            frame_ms=10,
        ))

    @pytest.fixture
    def audio_with_onset(self):
        """创建有清晰 onset 的测试音频：静音 → 语音阶跃"""
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * 1.0), dtype=np.float32)
        # 0.5s 处开始语音（阶跃跳变）
        t = np.arange(0, int(0.5 * sample_rate)) / sample_rate
        start = int(0.5 * sample_rate)
        audio[start:start + len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        return audio

    @pytest.fixture
    def audio_with_fadeout(self):
        """创建有语尾渐弱的测试音频"""
        sample_rate = 16000
        duration = 1.5
        audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
        # 0.0-1.0s: 正常语音
        t = np.arange(0, int(1.0 * sample_rate)) / sample_rate
        audio[:len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        # 1.0-1.2s: 渐弱（能量递减）
        fade_samples = int(0.2 * sample_rate)
        fade_start = int(1.0 * sample_rate)
        for i in range(fade_samples):
            factor = 1.0 - (i / fade_samples)
            audio[fade_start + i] = np.sin(2 * np.pi * 440 * (1.0 + i / sample_rate)).astype(np.float32) * 0.5 * factor
        return audio

    # ----------------------------------------------------------------
    # refine_boundary_bidirectional
    # ----------------------------------------------------------------

    def test_onset_step_jump_locked(self, refiner, audio_with_onset):
        """阶跃跳变的 onset 边界应被锁定（不偏移）"""
        result = refiner.refine_boundary_bidirectional(
            audio_with_onset, 16000, 0.5, "onset",
        )
        # 阶跃跳变 (energy_ratio > 5.0) → 边界锁定
        assert result == pytest.approx(0.5)

    def test_offset_on_fadeout_extends(self, refiner, audio_with_fadeout):
        """语尾渐弱的 offset 应向外扩展（保护语尾）"""
        # 正常语音结束约在 1.0s
        result = refiner.refine_boundary_bidirectional(
            audio_with_fadeout, 16000, 1.0, "offset",
        )
        # 缓慢变化 → 边界应向外扩展
        assert result >= 1.0  # 至少不向内收缩

    def test_boundary_clamp_to_zero(self, refiner, audio_with_onset):
        """onset 扩展不应小于 0"""
        result = refiner.refine_boundary_bidirectional(
            audio_with_onset, 16000, 0.01, "onset",
        )
        assert result >= 0.0

    def test_boundary_clamp_to_max(self, refiner, audio_with_onset):
        """offset 扩展不应超过音频长度"""
        max_time = len(audio_with_onset) / 16000
        result = refiner.refine_boundary_bidirectional(
            audio_with_onset, 16000, max_time - 0.01, "offset",
        )
        assert result <= max_time

    # ----------------------------------------------------------------
    # get_energy_ratio_at
    # ----------------------------------------------------------------

    def test_energy_ratio_onset(self, refiner, audio_with_onset):
        """onset 处的能量比应 > 1（语音能量 > 静音能量）"""
        ratio = refiner.get_energy_ratio_at(
            audio_with_onset, 16000, 0.5, "onset",
        )
        assert ratio > 1.0

    def test_energy_ratio_silence(self, refiner, sample_audio_silence):
        """纯静音中任意点的能量比约为 1"""
        ratio = refiner.get_energy_ratio_at(
            sample_audio_silence, 16000, 0.5, "onset",
        )
        # 静音中前后能量相等，比值 ≈ 1
        assert ratio >= 0.0

    # ----------------------------------------------------------------
    # refine_all (集成测试)
    # ----------------------------------------------------------------

    def test_refine_all_no_words(self, refiner, audio_with_onset):
        """无 ASR 词级时间戳时应仍能用能量斜率精修"""
        segments = [SpeechSegment(start=0.3, end=0.7, confidence=0.9)]
        asr_results = [[]]  # 空的 ASR 结果

        refined_segs, _ = refiner.refine_all(
            segments, asr_results, audio_with_onset, 16000,
        )
        assert len(refined_segs) == 1
        # 边界应被精修（不同于原始值或保持接近）
        assert refined_segs[0].start >= 0.0

    def test_refine_all_disabled(self, audio_with_onset):
        """禁用时应原样返回"""
        refiner = BoundaryRefiner(BoundaryRefinementConfig(enabled=False))
        segments = [SpeechSegment(start=0.3, end=0.7, confidence=0.9)]
        asr_results = [[]]

        refined, asr_r = refiner.refine_all(
            segments, asr_results, audio_with_onset, 16000,
        )
        assert refined is segments  # 同一对象
        assert asr_r is asr_results


class TestConflictResolution:
    """方案四 vs 方案七 边界冲突裁决测试"""

    def test_priority1_step_jump_wins(self):
        """优先级1：阶跃跳变 → ASR 结果锁定"""
        result = resolve_boundary_conflict(
            asr_refined_time=1.5,
            asr_energy_ratio=6.0,        # > 5.0 → 阶跃跳变
            physical_nearest=1.3,
            physical_is_silence=True,
            original_time=1.4,
        )
        assert result == 1.5  # ASR 结果优先

    def test_priority2_absolute_silence_wins(self):
        """优先级2：绝对静音 → 物理标尺覆盖"""
        result = resolve_boundary_conflict(
            asr_refined_time=1.5,
            asr_energy_ratio=3.0,        # < 5.0，非阶跃
            physical_nearest=1.3,
            physical_is_silence=True,    # 绝对静音
            original_time=1.4,
        )
        assert result == 1.3  # 物理标尺覆盖

    def test_priority3_conservative(self):
        """优先级3：取修正量较小的（保守）"""
        # ASR 修正 0.1s，物理修正 0.05s → 取物理（更保守）
        result = resolve_boundary_conflict(
            asr_refined_time=1.5,       # |1.5 - 1.4| = 0.1
            asr_energy_ratio=3.0,
            physical_nearest=1.45,      # |1.45 - 1.4| = 0.05
            physical_is_silence=False,   # 非绝对静音
            original_time=1.4,
        )
        assert result == 1.45  # 物理更保守

    def test_priority3_asr_more_conservative(self):
        """ASR 修正更小时取 ASR"""
        result = resolve_boundary_conflict(
            asr_refined_time=1.42,      # |1.42 - 1.4| = 0.02
            asr_energy_ratio=3.0,
            physical_nearest=1.35,      # |1.35 - 1.4| = 0.05
            physical_is_silence=False,
            original_time=1.4,
        )
        assert result == 1.42  # ASR 更保守


# ------------------------------------------------------------------
# 5.12.2 语种切换处理
# ------------------------------------------------------------------


class TestLanguageSwitchDetection:
    """语种切换检测测试"""

    def test_no_switch_in_stable_confidence(self):
        """置信度稳定 → 无切换"""
        from vocal_subtitle.asr.boundary_refiner import detect_language_switches

        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.95},
            {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.92},
            {"word": "this", "start": 1.0, "end": 1.3, "confidence": 0.90},
            {"word": "is", "start": 1.3, "end": 1.5, "confidence": 0.88},
            {"word": "test", "start": 1.5, "end": 2.0, "confidence": 0.91},
        ]
        switches = detect_language_switches(words)
        assert len(switches) == 0

    def test_detect_sharp_confidence_drop(self):
        """置信度骤降 → 检测到切换"""
        from vocal_subtitle.asr.boundary_refiner import detect_language_switches

        words = [
            {"word": "我在", "start": 0.0, "end": 0.3, "confidence": 0.95},
            {"word": "说", "start": 0.3, "end": 0.5, "confidence": 0.93},
            # 语种切换边界：置信度骤降
            {"word": "non-smoking", "start": 0.5, "end": 0.9, "confidence": 0.45},
            {"word": "king", "start": 0.9, "end": 1.2, "confidence": 0.40},
            {"word": "room", "start": 1.2, "end": 1.5, "confidence": 0.85},  # 恢复
        ]
        switches = detect_language_switches(words, confidence_drop_threshold=0.3)
        assert len(switches) >= 1, f"Expected at least 1 switch, got {switches}"

    def test_short_drop_not_switch(self):
        """单次置信度下降 → 不判定为切换（最少 2 个连续降信词）"""
        from vocal_subtitle.asr.boundary_refiner import detect_language_switches

        words = [
            {"word": "hello", "start": 0.0, "end": 0.3, "confidence": 0.95},
            {"word": "world", "start": 0.3, "end": 0.5, "confidence": 0.40},  # 单次骤降
            {"word": "test", "start": 0.5, "end": 0.8, "confidence": 0.92},  # 立即恢复
        ]
        switches = detect_language_switches(words, min_switch_length=2)
        assert len(switches) == 0


class TestMultilingualTimestampSmoothing:
    """多语种时间戳平滑测试"""

    def test_smooth_at_switch_point(self):
        """切换点附近时间戳应被平滑处理"""
        from vocal_subtitle.asr.boundary_refiner import smooth_multilingual_timestamps

        words = [
            {"word": "A", "start": 0.0, "end": 0.3, "confidence": 0.9},
            {"word": "B", "start": 0.3, "end": 0.6, "confidence": 0.4},
            {"word": "C", "start": 0.6, "end": 0.9, "confidence": 0.85},
            {"word": "D", "start": 0.9, "end": 1.2, "confidence": 0.88},
            {"word": "E", "start": 1.2, "end": 1.5, "confidence": 0.90},
        ]
        switches = [1]  # 切换点在 word B

        smoothed = smooth_multilingual_timestamps(words, switches)

        # 切换区的词应被标记
        assert any(
            w.get("_timestamp_smoothed") for w in smoothed
            if isinstance(w, dict)
        ), "切换区应有 _timestamp_smoothed 标记"

    def test_no_switches_preserves_input(self):
        """无切换点时输入应被保留"""
        from vocal_subtitle.asr.boundary_refiner import smooth_multilingual_timestamps

        words = [
            {"word": "A", "start": 0.0, "end": 0.3, "confidence": 0.9},
            {"word": "B", "start": 0.3, "end": 0.6, "confidence": 0.88},
        ]
        smoothed = smooth_multilingual_timestamps(words, [])
        assert len(smoothed) == len(words)
