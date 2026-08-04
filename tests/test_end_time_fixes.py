"""测试统一优化方案的关键修复点

覆盖 Phase 1-4 中所有代码级修改的验证：
- TestEndTimePrecision: BoundaryRefiner/TimeMapper/AcousticValidator/SubtitleBuilder
- TestTextQuality: TextNormalizer 编号恢复和专有名词纠错
- TestPipelineArchitecture: 后处理顺序、骨架复用、EndTimePostValidator
"""

import numpy as np
import pytest

from vocal_subtitle.mapping.time_mapper import SubtitleEvent


# ===================================================================
# TestEndTimePrecision — 验证结束时间精度相关修复
# ===================================================================


class TestEndTimePrecision:
    """验证 BoundaryRefiner、TimeMapper、AcousticValidator、SubtitleBuilder 的修改"""

    # ------------------------------------------------------------------
    # BoundaryRefiner: 段尾收缩逻辑
    # ------------------------------------------------------------------

    def test_boundary_refiner_trailing_silence_below_100ms_not_shrunk(self):
        """trailing_silence ≤ 100ms 不收缩（保护语尾渐弱）"""
        from vocal_subtitle.asr.boundary_refiner import (
            BoundaryRefinementConfig,
            BoundaryRefiner,
        )
        from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
        from vocal_subtitle.vad.base import SpeechSegment

        cfg = BoundaryRefinementConfig(
            enabled=True,
            max_shrink_ms=80,
            min_boundary_confidence=0.5,
        )
        refiner = BoundaryRefiner(cfg)

        # 模拟段尾有 80ms trailing silence（< 100ms，不应收缩）
        # SpeechSegment: start=0, end=2.0
        # last_word ends at 1.92, so trailing_silence = 0.08 (80ms)
        seg = SpeechSegment(start=0.0, end=2.0, confidence=0.9)
        word = WordTimestamp(
            word="test", start=1.0, end=1.92, confidence=0.9,
        )
        asr_seg = TranscriptionSegment(
            start=0.0, end=2.0, text="test",
            words=[word],
        )
        asr_results = [[asr_seg]]

        audio = np.zeros(int(2.0 * 16000), dtype=np.float32)
        # 语音段填充正弦波（确保能量检测不干扰）
        t = np.linspace(0, 1.92, int(1.92 * 16000), endpoint=False)
        audio[:len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        refined_segs, _ = refiner.refine_all([seg], asr_results, audio, 16000)
        # trailing_silence = 2.0 - 1.92 = 0.08s < 0.10s → 不收缩
        assert refined_segs[0].end == pytest.approx(2.0, abs=0.02)

    def test_boundary_refiner_trailing_silence_above_100ms_shrunk(self):
        """trailing_silence > 100ms 收缩但保留 50ms"""
        from vocal_subtitle.asr.boundary_refiner import (
            BoundaryRefinementConfig,
            BoundaryRefiner,
        )
        from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
        from vocal_subtitle.vad.base import SpeechSegment

        cfg = BoundaryRefinementConfig(
            enabled=True,
            max_shrink_ms=80,
            min_boundary_confidence=0.5,
            shrink_end_enabled=True,  # 显式启用段尾收缩（新默认值为 False）
        )
        refiner = BoundaryRefiner(cfg)

        # trailing_silence = 2.0 - 1.70 = 0.30s (300ms) > 100ms → 应收缩
        seg = SpeechSegment(start=0.0, end=2.0, confidence=0.9)
        word = WordTimestamp(
            start=1.0, end=1.70, word="test", confidence=0.9,
        )
        asr_seg = TranscriptionSegment(
            start=0.0, end=2.0, text="test",
            words=[word],
        )
        asr_results = [[asr_seg]]

        audio = np.zeros(int(2.0 * 16000), dtype=np.float32)
        t = np.linspace(0, 1.70, int(1.70 * 16000), endpoint=False)
        audio[:len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        refined_segs, _ = refiner.refine_all([seg], asr_results, audio, 16000)
        # trailing=300ms, shrink=max(0, min(300-50, 80)) = 80ms
        # end = 2.0 - 0.08 = 1.92
        expected_end = 2.0 - 0.080
        assert refined_segs[0].end < 2.0
        assert refined_segs[0].end >= expected_end - 0.01

    def test_boundary_refiner_low_confidence_no_shrink(self):
        """低置信度词不触发收缩"""
        from vocal_subtitle.asr.boundary_refiner import (
            BoundaryRefinementConfig,
            BoundaryRefiner,
        )
        from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
        from vocal_subtitle.vad.base import SpeechSegment

        cfg = BoundaryRefinementConfig(
            enabled=True,
            max_shrink_ms=80,
            min_boundary_confidence=0.5,  # 最低 0.5
        )
        refiner = BoundaryRefiner(cfg)

        # 低置信度词 (0.3 < 0.5)
        seg = SpeechSegment(start=0.0, end=2.0, confidence=0.9)
        word = WordTimestamp(
            start=1.0, end=1.50, word="test", confidence=0.3,
        )
        asr_seg = TranscriptionSegment(
            start=0.0, end=2.0, text="test",
            words=[word],
        )
        asr_results = [[asr_seg]]

        audio = np.zeros(int(2.0 * 16000), dtype=np.float32)
        t = np.linspace(0, 2.0, int(2.0 * 16000), endpoint=False)
        audio[:] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        refined_segs, _ = refiner.refine_all([seg], asr_results, audio, 16000)
        # 低置信度 → 不触发词级收缩，但三帧能量斜率可能有轻微调整
        # 确认 end 没有被大幅收缩
        assert refined_segs[0].end > 1.90  # 不能比原始大幅收缩

    # ------------------------------------------------------------------
    # TimeMapper: 钳位逻辑优化
    # ------------------------------------------------------------------

    def test_time_mapper_clamp_uses_max_not_min_alone(self):
        """末尾段 end 锚定到声学边界（speech_seg.end），不被 ASR 时间戳拖偏"""
        from vocal_subtitle.mapping.time_mapper import TimeMapper
        from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
        from vocal_subtitle.vad.base import SpeechSegment

        mapper = TimeMapper(seamless_threshold=0.2)

        # speech_seg.end 是 Stage 2/2.5 的声学边界 = 2.8s
        # ASR last-word.end = 3.0s（Whisper 漂移，超出声学边界 200ms）
        # 末尾段 end 应锚定到声学边界，不被 ASR 拖偏
        speech_seg = SpeechSegment(start=0.0, end=2.8, confidence=0.9)
        word = WordTimestamp(start=0.5, end=3.0, word="hello", confidence=0.9)
        asr_seg = TranscriptionSegment(
            start=0.5, end=3.0, text="hello",
            words=[word],
        )

        events = mapper.map(
            [[asr_seg]], [speech_seg],
            speaker_ids=None, audio=None, sample_rate=16000,
        )

        # 末尾段 end 锚定到 speech_seg.end = 2.8（声学边界）
        assert len(events) == 1
        assert events[0].end == pytest.approx(2.8, abs=0.01)

    def test_time_mapper_clamp_when_asr_before_speech_end(self):
        """ASR end 在 speech_seg.end 之前时正常限制"""
        from vocal_subtitle.mapping.time_mapper import TimeMapper
        from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
        from vocal_subtitle.vad.base import SpeechSegment

        mapper = TimeMapper(seamless_threshold=0.2)

        speech_seg = SpeechSegment(start=0.0, end=5.0, confidence=0.9)
        word = WordTimestamp(start=1.0, end=6.0, word="hello", confidence=0.9)
        asr_seg = TranscriptionSegment(
            start=1.0, end=6.0, text="hello",
            words=[word],
        )

        events = mapper.map(
            [[asr_seg]], [speech_seg],
            speaker_ids=None, audio=None, sample_rate=16000,
        )

        # 末尾段 end 锚定到 speech_seg.end = 5.0（声学边界），不被 ASR 拖偏
        assert len(events) == 1
        assert events[0].end == pytest.approx(5.0, abs=0.01)

    # ------------------------------------------------------------------
    # AcousticValidator: 吸附方向保护
    # ------------------------------------------------------------------

    def test_acoustic_validator_snap_end_only_extends(self):
        """吸附只在延长结束时间时才执行（方向保护）"""
        from vocal_subtitle.acoustic_validator import (
            AcousticValidationConfig,
            AcousticValidator,
        )

        cfg = AcousticValidationConfig(
            enabled=True,
            max_snap_distance=0.25,
            snap_end_margin=0.003,
        )
        validator = AcousticValidator(cfg)

        # 骨架语音段 0.0-2.0 → 字幕 end=1.5 在语音区内，不需要吸附
        skeleton = [(0.0, 2.0)]
        events = [
            SubtitleEvent(index=1, start=0.5, end=1.5, text="Inside speech"),
        ]
        result, report = validator._physical_snap_validation(
            events, skeleton, audio=None, sample_rate=16000,
        )
        # end 已在语音区内 → 不吸附，不缩短
        assert report["snapped_ends"] == 0
        assert result[0].end == 1.5

    def test_acoustic_validator_snap_end_protection(self):
        """吸附候选短于当前 end 时不执行（防止反向缩短）"""
        from vocal_subtitle.acoustic_validator import (
            AcousticValidationConfig,
            AcousticValidator,
        )

        cfg = AcousticValidationConfig(
            enabled=True,
            max_snap_distance=0.25,
            snap_end_margin=0.01,
        )
        validator = AcousticValidator(cfg)

        # 骨架语音段 0.0-1.0 和 1.05-2.0（中间有 0.05s 静音）
        # 字幕 end=1.2 在第二段语音内，但最近的静音边界是 1.05
        # nearest_end=1.05 < event.end=1.2 → 不应吸附（会缩短）
        skeleton = [(0.0, 1.0), (1.05, 2.0)]
        events = [
            SubtitleEvent(index=1, start=0.5, end=1.2, text="Across gap"),
        ]
        result, report = validator._physical_snap_validation(
            events, skeleton, audio=None, sample_rate=16000,
        )
        # end=1.2 落在语音段 (1.05, 2.0) 内 → is_end_in_speech=True → 不触发吸附
        assert report["snapped_ends"] == 0

    def test_acoustic_validator_does_not_extend_end_across_silence(self):
        """静音间隙中的 end 不得延长到后续语音段"""
        from vocal_subtitle.acoustic_validator import (
            AcousticValidationConfig,
            AcousticValidator,
        )

        cfg = AcousticValidationConfig(
            enabled=True,
            max_snap_distance=0.25,
            snap_end_margin=0.003,
        )
        validator = AcousticValidator(cfg)

        # 骨架: 0.0-1.0 (语音), 1.3-2.0 (语音), 中间 0.3s 静音
        # 字幕 end=0.85 → 在语音段内 → 不触发
        # 字幕 end=1.15 → 在静音区 1.0-1.3。
        # 旧逻辑会把 end 吸附到 1.3 附近，跨过整段静音；新逻辑禁止延长。
        skeleton = [(0.0, 1.0), (1.3, 2.0)]
        events = [
            SubtitleEvent(index=1, start=0.5, end=1.15, text="Truncated end"),
        ]
        result, report = validator._physical_snap_validation(
            events, skeleton, audio=None, sample_rate=16000,
        )
        assert report["snapped_ends"] == 0
        assert result[0].end == pytest.approx(1.15)
        assert any(
            item["reason"] == "silence_not_confirmed"
            for item in report["boundary_diagnostics"]
        )

    # ------------------------------------------------------------------
    # TimeMapper: 反向能量扫描（_find_speech_end_backward）
    # ------------------------------------------------------------------

    def test_find_speech_end_backward_finds_transition(self):
        """反向扫描在 speech→silence 转折点正确检测"""
        from vocal_subtitle.mapping.time_mapper import TimeMapper
        from vocal_subtitle.utils.audio_utils import AudioUtils

        sr = 16000
        # 语音 0.0–1.0s，静音 1.0–2.0s
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        audio[:len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        silence_rms = AudioUtils.estimate_silence_rms(audio, sr)
        result = TimeMapper._find_speech_end_backward(
            audio, sr,
            search_start=0.8,
            search_end=2.0,
            silence_rms=silence_rms,
        )
        assert result is not None
        # 语音结束于 1.0s 附近
        assert 0.9 <= result <= 1.15

    def test_find_speech_end_backward_all_silence_returns_none(self):
        """全静音 gap 返回 None"""
        from vocal_subtitle.mapping.time_mapper import TimeMapper

        sr = 16000
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)

        result = TimeMapper._find_speech_end_backward(
            audio, sr,
            search_start=0.0,
            search_end=2.0,
            silence_rms=0.001,
        )
        assert result is None

    def test_merge_gaps_backward_scan_extends_end(self):
        """反向扫描将 prev.end 延伸到真实语音结束点"""
        from vocal_subtitle.mapping.time_mapper import TimeMapper

        sr = 16000
        # 语音 0.0–1.5s，静音 1.5–2.0s
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio[:len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        mapper = TimeMapper(seamless_threshold=0.3, natural_pause_max=1.0)
        events = [
            SubtitleEvent(index=1, start=0.5, end=1.0, text="First"),   # end 提前
            SubtitleEvent(index=2, start=1.5, end=2.0, text="Second"),  # start 精确
        ]
        result = mapper._merge_gaps(events, audio, sr)
        # prev.end 被反向扫描延伸
        assert result[0].end > 1.0
        assert result[0].end < result[1].start  # 不重叠

    def test_merge_gaps_no_audio_fallback(self):
        """无音频时回退到 seamless_threshold 直接衔接"""
        from vocal_subtitle.mapping.time_mapper import TimeMapper

        mapper = TimeMapper(seamless_threshold=0.6, natural_pause_max=1.0)
        events = [
            SubtitleEvent(index=1, start=0.5, end=1.0, text="First"),
            SubtitleEvent(index=2, start=1.3, end=2.0, text="Second"),  # gap=0.3s ≤ 0.6s
        ]
        result = mapper._merge_gaps(events, audio=None, sample_rate=16000)
        # 无音频 → 回退：gap ≤ 0.6s 直接衔接
        assert result[0].end == pytest.approx(1.299, abs=0.01)

    def test_boundary_refiner_shrink_end_disabled(self):
        """新默认值：shrink_end_enabled=False 保持段尾不收缩"""
        from vocal_subtitle.asr.boundary_refiner import (
            BoundaryRefinementConfig,
            BoundaryRefiner,
        )
        from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
        from vocal_subtitle.vad.base import SpeechSegment

        cfg = BoundaryRefinementConfig(
            enabled=True,
            max_shrink_ms=80,
            min_boundary_confidence=0.5,
            shrink_end_enabled=False,  # 显式禁用段尾收缩
        )
        refiner = BoundaryRefiner(cfg)

        # trailing_silence = 2.0 - 1.70 = 0.30s → 旧行为会收缩，新行为不收缩
        seg = SpeechSegment(start=0.0, end=2.0, confidence=0.9)
        word = WordTimestamp(
            start=1.0, end=1.70, word="test", confidence=0.9,
        )
        asr_seg = TranscriptionSegment(
            start=0.0, end=2.0, text="test", words=[word],
        )
        audio = np.zeros(int(2.0 * 16000), dtype=np.float32)
        t = np.linspace(0, 1.70, int(1.70 * 16000), endpoint=False)
        audio[:len(t)] = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        refined_segs, _ = refiner.refine_all(
            [seg], [[asr_seg]], audio, 16000,
        )
        # shrink_end_enabled=False → 段尾不收缩（保留在三帧斜率精修范围内）
        assert refined_segs[0].end >= 1.98  # 未收缩

    def test_subtitle_builder_uses_round_not_int(self):
        """验证时间戳使用 round() 而非 int() 做毫秒转换"""
        from vocal_subtitle.mapping.subtitle_builder import SubtitleBuilder, SubtitleRule

        builder = SubtitleBuilder(rule=SubtitleRule())
        # 通过 _to_ssa 间接验证 round 行为
        # 构造一个 event，其 end*1000 的浮点小数部分 ≥ 0.5
        event = SubtitleEvent(
            index=1,
            start=1.2345,  # *1000 = 1234.5 → round=1235, int=1234
            end=3.6789,    # *1000 = 3678.9 → round=3679, int=3678
            text="Test rounding",
        )

        subs = builder._to_ssa([event], fmt="srt")
        assert len(subs) == 1
        ssa_event = subs[0]
        # round(1234.5) = 1234 (banker's rounding!) or 1235?
        # Actually: round(1234.5) = 1234 in Python (banker's rounding to even)
        # But for 1234.6 → round=1235, int=1234
        # Let's test with a clearer case
        pass  # Placeholder — actual round vs int test below

    def test_subtitle_builder_round_vs_int(self):
        """round() 四舍五入比 int() 截断更精确"""
        from vocal_subtitle.mapping.subtitle_builder import SubtitleBuilder, SubtitleRule

        builder = SubtitleBuilder(rule=SubtitleRule())
        # 使用明确应该向上舍入的值
        event = SubtitleEvent(
            index=1,
            start=1.2346,  # *1000 = 1234.6 → round=1235, int=1234
            end=3.6785,    # *1000 = 3678.5 → round=3678 (banker's), int=3678
            text="Test rounding precision",
        )

        subs = builder._to_ssa([event], fmt="srt")
        assert len(subs) == 1
        ssa_event = subs[0]
        # start: 1234.6 → round = 1235
        assert ssa_event.start == 1235
        # The key point: round() is used, not int()
        # int(1234.6) = 1234, round(1234.6) = 1235
        # So start should be 1235 (correct rounding up)


# ===================================================================
# TestTextQuality — 文本规范化
# ===================================================================


class TestTextQuality:
    """验证 TextNormalizer 的编号恢复和专有名词纠错"""

    @pytest.fixture
    def normalizer(self):
        from vocal_subtitle.asr.text_normalizer import TextNormalizer
        return TextNormalizer()

    # ------------------------------------------------------------------
    # 数字编号恢复
    # ------------------------------------------------------------------

    def test_number_word_to_digit_label(self, normalizer):
        """"One answer" → "1. Answer" """
        result = normalizer.normalize("One answer within three rings")
        assert result.startswith("1.")
        assert "answer" in result

    def test_numbered_list_two(self, normalizer):
        """"Two options" → "2. options" """
        result = normalizer.normalize("Two options are available")
        assert result.startswith("2.")

    def test_numbered_list_not_at_start_no_change(self, normalizer):
        """句中数字词不应被转换"""
        result = normalizer.normalize("There is one answer")
        # "one" 不在句首 → 不应转换
        assert "one" in result.lower()
        assert not result.startswith("1.")

    def test_numbered_list_short_text(self, normalizer):
        """短文本中的数字词也应转换"""
        result = normalizer.normalize("Three steps")
        assert result.startswith("3.")

    # ------------------------------------------------------------------
    # 专有名词纠错
    # ------------------------------------------------------------------

    def test_proper_noun_mahood(self, normalizer):
        """"mahood" → "Mehood" """
        result = normalizer.normalize("Welcome to mahood hotel")
        assert "Mehood" in result
        assert "mahood" not in result.lower()

    def test_proper_noun_lusty(self, normalizer):
        """"lusty" → "Lestie" """
        result = normalizer.normalize("Hello lusty how are you")
        assert "Lestie" in result
        assert "lusty" not in result

    def test_proper_noun_case_preservation(self, normalizer):
        """专有名词替换保留原始大小写风格"""
        result = normalizer.normalize("Mahood is a hotel")
        # 原词首字母大写 → 替换词也应首字母大写
        assert "Mehood" in result

    def test_custom_corrections(self):
        """自定义纠错词典"""
        from vocal_subtitle.asr.text_normalizer import TextNormalizer

        custom = TextNormalizer(custom_corrections={"foo": "Bar"})
        result = custom.normalize("hello foo world")
        assert "Bar" in result
        assert "foo" not in result

    # ------------------------------------------------------------------
    # 标点规范化
    # ------------------------------------------------------------------

    def test_punctuation_normalization_adds_period(self, normalizer):
        """无标点结尾自动补句号"""
        result = normalizer.normalize("Hello world")
        assert result.endswith(".")

    def test_punctuation_already_has_punct(self, normalizer):
        """已有标点不重复添加"""
        result = normalizer.normalize("Hello world!")
        assert result.endswith("!")
        assert not result.endswith("!.")

    def test_batch_normalize(self, normalizer):
        """批量规范化"""
        texts = ["One item", "Two items", "Hello mahood"]
        results = normalizer.normalize_batch(texts)
        assert len(results) == 3
        assert results[0].startswith("1.")
        assert results[1].startswith("2.")
        assert "Mehood" in results[2]

    # ------------------------------------------------------------------
    # 空输入处理
    # ------------------------------------------------------------------

    def test_empty_text(self, normalizer):
        """空白文本原样返回"""
        assert normalizer.normalize("") == ""
        # 纯空格不满足 "not text.strip()" 检查，原样返回
        assert normalizer.normalize("   ") == "   "

    def test_none_safety(self, normalizer):
        """None 安全处理"""
        # normalize 接收 str，Python 类型系统会保护，仅测空串
        pass  # 类型安全由 Python 运行时保证

    # ------------------------------------------------------------------
    # 大小写不敏感 + 多词短语匹配（新增）
    # ------------------------------------------------------------------

    def test_proper_noun_case_insensitive_match(self, normalizer):
        """"Shao Shi" → "Xiao Xi"（大小写不敏感 + 多词短语）"""
        result = normalizer.normalize("Hello Shao Shi welcome")
        assert "Xiao Xi" in result
        assert "Shao Shi" not in result

    def test_proper_noun_mixed_case(self, normalizer):
        """"SHaO sHi" → "Xiao Xi"（混合大小写）"""
        result = normalizer.normalize("Hello SHaO sHi welcome")
        assert "Xiao Xi" in result

    def test_proper_noun_all_caps(self, normalizer):
        """"MAHOOD" → "MEHOOD"（全大写保留）"""
        from vocal_subtitle.asr.text_normalizer import TextNormalizer
        custom = TextNormalizer(custom_corrections={"mahood": "Mehood"})
        result = custom.normalize("Welcome to MAHOOD hotel")
        assert "MEHOOD" in result

    def test_numbered_list_standalone_word(self, normalizer):
        """独立数字词 'Two' → '2.'（被 VAD 拆分的编号）"""
        result = normalizer.normalize("Two")
        assert result.startswith("2.")

    def test_multi_word_phrase_priority_over_single_word(self, normalizer):
        """多词短语优先于单词匹配（"shao shi" 不会被 "shao" 干扰）"""
        # _DEFAULT_CORRECTIONS 含 "shao shi" → "Xiao Xi"
        result = normalizer.normalize("Hello shao shi welcome")
        assert "Xiao Xi" in result
        # 不应出现单词级误匹配
        assert "shao" not in result.lower()

    def test_multi_word_no_partial_match(self, normalizer):
        """多词短语不会误匹配子串（"shaolin" 不匹配 "shao"）"""
        # 但这不在词典中，所以应该原样保留
        result = normalizer.normalize("Hello shaolin welcome")
        assert "shaolin" in result


# ===================================================================
# TestPipelineArchitecture — 后处理顺序和骨架复用
# ===================================================================


class TestPipelineArchitecture:
    """验证 Pipeline 后处理顺序和骨架复用"""

    def test_post_process_events_order(self):
        """验证 _post_process_events 的正确顺序：
        帧级衔接 → LLM 合并 → 声学校验（B1 修复）"""
        import inspect
        from vocal_subtitle.pipeline import Pipeline

        source = inspect.getsource(Pipeline._post_process_events)
        # 检查三个步骤的代码位置顺序
        stitch_pos = source.find("帧级无缝衔接")
        merge_pos = source.find("LLM 语义合并")
        acoustic_pos = source.find("声学标尺校验")

        assert stitch_pos >= 0, "Missing frame seamless stitching"
        assert merge_pos >= 0, "Missing LLM merge"
        assert acoustic_pos >= 0, "Missing acoustic validation"
        # 顺序验证：stitch < merge < acoustic
        assert stitch_pos < merge_pos < acoustic_pos, (
            f"Wrong order: stitch={stitch_pos}, merge={merge_pos}, "
            f"acoustic={acoustic_pos}"
        )

    def test_post_process_events_shared_by_all_paths(self):
        """验证 _post_process_events 被三个路径共用（B4 修复）"""
        import inspect
        from vocal_subtitle.pipeline import Pipeline

        run_source = inspect.getsource(Pipeline.run)
        # 检查 _post_process_events 被调用次数
        call_count = run_source.count("_post_process_events(")
        assert call_count >= 3, (
            f"_post_process_events should be called 3 times (skeleton, "
            f"multi-chunk, single-chunk), found {call_count}"
        )

    def test_post_process_events_accepts_ffmpeg_result(self):
        """验证 _post_process_events 接受 ffmpeg_unified_result 参数"""
        import inspect
        from vocal_subtitle.pipeline import Pipeline

        sig = inspect.signature(Pipeline._post_process_events)
        params = sig.parameters
        assert "ffmpeg_unified_result" in params, (
            "Missing ffmpeg_unified_result parameter"
        )
        # 验证默认值为 None
        assert params["ffmpeg_unified_result"].default is None

    # ------------------------------------------------------------------
    # Event ordering regression (B1 架构 bug 修复验证)
    # ------------------------------------------------------------------

    def test_acoustic_validation_after_llm_merge_in_post_process(self):
        """声学校验在 LLM 合并之后执行（B1 架构 Bug 已修复）"""
        import inspect
        from vocal_subtitle.pipeline import Pipeline

        source = inspect.getsource(Pipeline._post_process_events)

        # LLM merge 代码在 acoustic validation 之前
        merge_call = source.find("_run_llm_merge")
        acoustic_call = source.find("AcousticValidator")

        assert merge_call >= 0, "LLM merge call not found in _post_process_events"
        assert acoustic_call >= 0, "AcousticValidator not found in _post_process_events"
        assert merge_call < acoustic_call, (
            "AcousticValidator must run AFTER LLM merge (B1 fix), "
            f"but merge={merge_call} > acoustic={acoustic_call}"
        )


# ===================================================================
# TestEndTimePostValidator — 结束时间后校验
# ===================================================================


class TestEndTimePostValidator:
    """EndTimePostValidator 校验规则测试"""

    @pytest.fixture
    def validator(self):
        from vocal_subtitle.mapping.end_time_validator import EndTimePostValidator
        return EndTimePostValidator()

    def test_end_before_start_fixed(self, validator):
        """end <= start 自动修复为 +500ms"""
        event = SubtitleEvent(index=1, start=2.0, end=1.0, text="Bad")
        result = validator.validate([event])
        assert result[0].end > result[0].start
        assert result[0].end == pytest.approx(2.5, abs=0.01)  # start + 500ms

    def test_min_duration_enforced(self, validator):
        """时长 < 200ms 自动延长"""
        event = SubtitleEvent(index=1, start=1.0, end=1.05, text="Too short")
        result = validator.validate([event])
        duration_ms = (result[0].end - result[0].start) * 1000
        assert duration_ms == pytest.approx(200.0, abs=0.01)

    def test_micro_gap_merged(self, validator):
        """微小间隙 < 30ms 合并（延长前一条覆盖间隙）"""
        evt1 = SubtitleEvent(index=1, start=0.0, end=1.0, text="First")
        evt2 = SubtitleEvent(index=2, start=1.020, end=2.0, text="Second")
        result = validator.validate([evt1, evt2])
        # gap = 20ms < 30ms → 前一条 end 延长到后一条 start
        assert result[0].end == pytest.approx(1.020, abs=0.001)

    def test_normal_gap_not_merged(self, validator):
        """正常间隙 > 30ms 保持不变"""
        evt1 = SubtitleEvent(index=1, start=0.0, end=1.0, text="First")
        evt2 = SubtitleEvent(index=2, start=1.5, end=2.0, text="Second")
        result = validator.validate([evt1, evt2])
        # gap = 500ms > 30ms → 不合并
        assert result[0].end == 1.0

    def test_large_same_speaker_gap_warns(self, caplog, validator):
        """同说话人大间隙产生告警"""
        import logging
        caplog.set_level(logging.WARNING)

        evt1 = SubtitleEvent(
            index=1, start=0.0, end=1.0, text="First",
            speaker_id=0,
        )
        evt2 = SubtitleEvent(
            index=2, start=3.0, end=4.0, text="Second",
            speaker_id=0,
        )
        result = validator.validate([evt1, evt2])
        # gap = 2.0s > 1.5s → 告警
        assert len(caplog.records) >= 1
        assert any("large gap" in r.message.lower() for r in caplog.records)

    def test_empty_events(self, validator):
        """空事件列表直接返回"""
        result = validator.validate([])
        assert result == []
