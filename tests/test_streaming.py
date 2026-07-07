"""测试流式架构模块 (5.12.5)"""

import numpy as np
import pytest

from vocal_subtitle.streaming import (
    PipelineMode,
    StreamingBuffer,
    StreamingMergeEngine,
    resolve_streaming_modules,
)


class TestPipelineMode:
    """PipelineMode 测试"""

    def test_default_offline(self):
        mode = PipelineMode()
        assert mode.mode == "offline"
        assert not mode.is_streaming()

    def test_streaming_mode(self):
        mode = PipelineMode(mode="streaming")
        assert mode.is_streaming()

    def test_streaming_defaults(self):
        mode = PipelineMode(mode="streaming")
        assert mode.streaming_chunk_duration == 2.0
        assert mode.streaming_overlap_duration == 0.5
        assert mode.streaming_max_latency == 3.0


class TestStreamingModules:
    """流式降级模块映射测试"""

    def test_macro_chunk_disabled(self):
        modules = resolve_streaming_modules()
        assert not modules["macro_chunk"], "流式模式不应启用宏观切块"

    def test_acoustic_validation_disabled(self):
        modules = resolve_streaming_modules()
        assert not modules["acoustic_validation"], "流式模式不应启用全局声学校验"

    def test_llm_merge_local_only(self):
        modules = resolve_streaming_modules()
        assert modules["llm_merge"] == "local_only", "流式模式应降级为本地 NLP"

    def test_core_modules_enabled(self):
        modules = resolve_streaming_modules()
        assert modules["pre_split"], "预切分应在流式模式下可用"
        assert modules["asr_refine"], "ASR 精修应在流式模式下可用"
        assert modules["frame_seamless"], "帧衔接应在流式模式下可用"


class TestStreamingBuffer:
    """StreamingBuffer 测试"""

    @pytest.fixture
    def buffer(self):
        return StreamingBuffer(
            chunk_duration=0.5,       # 500ms 窗口（便于测试）
            overlap_duration=0.125,   # 125ms 重叠
            sample_rate=16000,
        )

    def test_not_ready_initially(self, buffer):
        assert not buffer.ready()

    def test_ready_after_enough_data(self, buffer):
        """缓冲区累积足够样本后应 ready"""
        # 追加 0.6s 的音频（> 0.5s chunk）
        buffer.append(np.zeros(9600, dtype=np.float32))  # 0.6s @ 16kHz
        assert buffer.ready()

    def test_window_size(self, buffer):
        """窗口大小应等于 chunk_duration"""
        buffer.append(np.zeros(16000, dtype=np.float32))  # 1s
        window = buffer.get_window()
        assert len(window) == 8000  # 0.5s × 16000

    def test_advance_maintains_overlap(self, buffer):
        """advance 后缓冲区应保留重叠部分"""
        buffer.append(np.zeros(16000, dtype=np.float32))  # 1s
        assert buffer.ready()

        window = buffer.get_window()
        assert len(window) == 8000

        buffer.advance()
        # advance = chunk_samples - overlap_samples = 8000 - 2000 = 6000
        # remaining = 16000 - 6000 = 10000
        assert buffer.remaining() == 10000

    def test_global_offset_tracks_position(self, buffer):
        """global_offset 应正确跟踪已处理的采样点"""
        buffer.append(np.zeros(16000, dtype=np.float32))
        buffer.get_window()
        buffer.advance()  # 消费 6000 samples

        assert buffer.global_offset() == pytest.approx(6000 / 16000, abs=0.001)

    def test_flush_returns_remaining(self, buffer):
        """flush 应返回剩余不足一个窗口的数据"""
        # 追加 0.3s (< 0.5s chunk)
        buffer.append(np.zeros(4800, dtype=np.float32))
        assert not buffer.ready()

        tail = buffer.flush()
        assert tail is not None
        assert len(tail) == 4800

    def test_flush_empty_returns_none(self, buffer):
        """空缓冲区 flush 返回 None"""
        tail = buffer.flush()
        assert tail is None

    def test_multiple_append_accumulates(self, buffer):
        """多次 append 应累积数据"""
        buffer.append(np.zeros(4000, dtype=np.float32))  # 0.25s
        assert not buffer.ready()
        buffer.append(np.zeros(5000, dtype=np.float32))  # 0.3125s, total 0.5625s
        assert buffer.ready()


class TestStreamingMergeEngine:
    """流式合并引擎测试"""

    @pytest.fixture
    def engine(self):
        return StreamingMergeEngine()

    def test_very_short_gap_always_merge(self, engine):
        """< 120ms 间隙一定合并"""
        prev = {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "A"}
        curr = {"start": 1.05, "end": 2.0, "text": "world", "speaker": "A"}
        assert engine.decide_merge_streaming(curr, prev)

    def test_long_gap_never_merge(self, engine):
        """> 800ms 间隙一定不合并"""
        prev = {"start": 0.0, "end": 1.0, "text": "Hello.", "speaker": "A"}
        curr = {"start": 2.0, "end": 3.0, "text": "World.", "speaker": "A"}
        assert not engine.decide_merge_streaming(curr, prev)

    def test_different_speakers_never_merge(self, engine):
        """不同说话人不合并"""
        prev = {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "A"}
        curr = {"start": 1.15, "end": 2.0, "text": "world", "speaker": "B"}
        assert not engine.decide_merge_streaming(curr, prev)

    def test_comma_ending_merge(self, engine):
        """逗号结尾 + 中等间隙 → 合并"""
        prev = {"start": 0.0, "end": 1.0, "text": "Before ending,", "speaker": "A"}
        curr = {"start": 1.35, "end": 2.0, "text": "repeat key details", "speaker": "A"}
        assert engine.decide_merge_streaming(curr, prev)

    def test_sentence_ending_no_merge(self, engine):
        """句号结尾 + 中等间隙 → 不合并"""
        prev = {"start": 0.0, "end": 1.0, "text": "That is right.", "speaker": "A"}
        curr = {"start": 1.40, "end": 2.0, "text": "Next point.", "speaker": "A"}
        assert not engine.decide_merge_streaming(curr, prev)

    def test_buffer_decision_accumulates(self, engine):
        """缓冲区决策应累积到足够条目才输出"""
        frag1 = {"start": 0.0, "end": 0.5, "text": "Hello", "speaker": "A"}
        frag2 = {"start": 0.9, "end": 1.5, "text": "world.", "speaker": "A"}

        assert engine.buffer_decision(frag1) is None  # 仅 1 条
        assert engine.buffer_decision(frag2) is None  # 仅 2 条

        frag3 = {"start": 2.0, "end": 2.5, "text": "How", "speaker": "A"}
        result = engine.buffer_decision(frag3)
        # 现在有 3 条，前两条应被决策
        assert result is not None
        assert len(result) == 1  # frag1 被输出
        assert result[0]["text"] == "Hello"

    def test_flush_buffer(self, engine):
        """flush 应输出所有缓冲字幕"""
        engine.sentence_buffer = [
            {"start": 0.0, "end": 1.0, "text": "A", "speaker": "X"},
            {"start": 2.0, "end": 3.0, "text": "B", "speaker": "X"},
        ]
        result = engine.flush_buffer()
        assert len(result) == 2
