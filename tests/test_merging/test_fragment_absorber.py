"""静音幻听碎片吸收器测试（2026-09-13 诊断定案）

背景实例：TTS中文朗读测试-双人.wav 的「得了吧」——ASR 词时间戳在
换人边界偏早，把「得」配到 12.52-12.88 的静音区上（真实语音 12.92 才
开始），生成一整行几乎无语音能量的幻听碎片。
"""

import pytest

from vocal_subtitle.merging.fragment_absorber import absorb_silent_fragments
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


class TestAbsorbSilentFragments:
    def test_phantom_fragment_absorbed_forward(self):
        """「得|了吧」案例：碎片文本并入后继事件并重锚定到真实语音起点"""
        skeleton = [(10.98, 12.58), (12.92, 13.36), (13.98, 16.40)]
        de = SubtitleEvent(index=1, start=12.516, end=12.876, text="得")
        leba = SubtitleEvent(index=2, start=12.876, end=13.096, text="了吧。")
        ni = SubtitleEvent(
            index=3, start=13.988, end=16.42, text="你半路非要超近道走那条野路。",
        )
        out = absorb_silent_fragments([de, leba, ni], skeleton)
        assert len(out) == 2
        assert out[0].text == "得了吧。"
        # start 重锚定到真实语音起点（不再吞静音前的空白）
        assert out[0].start == pytest.approx(12.92)
        assert out[0].end == pytest.approx(13.096)
        # 后继事件不受影响
        assert out[1] is ni

    def test_real_short_utterance_kept(self):
        """有真实语音的短事件（如语气词「誒」）一律保留"""
        skeleton = [(63.70, 63.84)]
        e = SubtitleEvent(index=1, start=63.718, end=63.838, text="誒")
        out = absorb_silent_fragments([e], skeleton)
        assert len(out) == 1
        assert out[0] is e

    def test_isolated_fragment_dropped(self):
        """前后都无可归属语音的碎片整行丢弃"""
        skeleton = [(0.0, 0.5), (10.0, 11.0)]
        e = SubtitleEvent(index=1, start=5.0, end=5.3, text="。")
        out = absorb_silent_fragments([e], skeleton)
        assert out == []

    def test_long_event_never_absorbed(self):
        """长事件即使覆盖率低也不吸收（保守边界）"""
        skeleton = [(0.0, 1.0)]
        e = SubtitleEvent(index=1, start=2.0, end=4.0, text="长字幕")
        out = absorb_silent_fragments([e], skeleton)
        assert len(out) == 1

    def test_empty_skeleton_is_noop(self):
        e = SubtitleEvent(index=1, start=1.0, end=1.2, text="得")
        out = absorb_silent_fragments([e], [])
        assert len(out) == 1


class TestReanchorWordTimestamps:
    def test_silent_word_shifted_to_onset(self):
        """骑在静音上的词起点重锚定到真实语音起点"""
        import numpy as np
        from vocal_subtitle.merging.fragment_absorber import (
            reanchor_word_timestamps,
        )
        from vocal_subtitle.asr.base import WordTimestamp

        sr = 16000
        audio = np.zeros(sr * 3, dtype=np.float32)
        t = np.arange(sr, dtype=np.float32) / sr
        # 语音：0-0.5 与 1.9-2.5；词"的"骑在 1.6-1.7 的静音上
        audio[:int(0.5 * sr)] = np.sin(2 * np.pi * 440 * t[:int(0.5 * sr)]) * 0.5
        audio[int(1.9 * sr):int(2.5 * sr)] = (
            np.sin(2 * np.pi * 440 * t[:int(0.6 * sr)]) * 0.5
        )
        event = SubtitleEvent(
            index=1, start=0.1, end=2.4, text="似的的十八盘",
            words=[
                WordTimestamp("似", 0.2, 0.4),
                WordTimestamp("的", 1.6, 1.7),
                WordTimestamp("十", 1.9, 2.1),
                WordTimestamp("八", 2.1, 2.3),
            ],
        )
        shifted = reanchor_word_timestamps([event], audio, sr)
        # "的"完全落在静音里 → 起点吸到 1.9（保持时长 0.1）
        assert shifted == 1
        assert event.words[1].start == pytest.approx(1.9)
        assert event.words[1].end == pytest.approx(2.0)

    def test_word_with_speech_energy_untouched(self):
        """词首有语音能量的词不动"""
        import numpy as np
        from vocal_subtitle.merging.fragment_absorber import (
            reanchor_word_timestamps,
        )
        from vocal_subtitle.asr.base import WordTimestamp

        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        t = np.arange(sr, dtype=np.float32) / sr
        audio[: sr] = np.sin(2 * np.pi * 440 * t) * 0.5
        event = SubtitleEvent(
            index=1, start=0.0, end=0.5, text="你好",
            words=[WordTimestamp("你", 0.05, 0.3), WordTimestamp("好", 0.3, 0.5)],
        )
        shifted = reanchor_word_timestamps([event], audio, sr)
        assert shifted == 0
        assert event.words[0].start == pytest.approx(0.05)
