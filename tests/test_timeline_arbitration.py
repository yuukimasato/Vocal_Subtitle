"""时间轴仲裁层(2026-09-11 定案,层2)三规则表测试。

R1 共识(段落级信任骨架) / R2 盲区(禁吸附+能量确认) /
R3 空洞(覆盖审计 × turns 换人边界联动),以及
timeline_arbitration=false 时的行为等价回归。
风格参考 tests/test_acoustic_validator.py。
"""

import numpy as np
import pytest

from vocal_subtitle.acoustic.arbitration import (
    char_overlap,
    event_word_spans,
    words_beyond,
)
from vocal_subtitle.acoustic.validator import AcousticValidator
from vocal_subtitle.asr.base import (
    ASREngine,
    TranscriptionSegment,
    WordTimestamp,
)
from vocal_subtitle.asr.local_recovery import (
    LocalRecoveryEngine,
    make_recovery_requests_from_coverage,
)
from vocal_subtitle.config import AcousticValidationConfig, PipelineConfig
from vocal_subtitle.mapping.time_mapper import SubtitleEvent
from vocal_subtitle.physical.coverage import (
    audit_physical_coverage,
    speaker_change_boundaries,
)
from vocal_subtitle.physical.ir import GlobalWord
from vocal_subtitle.physical.allocator import WordAllocation
from vocal_subtitle.physical.subtitle_bins import PhysicalSubtitleBin
from vocal_subtitle.pipeline import Pipeline


SAMPLE_RATE = 16000


def _validator(**overrides) -> AcousticValidator:
    """默认与生产 default.yaml 对齐的校验器(max_snap_distance=0.15)。"""
    params = dict(
        enabled=True,
        max_snap_distance=0.15,
        snap_start_margin=0.03,
        snap_end_margin=0.003,
        generate_report=True,
    )
    params.update(overrides)
    return AcousticValidator(AcousticValidationConfig(**params))


def _silence(duration: float, amplitude: float = 0.0005) -> np.ndarray:
    return np.full(int(SAMPLE_RATE * duration), amplitude, dtype=np.float32)


def _tone(duration: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def _audio_with_spans(total: float, spans) -> np.ndarray:
    """按 (start, end, kind) 片段拼接音频;kind: 'tone' | 'silence'。"""
    audio = np.zeros(int(SAMPLE_RATE * total), dtype=np.float32)
    for start, end, kind in spans:
        slice_obj = slice(
            int(start * SAMPLE_RATE), int(end * SAMPLE_RATE),
        )
        width = slice_obj.stop - slice_obj.start
        if kind == "tone":
            audio[slice_obj] = _tone(width / SAMPLE_RATE)[:width]
        else:
            audio[slice_obj] = _silence(width / SAMPLE_RATE)[:width]
    return audio


# ------------------------------------------------------------------
# R1 共识:段落级信任骨架
# ------------------------------------------------------------------


class TestR1Consensus:
    """R1:文本一致且超限幅漂移 → 边界钳到骨架端点。"""

    # 骨架:成员段 (0,3) 与 (5,8);事件 end 漂移 0.35s(> max_snap_distance)
    SKELETON = [(0.0, 3.0), (5.0, 8.0)]
    EVIDENCE = ((0.0, 8.0, "今天天气怎么样我们出去走走吧"),)

    def _drifted_event(self) -> SubtitleEvent:
        return SubtitleEvent(
            index=1, start=1.8, end=3.35,
            text="今天天气怎么样我们出去走走吧",
        )

    def test_consensus_clamps_beyond_max_snap_distance(self):
        """文本一致且漂移超限 → 整段信骨架,钳到成员段端点。"""
        validator = _validator(timeline_arbitration=True)
        events = [self._drifted_event()]
        result, report = validator.validate(
            events, evidence_candidates=self.EVIDENCE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # 端点解除 0.15s 限幅,直接钳到成员段 (0,3) 端点
        assert result[0].start == pytest.approx(0.03)
        assert result[0].end == pytest.approx(3.0 - 0.003)
        assert report["r1_applied"] == 1
        reasons = {
            item["reason"] for item in report["boundary_diagnostics"]
        }
        assert "r1_consensus_skeleton_start" in reasons
        assert "r1_consensus_skeleton_end" in reasons

    def test_word_times_stay_asr_inside_consensus(self):
        """R1 整段信骨架但词内时刻仍用 ASR(不修改 event.words)。"""
        validator = _validator(timeline_arbitration=True)
        event = self._drifted_event()
        word = WordTimestamp(
            word="今", start=1.75, end=1.8, confidence=0.9,
        )
        event.words = [word]
        result, _ = validator.validate(
            [event], evidence_candidates=self.EVIDENCE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # 词时间保持原样(相对坐标不被改写)
        assert result[0].words[0].start == pytest.approx(1.75)
        assert result[0].words[0].end == pytest.approx(1.8)

    def test_consensus_bypasses_reliable_word_boundary(self):
        """R1 信任级别高于 reliable-boundary:高置信词边界也让位骨架。"""
        validator = _validator(timeline_arbitration=True)
        event = self._drifted_event()
        event.words = [
            WordTimestamp(word="今", start=1.75, end=1.8, confidence=0.95),
        ]
        result, report = validator.validate(
            [event], evidence_candidates=self.EVIDENCE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        assert report["skipped_high_confidence"] == 0
        assert result[0].start == pytest.approx(0.03)
        assert result[0].end == pytest.approx(3.0 - 0.003)

    def test_not_triggered_when_overlap_chars_insufficient(self):
        """一致字符数 < arbitration_r1_min_overlap_chars → 不触发。"""
        validator = _validator(timeline_arbitration=True)
        events = [
            SubtitleEvent(index=1, start=1.8, end=3.35, text="好的"),
        ]
        result, report = validator.validate(
            events,
            evidence_candidates=((0.0, 8.0, "好的呀"),),
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # 落回现行行为:超限幅只标记,不吸附
        assert result[0].start == pytest.approx(1.8)
        assert result[0].end == pytest.approx(3.35)
        assert report.get("r1_applied", 0) == 0
        assert any(
            item["issue"] == "end_deviation"
            for item in report["events_flagged"]
        )

    def test_not_triggered_when_similarity_insufficient(self):
        """重合率 < arbitration_r1_min_similarity → 不触发(防短语假阳性)。"""
        validator = _validator(timeline_arbitration=True)
        baseline = "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸"
        evidence = "一二三四五六七八九十子丑寅卯辰巳午未申酉"
        # 一致字符 10 ≥ 6,但重合率 10/20 = 0.5 < 0.85
        events = [
            SubtitleEvent(index=1, start=1.8, end=3.35, text=baseline),
        ]
        result, report = validator.validate(
            events,
            evidence_candidates=((0.0, 8.0, evidence),),
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        assert result[0].start == pytest.approx(1.8)
        assert result[0].end == pytest.approx(3.35)
        assert report.get("r1_applied", 0) == 0

    def test_disabled_keeps_current_behavior(self):
        """开关关闭 → 无仲裁诊断,行为与现状逐字段一致。"""
        validator = _validator(timeline_arbitration=False)
        events = [self._drifted_event()]
        result, report = validator.validate(
            events, evidence_candidates=self.EVIDENCE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        assert result[0].start == pytest.approx(1.8)
        assert result[0].end == pytest.approx(3.35)
        assert "r1_applied" not in report
        assert any(
            item["issue"] == "end_deviation"
            for item in report["events_flagged"]
        )

    def test_multi_event_group_clamps_outer_boundaries_only(self):
        """同段多个共识事件只钳组的外边界,内部边界不互相吞并。"""
        validator = _validator(timeline_arbitration=True)
        events = [
            SubtitleEvent(
                index=1, start=1.0, end=1.5,
                text="第一句一共六个字",
            ),
            SubtitleEvent(
                index=2, start=2.0, end=3.35,
                text="第二句一共六个字",
            ),
        ]
        evidence = (
            (0.0, 1.6, "第一句一共六个字"),
            (1.9, 8.0, "第二句一共六个字"),
        )
        result, report = validator.validate(
            events, evidence_candidates=evidence,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        assert report["r1_applied"] == 2
        # 第一个事件:start 钳到段首,end 保持(组内边界)
        assert result[0].start == pytest.approx(0.03)
        assert result[0].end == pytest.approx(1.5)
        # 第二个事件:start 保持,end 钳到段尾
        assert result[1].start == pytest.approx(2.0)
        assert result[1].end == pytest.approx(3.0 - 0.003)
        # 不产生重叠
        assert result[0].end <= result[1].start


# ------------------------------------------------------------------
# R2 盲区:禁吸附 + 能量确认
# ------------------------------------------------------------------


class TestR2BlindZone:
    """R2:ASR 词落在骨架静音区 → 禁吸附裁词,能量确认裁决。

    几何:end=0.44 落在骨架静音区 (0.3,1.0),previous_end=0.3,
    distance=0.14 达限;回缩候选 candidate_end=0.297 会裁掉
    绝对时间 [0.35,0.43] 的盲区词。
    """

    SKELETON = [(0.0, 0.3), (1.0, 2.0)]

    def _event(self, word_text="嘿", confidence=0.9) -> SubtitleEvent:
        return SubtitleEvent(
            index=1, start=0.05, end=0.44, text=word_text + "盲区词",
            words=[WordTimestamp(word_text, 0.30, 0.38, confidence=confidence)],
        )

    def _audio(self, with_tone: bool) -> np.ndarray:
        spans = [(0.0, 5.0, "silence")]
        if with_tone:
            spans.append((0.40, 0.44, "tone"))
        return _audio_with_spans(5.0, spans)

    def test_blind_word_kept_when_energy_confirms(self):
        """词时刻能量高 → 保留 ASR 词时间,事件覆盖到词,标 skeleton_blind。"""
        validator = _validator(timeline_arbitration=True)
        result, report = validator.validate(
            [self._event()], audio=self._audio(with_tone=True),
            sample_rate=SAMPLE_RATE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # 现行 allow_end_shorten 会把 end 回缩到 0.297 裁掉真语音
        assert result[0].end == pytest.approx(0.44)
        assert report["snapped_ends"] == 0
        # end 侧回缩提议被 R2 否决 1 次；2026-09-13 起 start 侧能量吸附
        # 也会提议（锚点 0.40 落在事件自身音区内）并被 R2 否决 1 次，
        # 两次提议都被拦下，事件保持不动——保护语义不变，计数相加。
        assert report["r2_blind_kept"] == 2
        # start 侧提议同样被 R2 否决并标记,skeleton_blind 两侧各计 1 次
        assert report["skeleton_blind"] == 2
        assert any(
            item["issue"] == "skeleton_blind"
            for item in report["events_flagged"]
        )

    def test_blind_word_trimmed_when_energy_low(self):
        """词时刻能量低(幻觉) → 按现行裁剪策略回缩,标 r2_trimmed。

        低置信词不走 reliable-boundary 跳过,R2 无法确认时交回现行
        吸附路径执行裁剪。
        """
        validator = _validator(timeline_arbitration=True)
        result, report = validator.validate(
            [self._event(confidence=0.3)],
            audio=self._audio(with_tone=False),
            sample_rate=SAMPLE_RATE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        assert result[0].end == pytest.approx(0.3 - 0.003)
        assert report["snapped_ends"] == 1
        assert report["r2_trimmed"] == 1
        assert any(
            item["issue"] == "r2_hallucination_trim"
            for item in report["events_flagged"]
        )

    def test_local_noise_rejects_music_residue_that_global_accepts(self):
        """local_noise=True 用词周边局部噪声底,防音乐残留被误确认。

        同一段音频:全局噪声底被远端静音拉低 → 误确认;局部窗口被
        音乐残留填满 → 噪声底抬到词能量之上 → 拒绝确认。
        """
        # 音乐残留铺满 [0, 3.8];全局静音占比 >20%,把全局噪声底拉低
        audio = _audio_with_spans(5.5, [
            (0.0, 3.8, "tone"),
            (3.8, 5.5, "silence"),
        ])

        def _residue_event() -> SubtitleEvent:
            # 低置信词:不走 reliable-boundary 跳过,让 R2 裁决可见
            return SubtitleEvent(
                index=1, start=0.05, end=0.44, text="残留词",
                words=[WordTimestamp("残留", 0.0, 0.55, confidence=0.3)],
            )

        kept_validator = _validator(
            timeline_arbitration=True, arbitration_r2_local_noise=False,
        )
        _, global_report = kept_validator.validate(
            [_residue_event()],
            audio=audio, sample_rate=SAMPLE_RATE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # 全局噪声底(远端静音)→ 误确认为真语音 → kept
        assert global_report["r2_blind_kept"] == 1

        local_validator = _validator(
            timeline_arbitration=True, arbitration_r2_local_noise=True,
        )
        result, local_report = local_validator.validate(
            [_residue_event()],
            audio=audio, sample_rate=SAMPLE_RATE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # 局部噪声底(残留铺满窗口)→ 无法确认为真语音 → 不标 kept;
        # 端点候选区仍有残留能量,现行 rms 复核拒绝回缩,事件保持不动。
        assert local_report["r2_blind_kept"] == 0
        assert local_report["r2_trimmed"] == 0
        assert local_report["rms_overrides"] == 1
        assert result[0].end == pytest.approx(0.44)

    def test_blind_leading_word_kept_on_start_snap(self):
        """静音区词头保护：start 前方有真实语音能量时根本不发起吸附。

        2026-09-13 机制变更：start 侧保护改由 _leading_silence_ahead
        能量前置检查承担（前方 120ms 有语音 → 不产生吸附提议，比 R2
        词级评估更强的保护）；R2 盲区评估继续负责 end 侧回缩路径。
        本用例的词头 [0.0,0.26] 有 tone 能量 → start 保持不动。
        """
        skeleton = [(0.3, 1.0), (2.0, 3.0)]
        audio = _audio_with_spans(3.0, [
            (0.0, 3.0, "silence"),
            (0.18, 0.26, "tone"),
        ])
        event = SubtitleEvent(
            index=1, start=0.2, end=0.9, text="嘿前导词",
            words=[WordTimestamp("嘿", 0.0, 0.05, confidence=0.9)],
        )
        validator = _validator(timeline_arbitration=True)
        result, report = validator.validate(
            [event], audio=audio, sample_rate=SAMPLE_RATE,
            ffmpeg_unified_result={"skeleton": skeleton},
        )
        assert result[0].start == pytest.approx(0.2)
        assert report["snapped_starts"] == 0
        # 吸附提议未发生（能量前置检查拦下），R2 词级评估无需介入
        assert report["r2_blind_kept"] == 0
        # start 豁免不阻塞 end 校验(end=0.9 在成员段内部)
        assert report["r2_trimmed"] == 0

    def test_disabled_has_no_r2_diagnostics(self):
        """开关关闭 → 无 R2 诊断键，无能量否决兜底。

        2026-09-13 行为变更：start 侧能量吸附不再被 R2 保护（无仲裁层），
        start 会吸附到事件尾部音区起点 0.40+margin；该 start 已越过词
        [0.30,0.38]，end 回缩因 candidate < start 被跳过，真语音词依然
        失去覆盖——仲裁层关闭时缺陷仍在，只是表现从 end 裁剪变为
        start 吸附越词。
        """
        validator = _validator(timeline_arbitration=False)
        result, report = validator.validate(
            [self._event(confidence=0.3)],
            audio=self._audio(with_tone=True),
            sample_rate=SAMPLE_RATE,
            ffmpeg_unified_result={"skeleton": self.SKELETON},
        )
        # start 吸附到能量锚点（tone 起点 0.40 + margin 0.02）
        assert result[0].start == pytest.approx(0.42)
        # end 0.44 未回缩：candidate_end 0.297 < 新 start，方向保护跳过
        assert result[0].end == pytest.approx(0.44)
        assert report["snapped_ends"] == 0
        assert "r2_blind_kept" not in report
        assert "r2_trimmed" not in report
        assert not any(
            item["issue"] == "skeleton_blind"
            for item in report["events_flagged"]
        )


# ------------------------------------------------------------------
# R2 判定函数单元测试
# ------------------------------------------------------------------


class TestArbitrationHelpers:
    def test_char_overlap_normalizes_punct_and_case(self):
        overlap, similarity = char_overlap(
            "你好,世界!", "你好世界",
        )
        assert overlap == 4
        assert similarity == pytest.approx(1.0)

    def test_char_overlap_empty_input(self):
        assert char_overlap("", "abc") == (0, 0.0)
        assert char_overlap("abc", "") == (0, 0.0)

    def test_event_word_spans_relative_and_absolute(self):
        # 相对坐标契约:词时间相对 event.start
        relative = SubtitleEvent(
            index=1, start=10.0, end=12.0, text="t",
            words=[WordTimestamp("词", 0.5, 0.8, confidence=0.9)],
        )
        assert event_word_spans(relative) == [(10.5, 10.8)]
        # 绝对坐标契约:物理路径词带 raw_start/raw_end
        absolute = SubtitleEvent(index=1, start=10.0, end=12.0, text="t")
        absolute.words = [SimpleNamespaceWord(
            raw_start=3.0, raw_end=3.4, start=0.0, end=0.0,
        )]
        assert event_word_spans(absolute) == [(3.0, 3.4)]

    def test_words_beyond_sides(self):
        spans = [(1.0, 1.5), (2.0, 2.5)]
        # end 回缩到 2.2 裁掉跨界的 (2.0, 2.5)
        assert words_beyond(spans, 2.2, "end") == [(2.0, 2.5)]
        # start 后移到 2.2 裁掉两个起点在其之前的词
        assert words_beyond(spans, 2.2, "start") == [(1.0, 1.5), (2.0, 2.5)]


class SimpleNamespaceWord:
    """带 raw_* 绝对坐标的词替身(物理路径 GlobalWord 契约)。"""

    def __init__(self, raw_start, raw_end, start, end):
        self.raw_start = raw_start
        self.raw_end = raw_end
        self.start = start
        self.end = end


# ------------------------------------------------------------------
# R3 空洞:覆盖审计 + LocalRecovery 接线 × turns 联动
# ------------------------------------------------------------------


def _allocation(word_id: str, start: float, end: float) -> WordAllocation:
    word = GlobalWord(
        id=word_id,
        text=word_id,
        raw_start=start,
        raw_end=end,
        confidence=0.9,
        source_window_id="test",
        segment_id="segment",
    )
    return WordAllocation(word=word, physical_spans=(), accepted=True)


class TestR3SpeakerHole:
    def _bins(self):
        return [
            PhysicalSubtitleBin(
                "bin-1", 0.0, 1.0, "skeleton", physical_clip_id="clip-a",
            ),
            PhysicalSubtitleBin(
                "bin-2", 2.0, 3.0, "skeleton", physical_clip_id="clip-a",
            ),
        ]

    def test_turns_intersecting_hole_is_flagged(self):
        """空洞区间与换人边界相交 → 标 possible_speaker_hole。"""
        turns = [
            SimpleNamespaceTurn(0.0, 2.1, 0),
            SimpleNamespaceTurn(2.1, 4.0, 1),
        ]
        report = audit_physical_coverage(
            self._bins(),
            [_allocation("covered", 0.2, 0.5)],
            turns=turns,
        )
        assert report.recovery_ranges[0].possible_speaker_hole is True
        assert len(report.possible_speaker_holes) == 1
        assert "possible_speaker_hole:1" in report.alerts
        payload = report.to_dict()
        assert payload["recovery_ranges"][0]["possible_speaker_hole"] is True

    def test_turns_none_keeps_current_behavior(self):
        """turns=None → 行为与现状一致,无 possible_speaker_hole 标记。"""
        report = audit_physical_coverage(
            self._bins(),
            [_allocation("covered", 0.2, 0.5)],
        )
        assert report.recovery_ranges[0].possible_speaker_hole is False
        assert report.possible_speaker_holes == ()
        assert not any(
            alert.startswith("possible_speaker_hole")
            for alert in report.alerts
        )

    def test_same_speaker_boundary_not_flagged(self):
        """相邻 turn 同说话人 → 无换人边界,空洞不标记。"""
        turns = [
            SimpleNamespaceTurn(0.0, 2.1, 0),
            SimpleNamespaceTurn(2.1, 4.0, 0),
        ]
        report = audit_physical_coverage(
            self._bins(),
            [_allocation("covered", 0.2, 0.5)],
            turns=turns,
        )
        assert report.recovery_ranges[0].possible_speaker_hole is False
        assert report.possible_speaker_holes == ()

    def test_speaker_change_boundaries_accepts_tuple_turns(self):
        boundaries = speaker_change_boundaries([(0.0, 2.1, 0), (2.1, 4.0, 1)])
        assert boundaries == [pytest.approx(2.1)]

    def test_recovery_requests_carry_speaker_hole_metadata(self):
        """恢复入口:turns 相交的空洞请求带 possible_speaker_hole 元数据。"""
        turns = [
            SimpleNamespaceTurn(0.0, 2.1, 0),
            SimpleNamespaceTurn(2.1, 4.0, 1),
        ]
        report = audit_physical_coverage(
            self._bins(),
            [_allocation("covered", 0.2, 0.5)],
            turns=turns,
        )
        requests = make_recovery_requests_from_coverage(
            report.recovery_ranges, turns=turns,
        )
        assert requests[0].metadata["possible_speaker_hole"] is True

        plain = make_recovery_requests_from_coverage(
            audit_physical_coverage(
                self._bins(), [_allocation("covered", 0.2, 0.5)],
            ).recovery_ranges,
        )
        assert "possible_speaker_hole" not in plain[0].metadata


class SimpleNamespaceTurn:
    def __init__(self, start, end, speaker_id):
        self.start = start
        self.end = end
        self.speaker_id = speaker_id


class _RecoveryEngine(ASREngine):
    """局部重识别桩:对请求区间返回一个高置信候选词。"""

    @property
    def name(self):
        return "recovery-test"

    @property
    def model_name(self):
        return "recovery-test"

    def load_model(self):
        return None

    def transcribe(self, audio, sample_rate=16000, language=None, **kwargs):
        return [
            TranscriptionSegment(
                text="被吞的话轮",
                start=0.5,
                end=0.8,
                words=[WordTimestamp(
                    "被吞的话轮", 0.5, 0.8, confidence=0.9,
                )],
            )
        ]


class TestR3CoverageRecoveryWiring:
    def test_coverage_audit_feeds_local_recovery(self):
        """现行接线完好:覆盖审计 recovery_ranges → 恢复请求 → 局部重识别。

        R3 复用该链路,此处验证不因 turns 参数引入回归。
        """
        bins = [
            PhysicalSubtitleBin(
                "bin-1", 0.0, 1.0, "skeleton", physical_clip_id="clip-a",
            ),
            PhysicalSubtitleBin(
                "bin-2", 1.2, 2.0, "skeleton", physical_clip_id="clip-a",
            ),
        ]
        report = audit_physical_coverage(
            bins, [_allocation("covered", 0.1, 0.5)],
        )
        assert report.complete is False
        requests = make_recovery_requests_from_coverage(
            report.recovery_ranges,
        )
        assert requests
        engine = LocalRecoveryEngine(_RecoveryEngine())
        results = engine.process_requests(
            requests, np.zeros(int(SAMPLE_RATE * 2.5), dtype=np.float32),
            sample_rate=SAMPLE_RATE,
        )
        assert any(result.success for result in results)
        recovered = [
            candidate.to_global_word()
            for result in results
            if result.success
            for candidate in result.candidates
        ]
        assert recovered[0].metadata["source"] == "local_recovery"


# ------------------------------------------------------------------
# 通道与回归
# ------------------------------------------------------------------


class TestEvidenceChannelAndRegression:
    def test_asr_path_publishes_evidence_regions_when_enabled(self):
        """R1 通道:ASR 管线把全程 evidence 发布到共享配置对象。"""
        config = PipelineConfig()
        config.acoustic_validation.timeline_arbitration = True
        pipeline = Pipeline(config)
        pipeline._global_evidence = (
            SimpleNamespaceCandidate(0.5, 1.5, "共识文本"),
            SimpleNamespaceCandidate(None, 2.0, "无起点应跳过"),
        )
        pipeline._publish_arbitration_evidence()
        assert pipeline.config.acoustic_validation.arbitration_evidence_regions == (
            (0.5, 1.5, "共识文本"),
        )

    def test_asr_path_skips_publish_when_disabled(self):
        """开关关闭 → 不做运行期注入,配置保持 None(现状行为)。"""
        config = PipelineConfig()
        assert config.acoustic_validation.timeline_arbitration is False
        pipeline = Pipeline(config)
        pipeline._global_evidence = (
            SimpleNamespaceCandidate(0.5, 1.5, "共识文本"),
        )
        pipeline._publish_arbitration_evidence()
        assert (
            pipeline.config.acoustic_validation.arbitration_evidence_regions
            is None
        )

    def test_config_loader_reads_arbitration_fields(self, tmp_path):
        """default.yaml 的仲裁开关能被 ConfigLoader 读入(缺口补齐验证)。"""
        from vocal_subtitle.config import ConfigLoader

        (tmp_path / "arbitration.yaml").write_text(
            "pipeline:\n"
            "  acoustic_validation:\n"
            "    timeline_arbitration: true\n"
            "    arbitration_r1_min_overlap_chars: 4\n"
            "    arbitration_r1_min_similarity: 0.7\n"
            "    arbitration_r2_local_noise: false\n",
            encoding="utf-8",
        )
        config = ConfigLoader(configs_dir=tmp_path).load_profile("arbitration")
        acoustic = config.acoustic_validation
        assert acoustic.timeline_arbitration is True
        assert acoustic.arbitration_r1_min_overlap_chars == 4
        assert acoustic.arbitration_r1_min_similarity == pytest.approx(0.7)
        assert acoustic.arbitration_r2_local_noise is False

    def test_default_profile_keeps_arbitration_disabled(self):
        """回归:默认配置仲裁关闭,行为与现状等价。"""
        from vocal_subtitle.config import ConfigLoader

        config = ConfigLoader().load_profile("default")
        acoustic = config.acoustic_validation
        assert acoustic.timeline_arbitration is False
        assert acoustic.arbitration_evidence_regions is None


class SimpleNamespaceCandidate:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text
