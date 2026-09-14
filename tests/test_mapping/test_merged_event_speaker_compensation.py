"""合并塌缩场景的说话人补偿(_multi_speaker_spans/_relabel_multi_speaker_cues)。"""

from typing import Optional

from vocal_subtitle.application.pipeline_result import PipelineStats
from vocal_subtitle.diarization.base import SpeakerTurn
from vocal_subtitle.diarization.early_turns import EarlyTurnsState
from vocal_subtitle.mapping.pipeline_stage import PipelineMappingMixin
from vocal_subtitle.mapping.time_mapper import SubtitleEvent


class _Pipeline(PipelineMappingMixin):
    def __init__(self, config, turns, single_speaker=False):
        self.config = config
        self._early_turns_state = EarlyTurnsState(
            turns=turns,
            speaker_count=len({t.speaker_id for t in turns}),
            status="ok",
            attempted=True,
            single_speaker=single_speaker,
        )
        self._resolved_language = "zh"

    def _resolved_language_or_config(self) -> Optional[str]:
        return self._resolved_language


class _DiarConfig:
    enabled = True


class _Config:
    def __init__(self):
        self.diarization = _DiarConfig()


class _OffConfig:
    enabled = False


def _turns(*triples):
    return [
        SpeakerTurn(start=start, end=end, speaker_id=speaker_id)
        for start, end, speaker_id in triples
    ]


def test_multi_speaker_merged_event_is_detected_and_cues_relabeled():
    turns = _turns((0.0, 5.0, 0), (5.0, 10.0, 1))
    mixin = _Pipeline(_Config(), turns)

    # 合并塌缩的单一事件:横跨 A/B 两个 turn
    events = [SubtitleEvent(1, 0.5, 9.5, "整段文本" * 12, speaker_id=0, speaker_label="说话人A")]
    spans, state = mixin._multi_speaker_spans(events)
    assert spans == [(0.5, 9.5)]
    assert state is mixin._early_turns_state

    # finalizer 拆出的 cue 按主导 turn 重新归属
    cues = [
        SubtitleEvent(1, 0.5, 4.8, "前半", speaker_id=0, speaker_label="说话人A"),
        SubtitleEvent(2, 5.2, 9.5, "后半", speaker_id=0, speaker_label="说话人A"),
    ]
    relabeled, distinct = mixin._relabel_multi_speaker_cues(cues, spans, state)

    assert cues[0].speaker_id == 0
    assert cues[1].speaker_id == 1
    assert cues[1].speaker_label == "说话人B"
    assert relabeled == 1 and distinct == 2


def test_finalize_pipeline_end_to_end_bumps_speaker_count():
    turns = _turns((0.0, 5.0, 0), (5.0, 10.0, 1))
    mixin = _Pipeline(_Config(), turns)
    events = [SubtitleEvent(1, 0.5, 9.5, "整段文本" * 12, speaker_id=0, speaker_label="说话人A")]

    stats = PipelineStats(input_path="x.wav", duration_seconds=10.0)
    stats.speaker_count = 1
    finalized = mixin._finalize_events(events, stats, 10.0)

    labels = {cue.speaker_label for cue in finalized}
    assert "说话人B" in labels
    assert stats.speaker_count == 2


def test_single_speaker_state_never_touches_cues():
    turns = _turns((0.0, 10.0, 0))
    mixin = _Pipeline(_Config(), turns, single_speaker=True)
    events = [SubtitleEvent(1, 1.0, 9.0, "口播", speaker_id=0, speaker_label="说话人A")]

    stats = PipelineStats(input_path="x.wav", duration_seconds=10.0)
    stats.speaker_count = 1
    finalized = mixin._finalize_events(events, stats, 10.0)

    assert finalized[0].speaker_id == 0
    assert stats.speaker_count == 1


def test_disabled_diarization_keeps_labels_untouched():
    class _OffCfg:
        def __init__(self):
            self.diarization = _OffConfig()

    mixin = _Pipeline(_OffCfg(), _turns((0.0, 5.0, 0), (5.0, 10.0, 1)))
    events = [SubtitleEvent(1, 0.0, 10.0, "合并事件", speaker_id=0, speaker_label="说话人A")]

    stats = PipelineStats(input_path="x.wav", duration_seconds=10.0)
    stats.speaker_count = 1
    finalized = mixin._finalize_events(events, stats, 10.0)

    assert finalized[0].speaker_id == 0
