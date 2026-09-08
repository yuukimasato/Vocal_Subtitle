from vocal_subtitle.asr.base import TranscriptionSegment, WordTimestamp
from vocal_subtitle.diarization.base import DiarizationResult, SpeakerTurn
from vocal_subtitle.physical.ir import (
    GlobalSpeakerTimeline,
    GlobalTranscript,
    adapt_diarization_result,
    adapt_transcription_segments,
)
from vocal_subtitle.physical.decision_ir import decisions_to_global_transcript
from vocal_subtitle.asr.evidence import EvidenceDecision, EvidenceWord


def test_transcription_adapter_creates_stable_absolute_word_ids():
    source = [TranscriptionSegment(
        text="hello world",
        start=0.5,
        end=2.0,
        words=[
            WordTimestamp("hello", 0.5, 1.0, confidence=0.8, speaker_id=2),
            WordTimestamp("world", 1.1, 2.0, confidence=0.9),
        ],
        avg_logprob=-0.2,
        language="en",
    )]
    transcript = adapt_transcription_segments(
        source,
        source_window_id="ctx-a",
        segment_id_prefix="seg-a",
        time_offset=10.0,
        audio_duration=20.0,
    )

    assert [word.id for word in transcript.words] == [
        "gw:ctx-a:seg-a:0000:w0000",
        "gw:ctx-a:seg-a:0000:w0001",
    ]
    assert [(word.raw_start, word.raw_end) for word in transcript.words] == [(10.5, 11.0), (11.1, 12.0)]
    assert transcript.words[0].speaker_id == 2
    assert GlobalTranscript.from_dict(transcript.to_dict()).to_dict() == transcript.to_dict()


def test_global_speaker_timeline_preserves_canonical_ids_and_derives_speaker_list():
    result = DiarizationResult(
        turns=[SpeakerTurn(0.0, 1.0, 4), SpeakerTurn(0.5, 1.5, 2, overlapped=True)],
        exclusive_turns=[SpeakerTurn(0.0, 0.5, 4)],
        speaker_count=2,
        backend="canonical",
    )
    timeline = adapt_diarization_result(result, duration=2.0)
    assert timeline.speaker_ids == [2, 4]
    assert [turn.speaker_id for turn in timeline.turns] == [4, 2]
    assert GlobalSpeakerTimeline.from_dict(timeline.to_dict()).to_dict() == timeline.to_dict()


def test_decision_projection_repairs_word_order_before_global_ir_validation():
    decision = EvidenceDecision(
        candidate_ids=("candidate-1",),
        decision="keep",
        final_text="hello world",
        final_words=(
            EvidenceWord("word-2", "world", 1.4, 1.8, 0.9),
            EvidenceWord("word-1", "hello", 1.0, 1.3, 0.9),
        ),
        start=1.0,
        end=1.8,
        time_source="native_word_timestamp",
        confidence=0.9,
        risk_score=0.1,
        risk_level="low",
    )

    transcript = decisions_to_global_transcript([decision], audio_duration=2.0)

    assert [word.text for word in transcript.words] == ["hello", "world"]
    assert transcript.segments[0].word_ids == [
        "decision:000001:word:0000:word-1",
        "decision:000001:word:0001:word-2",
    ]
    assert transcript.diagnostics["word_order_repaired_count"] == 1
