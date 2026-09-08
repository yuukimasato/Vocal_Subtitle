import pytest

from vocal_subtitle.asr.engine_pairing import EnginePairRouter


def test_auto_pair_keeps_selected_whisper_as_primary_for_multilingual():
    decision = EnginePairRouter().route(
        language="en",
        primary="auto",
        secondary="auto",
        selected_primary="faster-whisper",
    )

    assert decision.primary == "faster-whisper"
    assert decision.primary_family == "whisper"
    assert decision.secondary == "qwen"
    assert decision.secondary_family == "qwen"
    assert decision.degraded is False


def test_funasr_non_chinese_falls_back_to_whisper():
    decision = EnginePairRouter().route(
        language="en", primary="funasr", secondary="auto"
    )

    assert decision.primary == "faster-whisper"
    assert decision.secondary == "qwen"
    assert decision.degraded is True
    assert decision.fallback_reason == "funasr_requires_chinese_language"


def test_same_family_secondary_is_rejected_by_default():
    decision = EnginePairRouter().route(
        language="en",
        primary="faster-whisper",
        secondary="whisper-cpp",
    )

    assert decision.secondary is None
    assert decision.degraded is True
    assert decision.fallback_reason == "same_family_secondary_rejected"


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="unsupported pair policy"):
        EnginePairRouter().route(language="en", policy="vote")
