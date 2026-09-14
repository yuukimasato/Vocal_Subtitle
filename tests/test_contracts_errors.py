"""结构化错误契约(2026-09-15 重构计划 Task 6)。"""

import pytest

from vocal_subtitle.contracts.errors import (
    ASRExecutionError,
    AudioDecodeError,
    DependencyUnavailableError,
    PipelineError,
    RecoverableStageError,
    TimelineViolationError,
    classify_exception,
    translate_exception,
)


def test_domain_errors_carry_stable_categories():
    assert DependencyUnavailableError("x").category == "dependency_unavailable"
    assert AudioDecodeError("x").category == "audio_decode"
    assert ASRExecutionError("x").category == "asr_execution"
    assert TimelineViolationError("x").category == "timeline_violation"
    assert RecoverableStageError("x").category == "stage_recoverable"
    assert DependencyUnavailableError("x").recoverable is True
    assert TimelineViolationError("x").recoverable is False


def test_error_serializes_to_stable_status():
    error = ASRExecutionError("funasr crashed", detail="seg 7")

    payload = error.to_dict()

    assert payload == {
        "category": "asr_execution",
        "message": "funasr crashed",
        "detail": "seg 7",
        "recoverable": True,
    }


def test_classify_maps_known_and_unknown_exceptions():
    assert classify_exception(ImportError("No module named 'torch'")) == (
        "dependency_unavailable"
    )
    assert classify_exception(FileNotFoundError("in.wav")) == "input_not_found"
    assert classify_exception(ValueError("weird")) == "execution_failed"
    assert classify_exception(
        DependencyUnavailableError("whisperx missing")
    ) == "dependency_unavailable"
    assert classify_exception(RuntimeError("faster-whisper is not installed")) == (
        "dependency_unavailable"
    )


def test_translate_wraps_unexpected_defects_without_losing_cause():
    original = ValueError("unexpected defect")

    translated = translate_exception(original)

    assert isinstance(translated, PipelineError)
    assert translated.category == "execution_failed"
    assert translated.__cause__ is None  # 未抛出,仅包装
    with pytest.raises(PipelineError):
        raise translated


def test_translate_keeps_domain_errors_as_is():
    error = ASRExecutionError("boom")

    assert translate_exception(error) is error
