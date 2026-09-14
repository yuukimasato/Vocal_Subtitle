"""RunContext 与阶段协议行为(2026-09-15 重构计划 Task 2)。

- 上下文默认值与懒创建 stats;
- 诊断按阶段分组合并;
- 取消令牌透传;
- 两个上下文实例之间完全隔离(无共享可变默认值);
- Stage 基类计时、失败记录与重新抛出。
"""

import pytest

from vocal_subtitle.asr.window_execution import CancellationToken
from vocal_subtitle.application.run_context import RunContext
from vocal_subtitle.application.stage_protocol import (
    STAGE_STATUS_FAILED,
    STAGE_STATUS_OK,
    Stage,
)


def _context(**overrides):
    from pathlib import Path

    base = dict(input_path=Path("in.wav"))
    base.update(overrides)
    return RunContext(**base)


def test_context_defaults_and_lazy_stats():
    context = _context()

    assert context.output_path is None
    assert context.output_format == "srt"
    assert context.skip_separation is False
    assert context.audio is None
    assert context.events == []
    assert context.diagnostics == {}
    assert context.cancelled() is False

    stats = context.ensure_stats()
    assert context.stats is stats  # 同一实例
    assert stats.duration_seconds == 0.0


def test_diagnostics_merge_by_stage_in_insertion_order():
    context = _context()

    context.add_diagnostic("asr", {"status": "ok", "elapsed_seconds": 1.5})
    context.add_diagnostic("asr", {"status": "degraded"})
    context.add_diagnostic("vad", {"status": "ok"})

    assert len(context.diagnostics["asr"]) == 2
    assert context.diagnostics_for("asr")[0]["elapsed_seconds"] == 1.5
    assert context.diagnostics_for("vad") == ({"status": "ok"},)
    with pytest.raises(ValueError):
        context.add_diagnostic("", {"status": "ok"})


def test_cancellation_token_propagates():
    context = _context(cancellation_token=CancellationToken())

    assert context.cancelled() is False
    context.cancel("user_abort")
    assert context.cancelled() is True
    with pytest.raises(Exception):
        context.raise_if_cancelled()


def test_two_contexts_are_isolated():
    first = _context()
    second = _context()

    first.events.append(object())
    first.add_diagnostic("asr", {"status": "ok"})
    first.audio = object()
    first.cancel("stop_first")

    assert second.events == []
    assert second.diagnostics == {}
    assert second.audio is None
    assert second.cancelled() is False


class _RecordingStage(Stage):
    name = "recording"

    def run(self, context):
        return {"status": STAGE_STATUS_OK, "detail": "done"}


class _FailingStage(Stage):
    name = "failing"

    def run(self, context):
        raise RuntimeError("boom")


def test_stage_records_timing_and_status():
    context = _context()

    _RecordingStage().execute(context)

    entries = context.diagnostics_for("recording")
    assert len(entries) == 1
    assert entries[0]["status"] == "ok"
    assert entries[0]["detail"] == "done"
    assert entries[0]["elapsed_seconds"] >= 0.0


def test_stage_failure_records_diagnostic_then_reraises():
    context = _context()

    with pytest.raises(RuntimeError, match="boom"):
        _FailingStage().execute(context)

    entries = context.diagnostics_for("failing")
    assert entries[0]["status"] == STAGE_STATUS_FAILED
    assert "boom" in entries[0]["error"]
