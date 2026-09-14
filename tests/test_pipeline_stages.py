"""生命周期阶段对象(2026-09-15 重构计划 Task 3)。

用合成 pipeline(不加载模型)验证:
- 每个阶段可独立以 RunContext 执行;
- 诊断与 state 写入符合契约;
- Preflight 失败结果被正确传递;
- MappingStage 组合 postprocess 交接不改变事件顺序。
"""

from types import SimpleNamespace

from vocal_subtitle.application.run_context import RunContext
from vocal_subtitle.application.stages.asr_stage import ASRStage
from vocal_subtitle.application.stages.audio_stage import AudioStage
from vocal_subtitle.application.stages.mapping_stage import MappingStage
from vocal_subtitle.application.stages.preflight_stage import PreflightStage


def _context():
    from pathlib import Path

    return RunContext(input_path=Path("in.wav"))


def test_preflight_stage_returns_failure_result_from_pipeline():
    failure = {"status": "failed", "events": []}

    class Pipeline:
        def _run_preflight_stage(self, context):
            assert isinstance(context, RunContext)
            return failure

    result = PreflightStage(Pipeline()).execute(_context())

    assert result is failure


def test_preflight_stage_passes_context_through():
    class Pipeline:
        def __init__(self):
            self.seen = None

        def _run_preflight_stage(self, context):
            self.seen = context
            return None

    pipeline = Pipeline()
    context = _context()

    assert PreflightStage(pipeline).execute(context) is None
    assert pipeline.seen is context


def test_audio_stage_forwards_progress_callback_and_returns_context():
    seen = {}

    class Pipeline:
        def _run_audio_stage(self, context, progress_callback=None):
            seen["callback"] = progress_callback
            seen["context"] = context
            return context

    callback = lambda *args: None
    context = _context()

    result = AudioStage(Pipeline()).execute(context, progress_callback=callback)

    assert result is context
    assert seen["callback"] is callback
    assert seen["context"] is context


def test_asr_stage_executes_and_returns_context():
    class Pipeline:
        def __init__(self):
            self.executed = False

        def _run_asr_stage(self, context):
            self.executed = True
            context.state["events"] = ["e1"]

    pipeline = Pipeline()
    context = _context()

    result = ASRStage(pipeline).execute(context)

    assert result is context
    assert pipeline.executed is True
    assert context.state["events"] == ["e1"]


def test_mapping_stage_preserves_event_order_through_postprocess():
    class Pipeline:
        def _post_process_events(self, events, vocals_path, audio, sample_rate, stats, **kwargs):
            self.kwargs = kwargs
            return list(reversed(events))

    pipeline = Pipeline()
    context = _context()
    context.stats = SimpleNamespace()
    events = ["a", "b", "c"]

    result = MappingStage(pipeline).execute(
        context, events, audio="audio", sample_rate=16000,
        vocals_path="vocals", ffmpeg_unified_result={"skeleton": []},
    )

    assert result == list(reversed(events))
    assert pipeline.kwargs["ffmpeg_unified_result"] == {"skeleton": []}
