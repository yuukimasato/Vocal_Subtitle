"""legacy 路径使用计数(2026-09-15 重构计划 Task 9)。

移除 legacy 路径必须以使用证据为前提(黄金集/兼容性测试 Establish
callers)。本环境无法建立完整证据,因此本任务只落地:
- legacy 入口使用计数器(每次进入 +1);
- 计数器进入 stats.quality_diagnostics["legacy_path_usage"];
- 已知 legacy 入口仍然存在(有调用者,不得移除)。
"""

from types import SimpleNamespace

from vocal_subtitle.application.asr_path import PipelineASRPathMixin


def _pipeline():
    """带计数器的最小宿主对象(不加载模型)。"""

    class _Host(PipelineASRPathMixin):
        # Mixin 方法只依赖 config/run_context 少量属性;计数器独立可用。
        def __init__(self):
            self._legacy_path_usage = {}

    return _Host()


def test_legacy_use_counter_increments_per_entry():
    host = _pipeline()

    host._record_legacy_use("global_transcription_legacy")
    host._record_legacy_use("global_transcription_legacy")
    host._record_legacy_use("legacy_degraded_fallback")

    assert host._legacy_path_usage == {
        "global_transcription_legacy": 2,
        "legacy_degraded_fallback": 1,
    }


def test_legacy_usage_is_reported_into_stats():
    host = _pipeline()
    host._record_legacy_use("legacy_degraded_fallback")
    stats = SimpleNamespace(quality_diagnostics={})

    host._report_legacy_usage(stats)

    assert stats.quality_diagnostics["legacy_path_usage"] == {
        "legacy_degraded_fallback": 1,
    }


def test_known_legacy_entry_points_still_exist():
    # 这些入口当前均有真实调用者;零证据不得移除(Task 9 前置条件)。
    for name in (
        "_run_global_transcription_path_legacy",
        "_resolve_asr_path",
        "_run_offline_production_review",
    ):
        assert hasattr(PipelineASRPathMixin, name), f"{name} 不应被移除"


def test_record_helper_tolerates_missing_counter_dict():
    host = _pipeline()
    host._legacy_path_usage = None  # 异常初始化也不得崩溃

    host._record_legacy_use("legacy")

    assert host._legacy_path_usage == {"legacy": 1}
