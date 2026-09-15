"""LLM 优化 agent loop 回归测试。

背景 bug：optimizer.py 中 ``_repair`` 被误缩进为模块级函数
``_clean_for_compare`` 的嵌套死代码（位于其 ``return`` 之后），
``agent_loop`` 调用 ``self._repair(...)`` 必然抛 AttributeError，
被 ``_optimize_chunk`` 的裸 except 静默吞掉 →
所有 LLM 优化结果整体回退为原文，``update_callback`` 永不触发
（UI 进度在整段 LLM 优化期间停在 0%，形似卡死）。
"""

import json
import logging
from types import SimpleNamespace

import llm_subtitle_optimizer.optimizer as optimizer_mod
from llm_subtitle_optimizer.optimizer import SubtitleOptimizer


def _fake_llm_response(payload: dict):
    content = json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_repair_method_is_reachable():
    """_repair 必须是类方法，而不是 _clean_for_compare 里的死代码。"""
    assert callable(getattr(SubtitleOptimizer, "_repair", None)), (
        "SubtitleOptimizer._repair missing — agent_loop will raise "
        "AttributeError on every chunk and silently discard LLM results"
    )


def test_agent_loop_applies_llm_result_and_fires_callback(monkeypatch):
    optimized = {
        "1": "大家好啊，今天呢",
        "2": "那么它其实就是这样的",
    }
    calls = []

    def fake_call_llm(messages, **kwargs):
        calls.append(messages)
        return _fake_llm_response(optimized)

    # 测试环境不强制安装 [llm] extra，用标准 json 顶替 json_repair
    import json as _json

    monkeypatch.setattr(optimizer_mod, "_ensure_json_repair", lambda: _json)
    monkeypatch.setattr(optimizer_mod, "call_llm", fake_call_llm)

    updates = []
    opt = SubtitleOptimizer(
        model="fake-model",
        thread_num=1,
        batch_num=2,
        base_url="http://localhost:9",
        api_key="fake",
        update_callback=updates.append,
    )
    result = opt.optimize(
        {
            "1": "大家好啊今天呢",
            "2": "那么它其实就是这样的",
        }
    )

    assert calls, "stubbed LLM should have been called"
    assert result["1"] == optimized["1"], (
        f"optimized text must be applied, got fallback: {result}"
    )
    assert updates, "update_callback must fire after chunk completion"
    assert updates[0][0]["optimized_text"] == optimized["1"]


def test_chunk_failure_is_logged_not_silently_swallowed(monkeypatch, caplog):
    """单个 chunk 失败时必须留下日志，而不是无声回退原文。"""

    def failing_call_llm(messages, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(optimizer_mod, "call_llm", failing_call_llm)

    opt = SubtitleOptimizer(
        model="fake-model",
        thread_num=1,
        batch_num=2,
        base_url="http://localhost:9",
        api_key="fake",
    )
    with caplog.at_level(logging.WARNING, logger="llm_subtitle_optimizer.optimizer"):
        result = opt.optimize({"1": "大家好啊今天呢"})

    assert result["1"] == "大家好啊今天呢"  # 回退原文
    assert any(
        "chunk" in r.message.lower() or "失败" in r.message for r in caplog.records
    ), "chunk failure must be logged"
