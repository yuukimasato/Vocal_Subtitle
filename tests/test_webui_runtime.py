"""WebUI process lifecycle tests."""

import logging
import signal

import uvicorn
from fastapi.testclient import TestClient

from vocal_subtitle.webui import runtime
from vocal_subtitle.webui.app import create_app


def test_diagnostic_server_logs_shutdown_signal(monkeypatch, caplog):
    called = {}

    def base_handle_exit(self, sig, frame):
        called["signal"] = sig

    monkeypatch.setattr(uvicorn.Server, "handle_exit", base_handle_exit)
    server = runtime.DiagnosticServer(uvicorn.Config(lambda scope, receive, send: None))

    with caplog.at_level(logging.WARNING, logger="vocal_subtitle.webui.runtime"):
        server.handle_exit(signal.SIGTERM, None)

    assert called["signal"] == signal.SIGTERM
    assert "shutdown signal 15 (SIGTERM)" in caplog.text


def test_run_server_uses_diagnostic_server(monkeypatch):
    captured = {}

    class FakeServer:
        should_exit = True

        def __init__(self, config):
            captured["config"] = config

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(runtime, "DiagnosticServer", FakeServer)
    monkeypatch.setattr(
        runtime,
        "install_exception_logging",
        lambda: captured.update(installed=True),
    )

    app = lambda scope, receive, send: None
    runtime.run_server(app, host="127.0.0.1", port=9876)

    assert captured["installed"] is True
    assert captured["ran"] is True
    assert captured["config"].app is app
    assert captured["config"].host == "127.0.0.1"
    assert captured["config"].port == 9876


def test_create_app_runs_stale_task_cleanup_during_lifespan(monkeypatch):
    called = []
    monkeypatch.setattr(
        "vocal_subtitle.webui.app._mark_stale_running_tasks",
        lambda: called.append(True),
    )

    with TestClient(create_app()):
        pass

    assert called == [True]
