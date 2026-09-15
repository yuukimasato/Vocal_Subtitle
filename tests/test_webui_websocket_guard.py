"""WebSocket 根路径健壮性(2026-09-15 CLI/WebUI 链路测试发现)。

未匹配任何 /ws 路由的 WebSocket 请求曾落入根 StaticFiles 挂载,
StaticFiles 断言 scope type 为 http 而抛 AssertionError → 500。
修复后应被干净拒绝(403),前端正常路径(/ws/tasks/{id})不受影响。
"""

import pytest
from fastapi.testclient import TestClient

from vocal_subtitle.webui.app import create_app


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


def test_unmatched_websocket_path_is_rejected_cleanly(client):
    with pytest.raises(Exception) as exc_info:
        with client.websocket_connect("/ws/tasks"):
            pass
    # TestClient 对非 101 升级抛 WebSocketDisconnect;不得是 500/AssertionError。
    message = str(exc_info.value)
    assert "500" not in message
    assert "AssertionError" not in message


def test_matched_websocket_path_still_accepts_ping(client):
    with client.websocket_connect("/ws/tasks/smoke-test") as ws:
        ws.send_json({"type": "ping"})
        reply = ws.receive_json()
        assert reply == {"type": "pong"}
