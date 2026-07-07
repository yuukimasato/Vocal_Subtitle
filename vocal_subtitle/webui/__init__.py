"""Web UI 模块 — FastAPI + WebSocket 驱动的图形界面

提供 REST API 和实时 WebSocket 进度推送，
前端为单文件 SPA (static/index.html)。

使用:
    from vocal_subtitle.webui.app import create_app
    app = create_app()
"""

from .app import create_app

__all__ = ["create_app"]
