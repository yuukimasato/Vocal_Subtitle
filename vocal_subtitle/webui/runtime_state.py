"""Shared WebUI runtime state.

Route modules depend on this state holder instead of importing ``webui.api``.
The properties intentionally resolve compatibility overrides from the legacy
module at call time, so existing tests and integrations that monkeypatch
``vocal_subtitle.webui.api._task_store`` continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from ..utils.task_history import TaskHistoryManager
from .websocket import ws_manager

_DEFAULT_TASK_STORE: Dict[str, Dict[str, Any]] = {}
_DEFAULT_SHADOW_EVALUATORS: Dict[str, Any] = {}
_DEFAULT_TASK_HISTORY = TaskHistoryManager()
_DEFAULT_UPLOAD_DIR = Path(__file__).parent.parent.parent / "cache" / "uploads"
_DEFAULT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _legacy_api():
    return sys.modules.get("vocal_subtitle.webui.api")


class WebUIRuntimeState:
    """Compatibility-aware access to process-local WebUI state."""

    @property
    def task_store(self) -> Dict[str, Dict[str, Any]]:
        api = _legacy_api()
        return getattr(api, "_task_store", _DEFAULT_TASK_STORE)

    @property
    def shadow_evaluators(self) -> Dict[str, Any]:
        api = _legacy_api()
        return getattr(api, "_shadow_evaluators", _DEFAULT_SHADOW_EVALUATORS)

    @property
    def task_history(self):
        api = _legacy_api()
        return getattr(api, "_task_history", _DEFAULT_TASK_HISTORY)

    @property
    def upload_dir(self) -> Path:
        api = _legacy_api()
        return getattr(api, "UPLOAD_DIR", _DEFAULT_UPLOAD_DIR)

    @property
    def websocket(self):
        return ws_manager


state = WebUIRuntimeState()

__all__ = ["WebUIRuntimeState", "state"]
