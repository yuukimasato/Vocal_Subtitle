"""WebUI server lifecycle helpers.

The WebUI must keep running after a pipeline task finishes.  Uvicorn owns
the actual event loop, so this module only adds diagnostics around its
normal signal handling instead of changing shutdown semantics.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from types import FrameType
from typing import Any

import uvicorn

logger = logging.getLogger(__name__)


def install_exception_logging() -> None:
    """Log uncaught exceptions in the main thread and WebUI worker threads."""

    previous_excepthook = sys.excepthook

    def excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any,
    ) -> None:
        logger.critical(
            "Uncaught exception terminated the WebUI process",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        previous_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = excepthook

    def thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Uncaught exception in WebUI thread %s",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = thread_excepthook


class DiagnosticServer(uvicorn.Server):
    """Uvicorn server that records why shutdown was requested."""

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        try:
            signal_name = signal.Signals(sig).name
        except ValueError:
            signal_name = str(sig)
        stack = "".join(traceback.format_stack(frame)) if frame else "<no frame>"
        logger.warning(
            "WebUI server received shutdown signal %s (%s)\n%s",
            sig,
            signal_name,
            stack,
        )
        super().handle_exit(sig, frame)


def run_server(
    app: Callable[..., Any],
    *,
    host: str,
    port: int,
) -> None:
    """Run the WebUI until Uvicorn receives an explicit shutdown signal."""

    install_exception_logging()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = DiagnosticServer(config)
    logger.info("Starting WebUI server on http://%s:%s", host, port)
    server.run()
    logger.info("WebUI server stopped (should_exit=%s)", server.should_exit)
