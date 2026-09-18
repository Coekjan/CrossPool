"""Terminal-aware task status formatting for the xtest command."""

from __future__ import annotations

import logging
import os
import sys


class TaskFormatter(logging.Formatter):
    """Color only task state words, leaving artifact paths copyable."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        """Prefix task messages with a terminal-aware state."""

        status = getattr(record, "status", None)
        message = record.getMessage()
        if status is None:
            return message
        if self.color:
            code = {"RUNNING": "33", "PASSED": "32", "FAILED": "31"}[status]
            status = f"\x1b[{code}m{status}\x1b[0m"
        return f"{status} {message}"


def configure_console() -> None:
    """Install one xtest-only stdout handler without changing runtime logs."""

    logger = logging.getLogger("xtest")
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TaskFormatter(color=sys.stdout.isatty() and "NO_COLOR" not in os.environ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
