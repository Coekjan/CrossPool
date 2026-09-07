"""Process-local runtime logging configuration."""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from xpool.config import get_global_config
from xpool.native import RuntimeRole

__all__ = ["configure"]

HANDLER_NAME = "xpool-runtime"
ROLE_LABELS = {
    RuntimeRole.DAEMON: "Daemon",
    RuntimeRole.INSTANCE: "Instance",
    RuntimeRole.ATNAGENT: "AtnAgent",
    RuntimeRole.FFNAGENT: "FfnAgent",
}
LEVEL_COLORS = {
    logging.DEBUG: "\x1b[36m",
    logging.INFO: "\x1b[32m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[31m",
}
COLOR_RESET = "\x1b[0m"


class ConsoleFormatter(logging.Formatter):
    """Format xpool runtime records as compact role-aware terminal lines."""

    def __init__(self, role: RuntimeRole, *, color: bool) -> None:
        """Create a formatter for one process role and color policy."""

        super().__init__()
        self.role = ROLE_LABELS[role]
        self.color = color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Render one record with local timestamp, level, role, and message."""

        timestamp = datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")
        level = f"{record.levelname:<8}"
        if self.color:
            level = f"{LEVEL_COLORS.get(record.levelno, '')}{level}{COLOR_RESET}"
        return f"{timestamp} {level} {self.role:<8} | {super().format(record)}"


def configure(role: RuntimeRole) -> None:
    """Configure the xpool logger from the process-global runtime policy.

    Repeated calls replace only the handler installed by this function. The
    logger writes to stderr and disables propagation without changing root,
    SGLang, or Uvicorn logging.
    """

    config = get_global_config().logging
    runtime_logger = logging.getLogger("xpool")
    for handler in tuple(runtime_logger.handlers):
        if handler.name == HANDLER_NAME:
            runtime_logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler()
    handler.name = HANDLER_NAME
    handler.setFormatter(ConsoleFormatter(role, color=config.color))
    runtime_logger.addHandler(handler)
    runtime_logger.setLevel(getattr(logging, config.level.upper()))
    runtime_logger.propagate = False
