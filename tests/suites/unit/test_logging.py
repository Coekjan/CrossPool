from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from typing import Literal

import pytest

import xpool.logging
from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from xpool.config import LoggingConfig
from xpool.native import RuntimeRole

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


class TtyBuffer(io.StringIO):
    """String buffer with controllable terminal detection."""

    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.fixture
def runtime_logger() -> Iterator[logging.Logger]:
    """Provide a clean xpool logger and restore its handlers after a test."""

    logger = logging.getLogger("xpool")
    handlers = tuple(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    for handler in handlers:
        logger.removeHandler(handler)
    yield logger
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate


def install_logging_config(
    *, level: Literal["debug", "info", "warning", "error", "critical"] = "info", color: bool = True
) -> None:
    """Install a synthetic config with the requested logging policy."""

    config = synthetic_config().model_copy(update={"logging": LoggingConfig(level=level, color=color)})
    install_test_config(config)


def test_runtime_logging_filters_and_formats_records(
    monkeypatch: pytest.MonkeyPatch,
    runtime_logger: logging.Logger,
) -> None:
    install_logging_config()
    stream = TtyBuffer(tty=False)
    monkeypatch.setattr(xpool.logging.sys, "stderr", stream)

    xpool.logging.configure(RuntimeRole.FFNAGENT)
    child = logging.getLogger("xpool.test")
    child.debug("hidden event")
    child.info("ready rank=%s", 2)

    output = stream.getvalue()
    assert "hidden event" not in output
    assert "\x1b[" not in output
    assert re.search(r"^\d{4}-\d{2}-\d{2}T[^ ]+ INFO     FfnAgent \| ready rank=2\n$", output)


@pytest.mark.parametrize(
    ("color", "tty", "colored"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_runtime_logging_colors_only_configured_tty(
    monkeypatch: pytest.MonkeyPatch,
    runtime_logger: logging.Logger,
    color: bool,
    tty: bool,
    colored: bool,
) -> None:
    install_logging_config(color=color)
    stream = TtyBuffer(tty=tty)
    monkeypatch.setattr(xpool.logging.sys, "stderr", stream)

    xpool.logging.configure(RuntimeRole.DAEMON)
    logging.getLogger("xpool.test").warning("warning event")

    output = stream.getvalue()
    assert ("\x1b[33m" in output) is colored
    assert "warning event" in output


def test_runtime_logging_configuration_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    runtime_logger: logging.Logger,
) -> None:
    install_logging_config()
    stream = TtyBuffer(tty=False)
    monkeypatch.setattr(xpool.logging.sys, "stderr", stream)

    xpool.logging.configure(RuntimeRole.INSTANCE)
    xpool.logging.configure(RuntimeRole.INSTANCE)
    logging.getLogger("xpool.test").info("one event")

    assert stream.getvalue().count("one event") == 1


def test_runtime_logging_preserves_exception_traceback(
    monkeypatch: pytest.MonkeyPatch,
    runtime_logger: logging.Logger,
) -> None:
    install_logging_config()
    stream = TtyBuffer(tty=False)
    monkeypatch.setattr(xpool.logging.sys, "stderr", stream)

    xpool.logging.configure(RuntimeRole.INSTANCE)
    try:
        raise ValueError("broken runtime")
    except ValueError:
        logging.getLogger("xpool.test").exception("instance failed")

    output = stream.getvalue()
    assert "Traceback (most recent call last)" in output
    assert "ValueError: broken runtime" in output
