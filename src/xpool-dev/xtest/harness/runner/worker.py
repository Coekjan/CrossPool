"""Run pytest in the supervised task root, including all fixture teardown."""

from __future__ import annotations

import sys
import time
from multiprocessing.connection import Connection

import pytest
import pytest_timeout

from xkit.task import TaskExecutionWindow, TaskRoot


class PytestTaskRoot(TaskRoot):
    """Move validated item timing to supervision when protection activates."""

    def __init__(self, connection: Connection) -> None:
        self.item: pytest.Item | None = None
        self.item_deadline: float | None = None
        super().__init__(connection)

    @pytest.hookimpl(tryfirst=True)
    def pytest_timeout_set_timer(self, item: pytest.Item, settings: pytest_timeout.Settings) -> bool | None:
        """Retain the original item start; unprotected items use the native timer."""

        with self.condition:
            self.item = item
            now = time.monotonic()
            self.item_deadline = now + settings.timeout
            if self.requested_event is not None:
                self.connection.send(TaskExecutionWindow(self.identity, self.item_deadline, now))
                return True
        return None

    @pytest.hookimpl(tryfirst=True)
    def pytest_timeout_cancel_timer(self, item: pytest.Item) -> bool | None:
        """End this interval after the timeout plugin's selected item boundary."""

        with self.condition:
            if self.item is not item:
                return None
            self.item = None
            self.item_deadline = None
            if self.requested_event is not None:
                self.connection.send(TaskExecutionWindow(self.identity, None, time.monotonic()))
                return True
        return None

    def activate(self) -> None:
        """Cancel the independent timer before permitting protected startup."""

        with self.condition:
            if self.item is not None:
                pytest_timeout.pytest_timeout_cancel_timer(self.item)
            if self.requested_event is None:
                self.connection.send(TaskExecutionWindow(self.identity, self.item_deadline, time.monotonic()))
        super().activate()


def main() -> int:
    """Preserve pytest arguments and verdict in this one task-root process."""

    root = PytestTaskRoot.from_environment()
    try:
        return int(pytest.main(sys.argv[1:], plugins=[] if root is None else [root]))
    finally:
        if root is not None:
            root.finish()


if __name__ == "__main__":
    raise SystemExit(main())
