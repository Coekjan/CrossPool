"""Background thread lifecycle helpers."""

from __future__ import annotations

import threading
from collections.abc import Callable

__all__ = ["BackgroundThread"]


class BackgroundThread:
    """Own one restartable daemon thread and its stop signal.

    Args:
        name: Thread name used in debuggers and logs.
        target: Worker callable. It receives the stop event that callers set
            through :meth:`stop`.
        join_timeout_s: Maximum seconds to wait for the worker to stop.
        daemon: Whether to create the underlying thread as a daemon thread.

    Side Effects:
        Starts, stops, and joins one process-local Python thread. The target
        owns all business behavior; this class only owns lifecycle mechanics.
    """

    def __init__(
        self,
        *,
        name: str,
        target: Callable[[threading.Event], None],
        join_timeout_s: float,
        daemon: bool = True,
    ) -> None:
        """Create a stopped background thread owner."""

        self.name = name
        self.target = target
        self.join_timeout_s = join_timeout_s
        self.daemon = daemon
        self.lock = threading.Lock()
        self.stop_signal = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.failure: Exception | None = None
        self.closed = False

    @classmethod
    def periodic(
        cls,
        *,
        name: str,
        interval_s: float,
        target: Callable[[], bool | None],
        join_timeout_s: float,
        daemon: bool = True,
    ) -> BackgroundThread:
        """Create a stopped periodic background thread owner.

        Args:
            name: Thread name used in debuggers and logs.
            interval_s: Delay between successful target calls.
            target: Single-iteration callable. Returning ``False`` stops the
                periodic loop without recording a failure; returning ``None``
                or ``True`` continues.
            join_timeout_s: Maximum seconds to wait for the worker to stop.
            daemon: Whether to create the underlying thread as a daemon thread.

        Returns:
            Background thread owner whose target runs periodically.
        """

        def run_periodically(stop_event: threading.Event) -> None:
            while not stop_event.is_set():
                if target() is False:
                    break
                if stop_event.wait(interval_s):
                    break

        return cls(
            name=name,
            target=run_periodically,
            join_timeout_s=join_timeout_s,
            daemon=daemon,
        )

    @property
    def thread(self) -> threading.Thread | None:
        """Return the current thread object, if one has been started."""

        with self.lock:
            return self.worker_thread

    @property
    def stop_event(self) -> threading.Event:
        """Return the stop event passed to the current or next worker run."""

        with self.lock:
            return self.stop_signal

    @property
    def is_running(self) -> bool:
        """Return whether the owned thread currently reports itself alive."""

        with self.lock:
            thread = self.worker_thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the worker unless a live worker already exists.

        Raises:
            RuntimeError: If the owner has been closed.
            Exception: If a previous target run failed.
        """

        self.raise_if_failed()
        with self.lock:
            if self.closed:
                raise RuntimeError(f"cannot restart closed background thread {self.name}")
            if self.worker_thread is not None and self.worker_thread.is_alive():
                return
            self.stop_signal = threading.Event()
            self.worker_thread = threading.Thread(
                target=self.run_target,
                args=(self.stop_signal,),
                name=self.name,
                daemon=self.daemon,
            )
            self.worker_thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and require it to terminate."""

        with self.lock:
            thread = self.worker_thread
            stop_event = self.stop_signal
        if thread is None:
            return
        stop_event.set()
        if threading.current_thread() is not thread:
            thread.join(self.join_timeout_s)
        if thread.is_alive():
            raise RuntimeError(f"background thread {self.name} did not stop within {self.join_timeout_s} seconds")
        with self.lock:
            if self.worker_thread is thread:
                self.worker_thread = None

    def close(self) -> None:
        """Stop the worker and prevent future restarts."""

        self.stop()
        with self.lock:
            self.closed = True

    def raise_if_failed(self) -> None:
        """Raise the first fatal exception escaped by the worker target."""

        with self.lock:
            failure = self.failure
        if failure is not None:
            raise failure

    def run_target(self, stop_event: threading.Event) -> None:
        """Run the target and retain its first ordinary exception."""

        try:
            self.target(stop_event)
        except Exception as exc:
            with self.lock:
                if self.failure is None:
                    self.failure = exc
