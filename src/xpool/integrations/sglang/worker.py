"""Serving worker exit boundaries around the pinned SGLang process entries."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from types import FrameType

import psutil
from sglang.srt.managers import data_parallel_controller, scheduler
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs

from xpool.utils.mps import is_terminal_device_error
from xpool.utils.procs import ProcUniqId

logger = logging.getLogger(__name__)
worker_lifecycle: WorkerLifecycle | None = None


@dataclass(slots=True)
class WorkerLifecycle:
    """Retain failure and local release facts for one scheduler process.

    Background callbacks and main-thread hooks share this owner. Normal return
    requires successful loop completion and local release. Failure notification
    requests parent-owned retirement; only a locally observed terminal device
    result authorizes immediate nonzero exit here. Neither proves whole-world
    resource retirement.

    Attributes:
        parent: Exact immediate serving parent, including the DP controller.
        failure: First original failure, preserved across later cleanup errors.
        loop_returned: Event loop completed without an exception.
        normal_release_complete: Engine release and Instance detach succeeded.
    """

    parent: ProcUniqId
    failure: BaseException | None = None
    loop_returned: bool = False
    normal_release_complete: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def current() -> WorkerLifecycle:
        """Require the lifecycle retained by the actual scheduler spawn entry."""

        if worker_lifecycle is None:
            raise RuntimeError("xpool scheduler has no worker lifecycle")
        return worker_lifecycle

    def fail(self, error: BaseException) -> None:
        """Retain the first failure and request parent-owned retirement.

        Called on the reporting thread. Every error is checked for a terminal
        device result, including later errors after the original failure. Such
        a result exits nonzero without another device synchronization. Missing
        metadata and failed notification supply no permission to exit.
        """

        with self.lock:
            first_failure = self.failure is None
            if first_failure:
                self.failure = error
        if first_failure:
            logger.error(
                "serving worker failed pid=%s", os.getpid(), exc_info=(type(error), error, error.__traceback__)
            )
            try:
                self.parent.send_signal(signal.SIGQUIT)
            except (OSError, psutil.Error):
                logger.exception("serving failure notification failed pid=%s parent=%s", os.getpid(), self.parent.pid)
        if is_terminal_device_error(error):
            logger.error("terminal device result requires worker exit pid=%s error=%s", os.getpid(), error)
            os._exit(1)

    def finish(self) -> None:
        """Allow proven normal local return or retain the worker for its owner.

        Upstream return alone is insufficient because its entry swallows
        scheduler errors. Unconfirmed local release keeps this process alive
        until parent-controlled termination; no new cleanup deadline is added.
        """

        with self.lock:
            normal = self.failure is None and self.loop_returned and self.normal_release_complete
        if normal:
            return
        if self.failure is None:
            self.fail(RuntimeError("scheduler returned without confirmed normal local release"))
        while True:
            time.sleep(1.0)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: int | None,
    pipe_writer: Connection,
    display_tp_rank: int | None = None,
    display_dp_rank: int | None = None,
    display_moe_ep_rank: int | None = None,
) -> None:
    """Run the existing scheduler under its resource-owning exit boundary.

    Parameter names and calling convention belong to SGLang's spawn target.
    Main-thread hooks use the scoped owner; background producers receive its
    bound failure callback. The original scheduler implementation is retained.
    """

    global worker_lifecycle
    previous = worker_lifecycle
    lifecycle = WorkerLifecycle(ProcUniqId(os.getppid()))
    worker_lifecycle = lifecycle
    try:
        scheduler.run_scheduler_process(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            attn_cp_rank,
            moe_dp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            pipe_writer,
            display_tp_rank,
            display_dp_rank,
            display_moe_ep_rank,
        )
    except BaseException as error:
        lifecycle.fail(error)
    lifecycle.finish()
    worker_lifecycle = previous


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer: Connection,
    run_scheduler_process_func: Callable[..., object] = run_scheduler_process,
) -> None:
    """Forward controller signals and retain its actual serving-parent boundary.

    Exiting the controller can trigger upstream parent-death worker kills.
    The serving owner therefore retires this controller after its workers;
    an upstream entry return does not independently authorize that exit.
    """

    parent = ProcUniqId(os.getppid())

    def forward_signal(signum: int, frame: FrameType | None) -> None:
        try:
            parent.send_signal(signal.Signals(signum))
        except (OSError, psutil.Error):
            logger.exception("controller notification failed pid=%s parent=%s", os.getpid(), parent.pid)

    for signum in (signal.SIGQUIT, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward_signal)
    try:
        load_plugins()
        data_parallel_controller.run_data_parallel_controller_process(
            server_args, port_args, pipe_writer, run_scheduler_process_func
        )
    except BaseException:
        logger.exception("serving controller failed pid=%s", os.getpid())
        forward_signal(signal.SIGQUIT, None)
    while True:
        time.sleep(1.0)
