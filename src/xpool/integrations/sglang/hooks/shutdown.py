"""Serving-owner retirement at the pinned engine's destructive exit boundaries."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing.resource_tracker
import os
import signal
import time
from collections.abc import Callable, Sequence
from multiprocessing.process import BaseProcess
from types import FrameType
from typing import Concatenate, NoReturn, cast

import psutil
import zmq
import zmq.asyncio
from sglang.srt.entrypoints import engine
from sglang.srt.managers.io_struct import ShutdownReq, async_sock_send, sock_send
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.tokenizer_manager import SignalHandler, TokenizerManager
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.server_args import PortArgs
from sglang.srt.utils.watchdog import SubprocessWatchdog

from xpool.config import get_global_config
from xpool.integrations.sglang.hooks.registry import SglangHook, SglangHookSet
from xpool.native import ABI_VERSION
from xpool.service.client import XpoolClient
from xpool.service.wire import MpsClientTermination
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MPS_TERMINATION_TIMEOUT_S, MpsEndpoint
from xpool.utils.procs import ProcUniqId

logger = logging.getLogger(__name__)


class ShutdownHookSet(SglangHookSet):
    """Retain the existing engine's ready workers and first retirement budget.

    Startup signals record cancellation until the original factory returns its
    complete ready world. Destructive exits confirm every affected MPS client
    before signaling host processes. Failure retains this actual owner rather
    than unwinding into another upstream killer.
    """

    def __init__(self) -> None:
        self.deadline: float | None = None
        self.cancelled: int | None = None
        self.launch_started = False
        self.retiring = False
        self.shutdown_requested = False
        self.tokenizer: TokenizerManager | MultiTokenizerRouter | None = None
        self.watchdog: SubprocessWatchdog | None = None
        self.schedulers: tuple[ProcUniqId, ...] | None = None
        self.caches: Sequence[BaseProcess] = ()
        self.terminated_clients: set[ProcUniqId] = set()

    def hooks(self) -> tuple[SglangHook, ...]:
        return (
            SglangHook(
                "sglang.srt.entrypoints.engine.Engine._launch_subprocesses",
                self.around_launch_subprocesses,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.tokenizer_manager.SignalHandler.sigterm_handler",
                self.around_sigterm_handler,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.tokenizer_manager.TokenizerManager._dispatch_to_scheduler",
                self.around_dispatch_to_scheduler,
                HookType.AROUND,
            ),
            SglangHook("sglang.srt.utils.common.kill_process_tree", self.around_kill_process_tree, HookType.AROUND),
            SglangHook(
                "sglang.srt.entrypoints.engine.Engine._terminate_weight_cache_daemons",
                self.around_terminate_weight_cache_daemons,
                HookType.AROUND,
            ),
        )

    def begin_retirement(self) -> float:
        if self.deadline is None:
            self.deadline = time.monotonic() + MPS_CLEANUP_TIMEOUT_S
        return self.deadline

    def startup_signal(self, signum: int, frame: FrameType | None) -> None:
        if self.cancelled is None:
            self.cancelled = signum
        self.begin_retirement()
        if self.schedulers is not None and not self.retiring:
            self.around_kill_process_tree(None, os.getpid(), include_parent=False)
            raise SystemExit(128 + self.cancelled)

    def around_launch_subprocesses[**P, R](
        self,
        original_fn: Callable[Concatenate[type[engine.Engine], P], R],
        engine_type: type[engine.Engine],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        self.launch_started = True
        # The engine parent handles cancellation independently of its launch
        # group. Workers retain their engine-installed signal handlers.
        signal.signal(signal.SIGINT, self.startup_signal)
        signal.signal(signal.SIGTERM, self.startup_signal)
        result = original_fn(engine_type, *args, **kwargs)
        # SGLang 0.5.20 returns six values but annotates only five. This assertion
        # belongs at that pinned factory seam, not at individual consumers.
        ready = cast(
            tuple[
                TokenizerManager | MultiTokenizerRouter | None,
                TemplateManager | None,
                PortArgs,
                engine.SchedulerInitResult,
                SubprocessWatchdog | None,
                Sequence[BaseProcess] | None,
            ],
            result,
        )
        self.tokenizer = ready[0]
        schedulers = ready[3]
        self.watchdog = ready[4]
        self.caches = () if ready[5] is None else ready[5]
        # DP's ready reply supplies actual workers separately from its CPU
        # controller. The general child list also includes detokenizers; use
        # the factory's actual scheduler handles from its retained watchdog.
        dp_workers = [pid for info in schedulers.scheduler_infos for pid in info.get(engine.SCHEDULER_PIDS_ARG, ())]
        workers = dp_workers
        if not workers:
            if self.watchdog is None:
                workers = schedulers.all_child_pids
            else:
                for process, name in zip(self.watchdog._processes, self.watchdog._names, strict=True):
                    if name.startswith("scheduler_"):
                        pid = process.pid
                        if pid is None:
                            raise RuntimeError("ready scheduler has no started process identity")
                        workers.append(pid)
        try:
            self.schedulers = tuple(ProcUniqId(pid) for pid in workers)
        except psutil.NoSuchProcess as error:
            self.retain_owner(error)
        if self.cancelled is not None:
            self.startup_signal(self.cancelled, None)
        return result

    def around_sigterm_handler(
        self,
        original_fn: Callable[[SignalHandler, int | None, FrameType | None], None],
        handler: SignalHandler,
        signum: int | None = None,
        frame: FrameType | None = None,
    ) -> None:
        # Tokenizer construction installs its handler before the factory's
        # scheduler-ready wait. Preserve startup cancellation at that handoff.
        if self.launch_started and self.schedulers is None:
            self.startup_signal(signal.SIGTERM if signum is None else signum, frame)
            return
        self.begin_retirement()
        original_fn(handler, signum, frame)

    def request_scheduler_shutdown(self, deadline: float) -> None:
        if self.watchdog is not None:
            self.watchdog.stop()
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        match self.tokenizer:
            case TokenizerManager():
                # Factory-ready cancellation precedes the ASGI event loop.
                # A synchronous view reuses the same bound socket and pinned
                # message serializer without creating another control channel.
                # get_zmq_socket annotates its optional return_bind_port tuple
                # too; TokenizerManager constructs this socket without it.
                sock_send(
                    zmq.Socket.shadow(cast(zmq.Socket, self.tokenizer.send_to_scheduler)),
                    ShutdownReq(),
                    flags=zmq.DONTWAIT,
                )
            case MultiTokenizerRouter():
                # The router owns an asyncio socket on its existing loop. A new
                # PUSH socket would connect to PUSH, not the scheduler's PULL.
                asyncio.run_coroutine_threadsafe(
                    # The pinned get_zmq_socket annotation omits its async
                    # Context's concrete socket variant.
                    async_sock_send(cast(zmq.asyncio.Socket, self.tokenizer.send_to_scheduler), ShutdownReq()),
                    self.tokenizer._loop,
                ).result(timeout=max(0.0, deadline - time.monotonic()))
            case None:
                # The Rust factory has no Python control socket. Its initialized
                # clients still pass through the confirmed termination boundary.
                return

    def around_dispatch_to_scheduler[R](
        self,
        original_fn: Callable[[TokenizerManager, object], R],
        tokenizer: TokenizerManager,
        message: object,
    ) -> R:
        # The running watchdog already sends ShutdownReq. Remember that actual
        # send so its later killer cannot block on a socket whose peers exited.
        if isinstance(message, ShutdownReq):
            self.begin_retirement()
            self.shutdown_requested = True
        return original_fn(tokenizer, message)

    def around_kill_process_tree(
        self,
        original_fn: Callable[..., None] | None,
        parent_pid: int | None,
        include_parent: bool = True,
        skip_pid: int | None = None,
        wait_timeout: float | None = 60,
    ) -> None:
        if parent_pid is None:
            parent_pid = os.getpid()
            include_parent = False
        try:
            parent = ProcUniqId(parent_pid)
        except psutil.NoSuchProcess:
            return
        deadline = self.begin_retirement()
        self.retiring = True
        try:
            if self.launch_started and self.schedulers is None:
                raise RuntimeError("serving factory did not publish its ready worker world")
            if parent_pid == os.getpid():
                self.request_scheduler_shutdown(deadline)
            targets = [target for target in parent.child_process_ids() if target.pid != skip_pid]
            affected_workers = (
                () if self.schedulers is None else tuple(worker for worker in self.schedulers if worker in targets)
            )
            cooperative_deadline = deadline - MPS_TERMINATION_TIMEOUT_S
            while any(worker.is_alive() for worker in affected_workers) and time.monotonic() < cooperative_deadline:
                time.sleep(min(0.05, max(0.0, cooperative_deadline - time.monotonic())))
            # The root can itself have imported a device-using module; even
            # include_parent=False is followed by its console's ordinary exit.
            context_targets = [*targets, parent] if include_parent or parent_pid == os.getpid() else targets
            self.confirm_context_termination(context_targets, deadline)
            cache_pids = {process.pid for process in self.caches}
            consumers = [target for target in targets if target.pid not in cache_pids]
            for target in reversed(consumers):
                target.send_signal(signal.SIGKILL)
            host_deadline = deadline if wait_timeout is None else min(deadline, time.monotonic() + wait_timeout)
            if wait_timeout is not None or cache_pids:
                while any(target.is_alive() for target in consumers):
                    if time.monotonic() >= host_deadline:
                        raise TimeoutError("serving host-process retirement is unconfirmed")
                    time.sleep(0.05)
            for target in targets:
                if target.pid in cache_pids:
                    # Consumers have retired before an IPC exporter can exit.
                    # TERM preserves the cache's socket/ready-file cleanup.
                    target.send_signal(signal.SIGTERM)
            while any(target.is_alive() for target in targets if target.pid in cache_pids):
                if time.monotonic() >= host_deadline:
                    for target in targets:
                        if target.pid in cache_pids:
                            target.send_signal(signal.SIGKILL)
                    break
                time.sleep(0.05)
            if include_parent:
                parent.send_signal(signal.SIGKILL)
                if parent.is_alive():
                    parent.send_signal(signal.SIGQUIT)
            if wait_timeout is not None:
                while any(target.is_alive() for target in targets) or (include_parent and parent.is_alive()):
                    if time.monotonic() >= host_deadline:
                        raise TimeoutError("serving host-process retirement is unconfirmed")
                    time.sleep(0.05)
        except Exception as error:
            self.retain_owner(error)

    def confirm_context_termination(self, targets: Sequence[ProcUniqId], deadline: float) -> None:
        visibility = visible_uuids()
        endpoint = MpsEndpoint(tuple(visibility[index] for index in get_global_config().atn.devices))
        servers = endpoint.parse_process_ids(
            endpoint.run_control("get_server_list", deadline=deadline), allow_empty=True
        )
        clients = {
            pid
            for server in servers
            for pid in endpoint.parse_process_ids(
                endpoint.run_control(f"get_client_list {server}", deadline=deadline), allow_empty=True
            )
        }
        cache_pids = {process.pid for process in self.caches}
        ordered = sorted(targets, key=lambda target: target.pid in cache_pids)
        client: XpoolClient | None = None
        try:
            for target in ordered:
                if not target.is_alive() or target in self.terminated_clients:
                    continue
                if target.pid not in clients:
                    # An initialized client can detach while finishing host
                    # cleanup. MPS absence supplies no permission to kill it.
                    if (self.schedulers is not None and target in self.schedulers) or target.pid in cache_pids:
                        while target.is_alive() and time.monotonic() < deadline:
                            time.sleep(0.05)
                        if target.is_alive():
                            raise TimeoutError(f"detached serving client {target.pid} has not retired")
                    continue
                if client is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("serving context termination deadline expired")
                    client = XpoolClient(timeout_s=min(5.0, remaining))
                client.terminate_serving_client(
                    MpsClientTermination(
                        pid=target.pid, create_time=target.create_time, abi_version=ABI_VERSION, deadline=deadline
                    )
                )
                self.terminated_clients.add(target)
        finally:
            if client is not None:
                client.close()

    def around_terminate_weight_cache_daemons(
        self,
        original_fn: Callable[[Sequence[BaseProcess], float], None],
        procs: Sequence[BaseProcess],
        timeout: float = 10.0,
    ) -> None:
        if not procs:
            return
        # Engine.shutdown reaches this helper before its scheduler killer.
        # Retire all IPC consumers first; the common guard orders exporters last.
        self.caches = procs
        # Joining these real multiprocessing handles still unregisters their
        # shared resources with the process's retained resource tracker.
        self.around_kill_process_tree(
            None,
            os.getpid(),
            include_parent=False,
            # CPython 3.12 retains this PID; typeshed omits the private field.
            skip_pid=multiprocessing.resource_tracker._resource_tracker._pid,  # ty: ignore[unresolved-attribute]
            wait_timeout=timeout,
        )
        for process in procs:
            process.join(timeout=0)

    def retain_owner(self, error: BaseException) -> NoReturn:
        logger.error(
            "serving cleanup unconfirmed; retaining owner for manual resolution pid=%s error=%s", os.getpid(), error
        )
        self.retiring = True
        while True:
            time.sleep(1.0)
