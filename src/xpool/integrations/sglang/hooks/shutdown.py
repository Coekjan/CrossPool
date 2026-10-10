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
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.io_struct import ShutdownReq, async_sock_send, sock_send
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.tokenizer_manager import SignalHandler, TokenizerManager
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils.watchdog import SubprocessWatchdog

from xpool.config import get_global_config
from xpool.integrations.sglang import worker
from xpool.integrations.sglang.hooks.registry import SglangHook, SglangHookSet
from xpool.native import ABI_VERSION
from xpool.service.client import XpoolClient
from xpool.service.wire import MpsClientTermination
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MPS_TERMINATION_TIMEOUT_S, MpsEndpoint
from xpool.utils.procs import ProcUniqId

logger = logging.getLogger(__name__)


class ShutdownHookSet(SglangHookSet):
    """Retain created serving children and the first retirement budget.

    Cancellation unwinds synchronous launch before destructive retirement.
    Unpublished creation and unfinished controller startup retain the owner.
    A failed generation bypasses request drain, while every destructive exit
    still requires device-context safety and consumer-before-exporter ordering.
    """

    def __init__(self) -> None:
        self.deadline: float | None = None
        self.cancelled: int | None = None
        self.launching = False
        self.creating = False
        self.creation_unconfirmed = False
        self.failed = False
        self.retiring = False
        self.shutdown_requested = False
        self.tokenizer: TokenizerManager | MultiTokenizerRouter | None = None
        self.watchdog: SubprocessWatchdog | None = None
        self.schedulers: tuple[ProcUniqId, ...] = ()
        self.controllers: tuple[ProcUniqId, ...] = ()
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
                "sglang.srt.entrypoints.engine._set_envs_and_config",
                self.after_set_envs_and_config,
                HookType.AFTER,
            ),
            SglangHook(
                "sglang.srt.entrypoints.engine.Engine._launch_scheduler_processes",
                self.around_launch_scheduler_processes,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.weight_cache.daemon.spawn_weight_cache_daemon",
                self.around_spawn_weight_cache_daemon,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.data_parallel_controller.DataParallelController.launch_dp_attention_schedulers",
                self.around_launch_dp_attention_schedulers,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.tokenizer_manager.SignalHandler.sigterm_handler",
                self.around_sigterm_handler,
                HookType.AROUND,
            ),
            SglangHook(
                "sglang.srt.managers.tokenizer_manager.SignalHandler.running_phase_sigquit_handler",
                self.around_sigquit_handler,
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
        if signum == signal.SIGQUIT:
            self.failed = True
        if self.retiring or self.creating:
            return
        if self.launching:
            raise SystemExit(128 + self.cancelled)
        self.around_kill_process_tree(None, os.getpid(), include_parent=False)
        raise SystemExit(128 + self.cancelled)

    def after_set_envs_and_config(self, result: None, server_args: ServerArgs) -> None:
        # Upstream installs its launch-phase killer here. Keep cancellation at
        # this owner until the synchronous factory has actually unwound.
        signal.signal(signal.SIGQUIT, self.startup_signal)

    def around_launch_scheduler_processes(
        self,
        original_fn: Callable[..., tuple[engine.SchedulerInitResult, Sequence[BaseProcess]]],
        engine_type: type[engine.Engine],
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable[..., object],
        *,
        placement_group: object | None = None,
    ) -> tuple[engine.SchedulerInitResult, Sequence[BaseProcess]]:
        """Retain scheduler factory handles before waiting for any ready reply."""

        if self.cancelled is not None:
            raise SystemExit(128 + self.cancelled)
        self.creating = True
        try:
            result = original_fn(
                engine_type,
                server_args,
                port_args,
                run_scheduler_process_func,
                placement_group=placement_group,
            )
            processes = result[1]
            identities: list[ProcUniqId] = []
            for process in processes:
                if process.pid is None:
                    raise RuntimeError("scheduler factory returned an unstarted process")
                identities.append(ProcUniqId(process.pid))
            if get_parallel().dp_size > 1:
                self.controllers = tuple(identities)
            else:
                self.schedulers = tuple(identities)
        except BaseException:
            # The factory's local list is unavailable when it does not return;
            # a child can have started before its handle was appended.
            self.creation_unconfirmed = True
            raise
        finally:
            self.creating = False
        if self.cancelled is not None:
            raise SystemExit(128 + self.cancelled)
        return result

    def around_spawn_weight_cache_daemon(
        self,
        original_fn: Callable[..., BaseProcess],
        server_args: ServerArgs,
        *,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dist_init_method: str,
    ) -> BaseProcess:
        """Retain each actual cache handle before the factory's ready loop."""

        if self.cancelled is not None:
            raise SystemExit(128 + self.cancelled)
        self.creating = True
        try:
            process = original_fn(
                server_args, gpu_id=gpu_id, tp_rank=tp_rank, pp_rank=pp_rank, dist_init_method=dist_init_method
            )
            self.caches = (*self.caches, process)
        except BaseException:
            self.creation_unconfirmed = True
            raise
        finally:
            self.creating = False
        if self.cancelled is not None:
            raise SystemExit(128 + self.cancelled)
        return process

    def around_launch_dp_attention_schedulers[**P, R](
        self,
        original_fn: Callable[Concatenate[DataParallelController, P], R],
        controller: DataParallelController,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Seal synchronous DP creation before its entry forwards a failure.

        The existing controller list retains published handles on both return
        and unwind. An incomplete list cannot rule out an unpublished spawn;
        such interruption retains this controller rather than forwarding a
        presumed creation seal to its parent.
        """

        previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)}

        def cancel_launch(signum: int, frame: FrameType | None) -> None:
            if self.cancelled is None:
                self.cancelled = signum
            self.begin_retirement()
            if not self.retiring:
                raise SystemExit(128 + self.cancelled)

        for signum in previous:
            signal.signal(signum, cancel_launch)
        try:
            return original_fn(controller, *args, **kwargs)
        finally:
            processes = controller.scheduler_procs
            self.schedulers = tuple(ProcUniqId(process.pid) for process in processes if process.pid is not None)
            if len(self.schedulers) != get_global_config().atn_world_size:
                self.retain_owner(RuntimeError("controller scheduler creation is incomplete or unpublished"))
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def around_launch_subprocesses[R](
        self,
        original_fn: Callable[..., R],
        engine_type: type[engine.Engine],
        server_args: ServerArgs,
        init_tokenizer_manager_func: Callable[..., object],
        run_scheduler_process_func: Callable[..., object],
        run_detokenizer_process_func: Callable[..., object],
        port_args: PortArgs | None = None,
        placement_group: object | None = None,
    ) -> R:
        self.launching = True
        # The engine parent handles cancellation independently of its launch
        # group. Workers retain their engine-installed signal handlers.
        signal.signal(signal.SIGINT, self.startup_signal)
        signal.signal(signal.SIGTERM, self.startup_signal)
        # The spawn alias requires the importable entry, not a hook closure.
        # Its pinned signature matches; upstream leaves its pipe unannotated.
        engine.run_data_parallel_controller_process = worker.run_data_parallel_controller_process  # ty: ignore[invalid-assignment]
        try:
            result = original_fn(
                engine_type,
                server_args=server_args,
                init_tokenizer_manager_func=init_tokenizer_manager_func,
                run_scheduler_process_func=worker.run_scheduler_process,
                run_detokenizer_process_func=run_detokenizer_process_func,
                port_args=port_args,
                placement_group=placement_group,
            )
        except BaseException as error:
            self.launching = False
            self.failed = True
            logger.error("serving launch failed pid=%s", os.getpid(), exc_info=error)
            self.around_kill_process_tree(None, os.getpid(), include_parent=False)
            raise
        self.launching = False
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
        if ready[5] is not None:
            self.caches = ready[5]
        # DP's existing ready reply supplies its actual workers. Ordinary
        # schedulers were retained at their factory before this handshake.
        dp_workers = [pid for info in schedulers.scheduler_infos for pid in info.get(engine.SCHEDULER_PIDS_ARG, ())]
        if dp_workers:
            try:
                self.schedulers = tuple(ProcUniqId(pid) for pid in dp_workers)
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
        if self.launching:
            self.startup_signal(signal.SIGTERM if signum is None else signum, frame)
            return
        self.begin_retirement()
        original_fn(handler, signum, frame)

    def around_sigquit_handler(
        self,
        original_fn: Callable[[SignalHandler, int | None, FrameType | None], None],
        handler: SignalHandler,
        signum: int | None = None,
        frame: FrameType | None = None,
    ) -> None:
        self.failed = True
        self.begin_retirement()
        if self.launching:
            self.startup_signal(signal.SIGQUIT, frame)
            return
        original_fn(handler, signum, frame)

    def request_scheduler_shutdown(self, deadline: float) -> None:
        """Request cooperative exit once; publication supplies no retirement proof."""

        if self.watchdog is not None:
            self.watchdog.stop()
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        try:
            match self.tokenizer:
                case TokenizerManager():
                    # Factory-ready cancellation precedes the ASGI event loop.
                    # Reuse its socket and serializer without another channel.
                    # get_zmq_socket also annotates a return_bind_port tuple;
                    # this manager constructs its socket without that option.
                    sock_send(
                        zmq.Socket.shadow(cast(zmq.Socket, self.tokenizer.send_to_scheduler)),
                        ShutdownReq(),
                        flags=zmq.DONTWAIT,
                    )
                case MultiTokenizerRouter():
                    # The router owns an asyncio socket on its existing loop.
                    # The pinned get_zmq_socket annotation omits that variant.
                    asyncio.run_coroutine_threadsafe(
                        async_sock_send(cast(zmq.asyncio.Socket, self.tokenizer.send_to_scheduler), ShutdownReq()),
                        self.tokenizer._loop,
                    ).result(timeout=max(0.0, deadline - time.monotonic()))
                case None:
                    # The Rust factory has no Python control socket. Clients
                    # still pass through the confirmed termination boundary.
                    return
        except (zmq.Again, TimeoutError):
            logger.warning("scheduler shutdown notification unavailable pid=%s; continuing safety checks", os.getpid())

    def around_dispatch_to_scheduler(
        self,
        original_fn: Callable[[TokenizerManager, object], None],
        tokenizer: TokenizerManager,
        message: object,
    ) -> None:
        # The running watchdog already sends ShutdownReq. Remember that actual
        # send so its later killer cannot block on a socket whose peers exited.
        if isinstance(message, ShutdownReq):
            self.begin_retirement()
            self.shutdown_requested = True
        try:
            original_fn(tokenizer, message)
        except zmq.Again:
            if not isinstance(message, ShutdownReq):
                raise
            logger.warning("scheduler shutdown notification unavailable pid=%s; continuing safety checks", os.getpid())

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
            if self.launching or self.creating or self.creation_unconfirmed:
                raise RuntimeError("serving child creation has not been sealed and confirmed")
            if self.controllers and not self.schedulers:
                raise RuntimeError("controller has not published its complete ready worker world")
            cooperative_deadline = deadline - MPS_TERMINATION_TIMEOUT_S
            if parent_pid == os.getpid():
                self.request_scheduler_shutdown(time.monotonic() if self.failed else cooperative_deadline)
            targets = [target for target in parent.child_process_ids() if target.pid != skip_pid]
            affected_workers = tuple(worker for worker in self.schedulers if worker in targets)
            while (
                not self.failed
                and any(worker.is_alive() for worker in affected_workers)
                and time.monotonic() < cooperative_deadline
            ):
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
                    if target in self.schedulers or target.pid in cache_pids:
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
        if self.launching:
            # The cache factory compensates before the enclosing launch has
            # unwound. Preserve its handles; the outer owner retires the domain.
            self.caches = procs
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
