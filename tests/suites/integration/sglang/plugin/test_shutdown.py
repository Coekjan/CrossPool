from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from collections.abc import Coroutine, Iterator, Sequence
from concurrent.futures import Future
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import zmq
import zmq.asyncio
from sglang.srt.entrypoints import engine
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.io_struct import ShutdownReq, sock_recv
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.tokenizer_manager import SignalHandler, TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs

from xpool.integrations.sglang import worker
from xpool.integrations.sglang.hooks import shutdown
from xpool.service.client import XpoolClient
from xpool.service.wire import MpsClientTermination
from xpool.utils.mps import MpsEndpoint
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import reset_global_config, synthetic_config
from xtest.harness.support.sglang import fakes
from xtest.harness.support.sglang.plugin import reset_plugin_required_hook_targets
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, reset_plugin_required_hook_targets.__name__)


@pytest.fixture
def serving_tree(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]]]:
    # Every process in this stand-in is CPU-only; emergency fixture cleanup
    # does not claim to retire device resources.
    root = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; "
            "children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']) for n in range(2)]; "
            "print(*(child.pid for child in children), flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert root.stdout is not None
    children = tuple(ProcUniqId(int(pid)) for pid in root.stdout.readline().split())
    try:
        yield root, children
    finally:
        for child in children:
            child.send_signal(signal.SIGKILL)
        if root.poll() is None:
            root.kill()
        root.wait(timeout=5)
        root.stdout.close()


@pytest.mark.usefixtures(published_sglang_config.__name__)
@pytest.mark.parametrize("child_kind", ["scheduler", "cache"])
def test_startup_cancellation_retires_owned_children_after_factory_unwinds(
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
    child_kind: str,
) -> None:
    root, children = serving_tree
    owner = shutdown.ShutdownHookSet()
    unwound = False
    initial_deadline: float | None = None
    server_args = fakes.server_args()
    port_args = PortArgs.__new__(PortArgs)
    process = cast(BaseProcess, SimpleNamespace(pid=children[0].pid))

    def create(*args: object, **kwargs: object) -> BaseProcess:
        nonlocal initial_deadline
        if child_kind == "scheduler":
            owner.startup_signal(signal.SIGTERM, None)
            initial_deadline = owner.deadline
            owner.startup_signal(signal.SIGINT, None)
            assert owner.deadline == initial_deadline
        assert not owner.retiring
        assert all(child.is_alive() for child in children)
        return process

    def create_schedulers(
        *args: object, **kwargs: object
    ) -> tuple[engine.SchedulerInitResult, tuple[BaseProcess, ...]]:
        return engine.SchedulerInitResult(scheduler_infos=[]), (create(*args, **kwargs),)

    def launch(engine_type: type[engine.Engine], **kwargs: object) -> None:
        nonlocal unwound, initial_deadline
        try:
            if child_kind == "scheduler":
                owner.around_launch_scheduler_processes(
                    create_schedulers, engine_type, server_args, port_args, lambda *args: None
                )
            else:
                cache = owner.around_spawn_weight_cache_daemon(
                    create, server_args, gpu_id=0, tp_rank=0, pp_rank=0, dist_init_method="unused"
                )
                try:
                    # Cache handles are already owned when its existing
                    # readiness wait receives cancellation.
                    owner.startup_signal(signal.SIGTERM, None)
                finally:
                    initial_deadline = owner.deadline
                    # Existing cache-factory compensation runs before the
                    # enclosing launch unwinds; it cannot retire consumers yet.
                    owner.around_terminate_weight_cache_daemons(lambda procs, timeout: pytest.fail(), (cache,))
                    assert all(child.is_alive() for child in children)
                    owner.startup_signal(signal.SIGINT, None)
            pytest.fail("cancelled creation must unwind before waiting for readiness")
        finally:
            unwound = True

    def retire(original_fn: object, parent_pid: int, *, include_parent: bool) -> None:
        assert unwound
        if child_kind == "scheduler":
            assert owner.schedulers == children[:1]
        else:
            assert tuple(cache.pid for cache in owner.caches) == (children[0].pid,)
        assert owner.deadline == initial_deadline
        assert not include_parent

    monkeypatch.setattr(owner, "around_kill_process_tree", retire)
    with pytest.raises(SystemExit) as cancelled:
        owner.around_launch_subprocesses(
            launch,
            engine.Engine,
            server_args=server_args,
            init_tokenizer_manager_func=lambda *args: None,
            run_scheduler_process_func=lambda *args: None,
            run_detokenizer_process_func=lambda *args: None,
        )
    assert cancelled.value.code == 128 + signal.SIGTERM
    assert root.poll() is None


def test_unpublished_startup_creation_retains_owner_and_original_failure(
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
) -> None:
    root, children = serving_tree
    owner = shutdown.ShutdownHookSet()
    original = RuntimeError("creation interrupted before publishing child handle")

    class Retained(Exception):
        pass

    def create(*args: object, **kwargs: object) -> tuple[engine.SchedulerInitResult, tuple[BaseProcess, ...]]:
        raise original

    def launch(engine_type: type[engine.Engine], **kwargs: object) -> None:
        owner.around_launch_scheduler_processes(
            create, engine_type, fakes.server_args(), PortArgs.__new__(PortArgs), lambda *args: None
        )

    def retain(error: BaseException) -> None:
        assert all(child.is_alive() for child in children)
        raise Retained from error

    monkeypatch.setattr(owner, "retain_owner", retain)
    monkeypatch.setattr(MpsEndpoint, "run_control", lambda *args, **kwargs: pytest.fail("unsealed creation"))
    with pytest.raises(Retained) as retained:
        owner.around_launch_subprocesses(
            launch,
            engine.Engine,
            server_args=fakes.server_args(),
            init_tokenizer_manager_func=lambda *args: None,
            run_scheduler_process_func=lambda *args: None,
            run_detokenizer_process_func=lambda *args: None,
        )
    assert retained.value.__cause__ is not None
    assert retained.value.__cause__.__context__ is original
    assert root.poll() is None


def test_dp_controller_unwinds_creation_before_forwarding_and_remains_owned(
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
) -> None:
    root, children = serving_tree
    owner = shutdown.ShutdownHookSet()
    controller = DataParallelController.__new__(DataParallelController)
    controller.scheduler_procs = [cast(BaseProcess, SimpleNamespace(pid=child.pid)) for child in children]
    config = synthetic_config(atn_devices=(0, 1), ffn_devices=(2,))
    monkeypatch.setattr(shutdown, "get_global_config", lambda: config)
    monkeypatch.setattr(worker, "load_plugins", lambda: None)
    unwound = False
    notified: list[signal.Signals] = []

    class Retained(Exception):
        pass

    def create(controller: DataParallelController, server_args: ServerArgs, port_args: PortArgs) -> None:
        nonlocal unwound
        try:
            signal.raise_signal(signal.SIGTERM)
        finally:
            unwound = True

    def launch(
        server_args: ServerArgs, port_args: PortArgs, pipe_writer: object, run_scheduler_process_func: object
    ) -> None:
        owner.around_launch_dp_attention_schedulers(create, controller, server_args, port_args)

    def notify(signum: signal.Signals) -> None:
        assert unwound
        assert owner.schedulers == children
        assert all(child.is_alive() for child in children)
        notified.append(signum)

    def wait(seconds: float) -> None:
        # After startup has unwound, repeated signals reach the serving root
        # through the restored forwarding handler rather than exiting DP.
        signal.raise_signal(signal.SIGTERM)
        raise Retained

    monkeypatch.setattr(worker.data_parallel_controller, "run_data_parallel_controller_process", launch)
    monkeypatch.setattr(worker, "ProcUniqId", lambda pid: SimpleNamespace(pid=root.pid, send_signal=notify))
    monkeypatch.setattr(worker, "time", SimpleNamespace(sleep=wait))
    reader, writer = multiprocessing.Pipe(duplex=False)
    with reader, writer, pytest.raises(Retained):
        worker.run_data_parallel_controller_process(fakes.server_args(), PortArgs.__new__(PortArgs), writer)
    assert notified == [signal.SIGQUIT, signal.SIGTERM]
    assert root.poll() is None


def test_running_signal_keeps_upstream_drain_and_first_cleanup_deadline() -> None:
    owner = shutdown.ShutdownHookSet()
    tokenizer = TokenizerManager.__new__(TokenizerManager)
    tokenizer.gracefully_exit = False
    handler = SignalHandler(tokenizer)
    owner.around_sigterm_handler(SignalHandler.sigterm_handler, handler)
    initial_deadline = owner.deadline
    assert tokenizer.gracefully_exit
    assert initial_deadline is not None
    owner.around_sigterm_handler(SignalHandler.sigterm_handler, handler, signal.SIGTERM, None)
    assert owner.deadline == initial_deadline


@pytest.mark.parametrize("connected", [True, False])
def test_ready_tokenizer_shutdown_uses_its_socket_before_event_loop(connected: bool) -> None:
    owner = shutdown.ShutdownHookSet()
    tokenizer = TokenizerManager.__new__(TokenizerManager)
    with zmq.asyncio.Context() as context, context.socket(zmq.PUSH) as sender, zmq.Context() as receiver_context:
        with receiver_context.socket(zmq.PULL) as receiver:
            port = sender.bind_to_random_port("tcp://127.0.0.1")
            sender.setsockopt(zmq.IMMEDIATE, 1)
            if connected:
                receiver.connect(f"tcp://127.0.0.1:{port}")
            tokenizer.send_to_scheduler = sender
            owner.tokenizer = tokenizer
            if connected:
                assert zmq.Socket.shadow(sender).poll(1000, zmq.POLLOUT)
            owner.request_scheduler_shutdown(time.monotonic() + 5)
            if connected:
                assert receiver.poll(1000)
                assert isinstance(sock_recv(receiver), ShutdownReq)
            owner.request_scheduler_shutdown(time.monotonic() + 5)
            assert not receiver.poll(0)


@pytest.mark.parametrize("failed", [False, True])
def test_router_notification_timeout_preserves_independent_retirement_checks(
    failed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = shutdown.ShutdownHookSet()
    owner.deadline = 1000.0
    owner.failed = failed
    waits: list[float | None] = []
    checked: list[tuple[Sequence[ProcUniqId], float]] = []

    class Notification(Future[None]):
        def result(self, timeout: float | None = None) -> None:
            waits.append(timeout)
            return super().result(timeout)

    def notify(coroutine: Coroutine[object, object, None], loop: asyncio.AbstractEventLoop) -> Future[None]:
        coroutine.close()
        future = Notification()
        future.set_exception(TimeoutError("router loop is unavailable"))
        return future

    def confirm(targets: Sequence[ProcUniqId], deadline: float) -> None:
        checked.append((targets, deadline))

    monkeypatch.setattr(shutdown.asyncio, "run_coroutine_threadsafe", notify)
    monkeypatch.setattr(shutdown, "time", SimpleNamespace(monotonic=lambda: 750.0))
    monkeypatch.setattr(ProcUniqId, "child_process_ids", lambda self: ())
    monkeypatch.setattr(owner, "confirm_context_termination", confirm)
    loop = asyncio.new_event_loop()
    try:
        with zmq.asyncio.Context() as context, context.socket(zmq.PUSH) as sender:
            tokenizer = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
            tokenizer.send_to_scheduler = sender
            tokenizer._loop = loop
            owner.tokenizer = tokenizer
            owner.around_kill_process_tree(None, os.getpid(), include_parent=False)
    finally:
        loop.close()

    assert waits == [0.0 if failed else 250.0 - shutdown.MPS_TERMINATION_TIMEOUT_S]
    assert checked == [([ProcUniqId(os.getpid())], 1000.0)]


@pytest.mark.parametrize("confirmed", [True, False])
def test_destructive_exit_confirms_all_affected_clients_before_any_host_signal(
    confirmed: bool,
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
) -> None:
    root, children = serving_tree
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000001")
    owner = shutdown.ShutdownHookSet()
    owner.deadline = time.monotonic() + shutdown.MPS_CLEANUP_TIMEOUT_S
    owner.failed = True
    owner.schedulers = children
    terminated: list[int] = []

    def wait(seconds: float) -> None:
        assert terminated, "failed-generation retirement must not consume ordinary cooperative drain"
        time.sleep(seconds)

    monkeypatch.setattr(shutdown, "time", SimpleNamespace(monotonic=time.monotonic, sleep=wait))

    def management(self: MpsEndpoint, command: str, *, deadline: float) -> str:
        return "42" if command == "get_server_list" else "\n".join(str(child.pid) for child in children)

    class Client:
        def terminate_serving_client(self, request: MpsClientTermination) -> None:
            assert request.deadline == owner.deadline
            assert all(child.is_alive() for child in children)
            terminated.append(request.pid)
            if not confirmed and len(terminated) == 2:
                raise RuntimeError("context termination unconfirmed")

        def close(self) -> None:
            pass

    retention_errors: list[BaseException] = []

    def retain(error: BaseException) -> None:
        assert all(child.is_alive() for child in children)
        retention_errors.append(error)
        raise RuntimeError("retention boundary reached") from error

    monkeypatch.setattr(MpsEndpoint, "run_control", management)
    monkeypatch.setattr(shutdown, "XpoolClient", lambda **kwargs: cast(XpoolClient, Client()))
    monkeypatch.setattr(owner, "retain_owner", retain)
    if confirmed:
        owner.around_kill_process_tree(None, root.pid, include_parent=False, wait_timeout=3)
        assert not any(child.is_alive() for child in children)
    else:
        with pytest.raises(RuntimeError, match="retention boundary reached"):
            owner.around_kill_process_tree(None, root.pid, include_parent=False, wait_timeout=3)
        assert len(retention_errors) == 1
        assert str(retention_errors[0]) == "context termination unconfirmed"
    assert terminated == [child.pid for child in children]
    assert root.poll() is None


def test_cache_exporter_context_is_terminated_after_its_consumers(
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
) -> None:
    root, children = serving_tree
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000001")
    consumer, exporter = children
    owner = shutdown.ShutdownHookSet()
    owner.deadline = time.monotonic() + 5
    owner.caches = (cast(BaseProcess, SimpleNamespace(pid=exporter.pid)),)
    terminated: list[int] = []

    class Client:
        def terminate_serving_client(self, request: MpsClientTermination) -> None:
            assert all(child.is_alive() for child in children)
            terminated.append(request.pid)

        def close(self) -> None:
            pass

    monkeypatch.setattr(shutdown, "XpoolClient", lambda **kwargs: cast(XpoolClient, Client()))
    monkeypatch.setattr(
        MpsEndpoint,
        "run_control",
        lambda self, command, **kwargs: "42" if command == "get_server_list" else f"{exporter.pid}\n{consumer.pid}",
    )
    owner.around_kill_process_tree(None, root.pid, include_parent=False)
    assert terminated == [consumer.pid, exporter.pid]
    assert not any(child.is_alive() for child in children)
    assert root.poll() is None
