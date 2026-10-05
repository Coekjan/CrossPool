from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from multiprocessing import Process
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import zmq
import zmq.asyncio
from sglang.srt.entrypoints import engine
from sglang.srt.managers.io_struct import ShutdownReq, sock_recv
from sglang.srt.managers.tokenizer_manager import SignalHandler, TokenizerManager
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import PortArgs
from sglang.srt.utils import common
from sglang.srt.utils.watchdog import SubprocessWatchdog

from xpool.integrations.sglang.hooks import shutdown
from xpool.service.client import XpoolClient
from xpool.service.wire import MpsClientTermination
from xpool.utils.mps import MpsEndpoint
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import reset_global_config
from xtest.harness.support.sglang.plugin import reset_plugin_required_hook_targets

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


def test_startup_cancellation_waits_for_factory_ready_without_renewing_budget(
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
) -> None:
    root, children = serving_tree
    owner = shutdown.ShutdownHookSet()
    completed = False
    initial_deadline: float | None = None
    tokenizer = TokenizerManager.__new__(TokenizerManager)
    tokenizer.gracefully_exit = False
    handler = SignalHandler(tokenizer)
    result = (
        None,
        None,
        PortArgs.__new__(PortArgs),
        engine.SchedulerInitResult(scheduler_infos=[], all_child_pids=[child.pid for child in children]),
        SubprocessWatchdog(
            processes=[cast(Process, SimpleNamespace(pid=child.pid)) for child in children],
            process_names=["scheduler_0", "detokenizer_0"],
        ),
        None,
    )

    def launch(engine_type: type[engine.Engine]) -> tuple[object, ...]:
        nonlocal completed, initial_deadline
        signal.raise_signal(signal.SIGTERM)
        handler.sigterm_handler(signal.SIGTERM, None)
        initial_deadline = owner.deadline
        assert owner.schedulers is None
        assert not tokenizer.gracefully_exit
        owner.startup_signal(signal.SIGINT, None)
        assert owner.deadline == initial_deadline
        assert all(child.is_alive() for child in children)
        completed = True
        return result

    def retire(original_fn: object, parent_pid: int, *, include_parent: bool) -> None:
        assert completed
        assert owner.schedulers == children[:1]
        assert owner.deadline == initial_deadline
        assert not include_parent

    # Cancellation also works for a non-leading parent. The plugin fixture
    # restores the session's signal handlers after the factory check.
    monkeypatch.setattr(shutdown.os, "getpgrp", lambda: os.getpid() + 1)
    monkeypatch.setattr(owner, "around_kill_process_tree", retire)
    HookRegistry.register(
        "sglang.srt.entrypoints.engine.Engine._launch_subprocesses", owner.around_launch_subprocesses, HookType.AROUND
    )
    HookRegistry.register(
        "sglang.srt.managers.tokenizer_manager.SignalHandler.sigterm_handler",
        owner.around_sigterm_handler,
        HookType.AROUND,
    )
    monkeypatch.setattr(engine.Engine, "_launch_subprocesses", classmethod(launch))
    HookRegistry.apply_hooks()
    with pytest.raises(SystemExit) as cancelled:
        engine.Engine._launch_subprocesses()  # ty: ignore[missing-argument]
    assert cancelled.value.code == 128 + signal.SIGTERM
    assert root.poll() is None


def test_running_signal_keeps_upstream_drain_and_first_cleanup_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = shutdown.ShutdownHookSet()
    tokenizer = TokenizerManager.__new__(TokenizerManager)
    tokenizer.gracefully_exit = False
    handler = SignalHandler(tokenizer)
    HookRegistry.register(
        "sglang.srt.managers.tokenizer_manager.SignalHandler.sigterm_handler",
        owner.around_sigterm_handler,
        HookType.AROUND,
    )
    HookRegistry.apply_hooks()
    handler.sigterm_handler()
    initial_deadline = owner.deadline
    assert tokenizer.gracefully_exit
    assert initial_deadline is not None
    handler.sigterm_handler(signal.SIGTERM, None)
    assert owner.deadline == initial_deadline


def test_ready_tokenizer_shutdown_uses_its_socket_before_event_loop() -> None:
    owner = shutdown.ShutdownHookSet()
    tokenizer = TokenizerManager.__new__(TokenizerManager)
    with zmq.asyncio.Context() as context, context.socket(zmq.PUSH) as sender, zmq.Context() as receiver_context:
        with receiver_context.socket(zmq.PULL) as receiver:
            port = sender.bind_to_random_port("tcp://127.0.0.1")
            receiver.connect(f"tcp://127.0.0.1:{port}")
            tokenizer.send_to_scheduler = sender
            owner.tokenizer = tokenizer
            assert zmq.Socket.shadow(sender).poll(1000, zmq.POLLOUT)
            owner.request_scheduler_shutdown(time.monotonic() + 5)
            assert receiver.poll(1000)
            assert isinstance(sock_recv(receiver), ShutdownReq)
            owner.request_scheduler_shutdown(time.monotonic() + 5)
            assert not receiver.poll(0)


@pytest.mark.parametrize("confirmed", [True, False])
def test_destructive_exit_confirms_all_affected_clients_before_any_host_signal(
    confirmed: bool,
    monkeypatch: pytest.MonkeyPatch,
    serving_tree: tuple[subprocess.Popen[str], tuple[ProcUniqId, ...]],
) -> None:
    root, children = serving_tree
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000001")
    owner = shutdown.ShutdownHookSet()
    owner.deadline = time.monotonic() + 5
    owner.schedulers = children
    # Reserve the whole short test budget for controlled termination instead
    # of waiting the production cooperative interval.
    monkeypatch.setattr(shutdown, "MPS_TERMINATION_TIMEOUT_S", 5.0)
    terminated: list[int] = []

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
    HookRegistry.register("sglang.srt.utils.common.kill_process_tree", owner.around_kill_process_tree, HookType.AROUND)
    HookRegistry.apply_hooks()
    if confirmed:
        common.kill_process_tree(root.pid, include_parent=False, wait_timeout=3)
        assert not any(child.is_alive() for child in children)
    else:
        with pytest.raises(RuntimeError, match="retention boundary reached"):
            common.kill_process_tree(root.pid, include_parent=False, wait_timeout=3)
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
