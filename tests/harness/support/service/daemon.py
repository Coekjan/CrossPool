"""Launch and control isolated daemon applications and installed processes."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI

import xpool.native
import xpool.service.daemon.app
import xpool.service.daemon.control
from tests.harness.support.config import TEST_MODEL_ID, install_test_config
from xpool.abi import ABI_VERSION
from xpool.config import XpoolConfig
from xpool.service.daemon.app import create_daemon
from xpool.service.daemon.mps import MpsProbeResult
from xpool.utils.procs import ProcUniqId


@dataclass(slots=True)
class FakeMonotonicClock:
    """Controllable monotonic clock shared by daemon route and state modules."""

    value: float = 1000.0

    def __call__(self) -> float:
        """Return the current synthetic monotonic timestamp."""

        return self.value

    def advance(self, seconds: float) -> None:
        """Advance synthetic monotonic time by a positive duration.

        Args:
            seconds: Positive duration in seconds.

        Raises:
            ValueError: If ``seconds`` is not positive.
        """

        if seconds <= 0:
            raise ValueError("fake monotonic clock advance must be positive")
        self.value += seconds


@pytest.fixture
def deterministic_daemon_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    reset_global_config: None,
) -> FakeMonotonicClock:
    """Replace daemon host dependencies with deterministic integration fakes."""

    clock = FakeMonotonicClock()
    monkeypatch.setattr(xpool.service.daemon.app.bootstrap, "init", lambda device, role: None)
    monkeypatch.setattr(xpool.service.daemon.app, "monotonic", clock)
    monkeypatch.setattr(xpool.service.daemon.control, "monotonic", clock)
    terminate_tree = ProcUniqId.terminate_tree
    kill_tree = ProcUniqId.kill_tree

    def terminate_non_test_process(process: ProcUniqId, *, term_grace_s: float) -> bool:
        if process.pid == os.getpid():
            return False
        return terminate_tree(process, term_grace_s=term_grace_s)

    monkeypatch.setattr(ProcUniqId, "terminate_tree", terminate_non_test_process)

    def kill_non_test_process(process: ProcUniqId) -> bool:
        if process.pid == os.getpid():
            return False
        return kill_tree(process)

    monkeypatch.setattr(ProcUniqId, "kill_tree", kill_non_test_process)
    monkeypatch.setattr(
        xpool.service.daemon.control,
        "probe_mps_controller",
        lambda: MpsProbeResult(True, "test MPS controller is online"),
    )
    monkeypatch.setattr(xpool.native.fabric, "create_uid", lambda: "ab" * 128)
    return clock


def create_app(config: XpoolConfig) -> FastAPI:
    """Create a daemon app with deterministic online MPS readiness."""

    install_test_config(config)
    app = create_daemon()
    app.state.control_plane.watchdog()
    return app


def run_watchdog(app: FastAPI) -> None:
    """Run one explicit daemon maintenance tick."""

    app.state.control_plane.watchdog()


def instance_registration(
    *,
    instance_id: str = TEST_MODEL_ID,
    rank: int = 0,
    abi_version: int = ABI_VERSION,
    max_tokens: int = 8,
    atn_tp_rank: int | None = None,
    atn_tp_size: int = 1,
    atn_dp_rank: int = 0,
    atn_dp_size: int = 1,
    pid: int | None = None,
) -> dict[str, object]:
    pid = process_pid(pid)
    return {
        "instance_id": instance_id,
        "rank": rank,
        "abi_version": abi_version,
        "pid": pid,
        "transport": transport_requirements(
            max_tokens=max_tokens,
            atn_tp_rank=rank if atn_tp_rank is None else atn_tp_rank,
            atn_tp_size=atn_tp_size,
            atn_dp_rank=atn_dp_rank,
            atn_dp_size=atn_dp_size,
        ),
        "workload": ffn_workload(),
    }


def atnagent_registration(
    *,
    cuda_device: int,
    pid: int | None = None,
    abi_version: int = ABI_VERSION,
) -> dict[str, int | float]:
    pid = process_pid(pid)
    return {
        "cuda_device": cuda_device,
        "abi_version": abi_version,
        "pid": pid,
    }


def ffnagent_registration(
    *,
    cuda_device: int = 1,
    pid: int | None = None,
    abi_version: int = ABI_VERSION,
) -> dict[str, int]:
    """Return one valid FfnAgent registration payload."""

    return {
        "pid": process_pid(pid),
        "abi_version": abi_version,
        "cuda_device": cuda_device,
    }


def ffn_workload(*, hidden_size: int = 2048) -> dict[str, object]:
    """Return one valid rank-independent FFN workload payload."""

    return {
        "model_config_digest": "a" * 64,
        "dtype": 1,
        "hidden_size": hidden_size,
        "layers": [{"layer_id": 0, "kind": 1}],
        "max_decode_rows": 4,
        "max_prefill_rows": 8,
    }


def atnagent_transport_arenas_path(cuda_device: int) -> str:
    return f"/atnagent/{cuda_device}/transport-arenas"


def atnagent_transport_leases_quiesce_path(cuda_device: int) -> str:
    return f"/atnagent/{cuda_device}/transport-leases/quiesce"


def instance_transport_arena_acquire_path(instance_id: str, rank: int) -> str:
    return f"/instance/{instance_id}/transport-arena/acquire?rank={rank}"


def process_ref(registration: Mapping[str, object] | None = None) -> dict[str, object]:
    if registration is None:
        return {"pid": process_pid(None), "abi_version": ABI_VERSION}
    return {"pid": registration["pid"], "abi_version": registration["abi_version"]}


def fabric_participant_report(
    registration: Mapping[str, object],
    *,
    generation: Mapping[str, object],
    pe: int,
    phase: str,
    plan_digest: str,
    invocation_failure: Mapping[str, object] | None = None,
    protocol_failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return one self-contained Agent Fabric participant report."""

    return {
        "owner": process_ref(registration),
        "generation": dict(generation),
        "pe": pe,
        "phase": phase,
        "plan_digest": plan_digest,
        "invocation_failure": None if invocation_failure is None else dict(invocation_failure),
        "protocol_failure": None if protocol_failure is None else dict(protocol_failure),
    }


def atnagent_transport_arenas(
    *arenas: tuple[str, int],
    publisher: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "publisher": process_ref(publisher),
        "bindings": atnagent_transport_arena_bindings(*arenas),
    }


def atnagent_transport_arena_bindings(*arenas: tuple[str, int]) -> list[dict[str, object]]:
    if not arenas:
        arenas = ((TEST_MODEL_ID, 0),)
    return [
        {"instance_id": instance_id, "rank": rank, "handle": instance_transport_arena(rank=rank)}
        for instance_id, rank in arenas
    ]


def transport_requirements(
    *,
    max_tokens: int = 8,
    atn_tp_rank: int = 0,
    atn_tp_size: int = 1,
    atn_dp_rank: int = 0,
    atn_dp_size: int = 1,
) -> dict[str, int]:
    return {
        "hidden_size": 4,
        "max_tokens": max_tokens,
        "atn_tp_rank": atn_tp_rank,
        "atn_tp_size": atn_tp_size,
        "atn_dp_rank": atn_dp_rank,
        "atn_dp_size": atn_dp_size,
    }


def instance_transport_arena(*, rank: int) -> dict[str, object]:
    return {
        "handle": f"{rank:02x}" * 64,
    }


def process_pid(pid: int | None) -> int:
    if pid is None:
        return ProcUniqId.current().pid
    return pid


def start_sleeping_proc() -> tuple[subprocess.Popen[bytes], ProcUniqId]:
    process = subprocess.Popen(
        ["sleep", "60"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, ProcUniqId(process.pid)


def stop_proc(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def request(app: FastAPI, method: str, path: str, *, json: object | None = None) -> httpx.Response:
    """Issue one HTTP request through the daemon's ASGI boundary."""

    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=json)

    return asyncio.run(run_request())
