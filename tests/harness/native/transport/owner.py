"""Typed subprocess harness for native Transport arena ownership."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import xpool.native
from tests.harness.native.transport.protocol import (
    TRANSPORT_COMMAND_TIMEOUT_SECONDS,
    TransportArenaPublished,
    TransportOwnerCommand,
    TransportOwnerSpec,
    TransportOwnerState,
    TransportOwnerStatus,
)
from tests.harness.native.transport.trace import write_transport_trace
from tests.harness.runner.child import PythonChildProcess
from xpool.abi import TensorDType
from xpool.cext import ensure_native_loaded
from xpool.runtime import RuntimeRole


@dataclass(slots=True)
class AtnAgentArenaController:
    """Control one subprocess-owned native Transport arena."""

    handle: str
    process: PythonChildProcess
    timeout_seconds: float = TRANSPORT_COMMAND_TIMEOUT_SECONDS
    activated: bool = True
    drained: bool = False
    destroyed: bool = False

    def command(self, command: TransportOwnerCommand) -> TransportOwnerStatus:
        """Send one typed lifecycle command and receive its status."""

        self.process.send(command)
        return self.process.receive(TransportOwnerStatus, timeout_seconds=self.timeout_seconds)

    def activate(self) -> None:
        """Launch the process-wide Resident after arena publication."""

        if self.activated:
            return
        response = self.command(TransportOwnerCommand.ACTIVATE)
        if response.state is not TransportOwnerState.ACTIVATED:
            raise RuntimeError(f"unexpected Transport activation state: {response.state}")
        self.activated = True

    def health(self) -> TransportOwnerStatus:
        """Return the child Resident health status."""

        return self.command(TransportOwnerCommand.HEALTH)

    def drain(self) -> None:
        """Drain the Resident while retaining the owner allocation."""

        if self.drained:
            return
        if not self.activated:
            self.drained = True
            return
        response = self.command(TransportOwnerCommand.DRAIN)
        if response.state is not TransportOwnerState.DRAINED:
            raise RuntimeError(f"unexpected Transport drain state: {response.state}")
        self.drained = True

    def destroy(self) -> None:
        """Destroy the owner allocation and reap the child."""

        if self.destroyed:
            return
        response = self.command(TransportOwnerCommand.DESTROY)
        if response.state is not TransportOwnerState.DESTROYED:
            raise RuntimeError(f"unexpected Transport destroy state: {response.state}")
        self.process.wait(timeout_seconds=self.timeout_seconds)
        self.drained = True
        self.destroyed = True


def drain_transport_resident() -> None:
    """Drain the process-wide Transport Resident with a bounded deadline."""

    xpool.native.transport.drain_async()
    deadline = time.monotonic() + TRANSPORT_COMMAND_TIMEOUT_SECONDS
    while xpool.native.transport.drain_pending():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out draining native Transport Resident")
        time.sleep(0.001)


def run_transport_owner(connection: Connection, spec: TransportOwnerSpec) -> None:
    """Own and serve one native Transport arena until terminal destroy."""

    ensure_native_loaded()
    debug_options = json.dumps(
        {
            "loopback": {
                "enable": spec.atnagent_loopback_enabled,
                "site": "atnagent" if spec.atnagent_loopback_enabled else None,
            },
            "transport_observer": {
                "enable": spec.observer_output_path is not None,
                "trace_capacity": 8192,
            },
            "fabric_observer": {"enable": False, "trace_capacity": 8192},
        }
    )
    xpool.native.initialize(int(RuntimeRole.ATNAGENT), spec.cuda_device, debug_options)
    arena = xpool.native.transport.create_arena(
        spec.instance_index,
        spec.instance_rank,
        spec.max_tokens,
        spec.hidden_size,
        int(spec.dtype),
        spec.atn_tp_rank,
        spec.atn_tp_size,
        spec.atn_dp_rank,
        spec.atn_dp_size,
    )
    activated = spec.activate_resident
    if activated:
        xpool.native.transport.activate()
    connection.send(TransportArenaPublished(str(arena)))
    drained = False
    while True:
        command = connection.recv()
        if not isinstance(command, TransportOwnerCommand):
            raise RuntimeError(f"invalid Transport owner command: {command!r}")
        match command:
            case TransportOwnerCommand.ACTIVATE:
                if not activated:
                    xpool.native.transport.activate()
                    activated = True
                connection.send(TransportOwnerStatus(TransportOwnerState.ACTIVATED))
            case TransportOwnerCommand.HEALTH:
                try:
                    xpool.native.transport.check_health()
                except RuntimeError as error:
                    connection.send(TransportOwnerStatus(TransportOwnerState.FAILED, str(error)))
                else:
                    connection.send(TransportOwnerStatus(TransportOwnerState.HEALTHY))
            case TransportOwnerCommand.DRAIN:
                if not drained:
                    if activated:
                        drain_transport_resident()
                    drained = True
                connection.send(TransportOwnerStatus(TransportOwnerState.DRAINED))
            case TransportOwnerCommand.DESTROY:
                if activated and not drained:
                    drain_transport_resident()
                if spec.observer_output_path is not None:
                    write_transport_trace(arena, spec.observer_output_path)
                xpool.native.transport.destroy_arenas([arena])
                connection.send(TransportOwnerStatus(TransportOwnerState.DESTROYED))
                return


@contextmanager
def controlled_atnagent_arena_process(
    *,
    workdir: Path,
    cuda_device: int,
    max_tokens: int,
    hidden_size: int,
    dtype: TensorDType,
    atn_dp_size: int,
    activate_resident: bool,
    atnagent_loopback_enabled: bool = True,
    observer_output_path: str = "",
    startup_timeout_s: float = TRANSPORT_COMMAND_TIMEOUT_SECONDS,
    instance_index: int = 0,
    instance_rank: int = 0,
    atn_tp_rank: int = 0,
    atn_tp_size: int = 1,
    atn_dp_rank: int = 0,
) -> Generator[AtnAgentArenaController, None, None]:
    """Start a controllable AtnAgent child and publish its arena."""

    spec = TransportOwnerSpec(
        cuda_device=cuda_device,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        atn_dp_size=atn_dp_size,
        activate_resident=activate_resident,
        atnagent_loopback_enabled=atnagent_loopback_enabled,
        observer_output_path=Path(observer_output_path) if observer_output_path else None,
        instance_index=instance_index,
        instance_rank=instance_rank,
        atn_tp_rank=atn_tp_rank,
        atn_tp_size=atn_tp_size,
        atn_dp_rank=atn_dp_rank,
    )
    workdir.mkdir(parents=True, exist_ok=False)
    process = PythonChildProcess.start(
        "atnagent-transport-owner",
        run_transport_owner,
        spec,
        log_path=workdir / "owner.log",
    )
    controller: AtnAgentArenaController | None = None
    try:
        published = process.receive(TransportArenaPublished, timeout_seconds=startup_timeout_s)
        controller = AtnAgentArenaController(
            handle=published.handle,
            process=process,
            timeout_seconds=startup_timeout_s,
            activated=activate_resident,
        )
        yield controller
    finally:
        try:
            if controller is not None and process.process.is_alive():
                controller.destroy()
        finally:
            if process.process.is_alive():
                PythonChildProcess.terminate_all((process,))
            process.close()


@contextmanager
def atnagent_arena_process(
    *,
    workdir: Path,
    cuda_device: int,
    max_tokens: int,
    hidden_size: int,
    dtype: TensorDType,
    atn_dp_size: int,
    activate_resident: bool,
    atnagent_loopback_enabled: bool = True,
    observer_output_path: str = "",
    startup_timeout_s: float = TRANSPORT_COMMAND_TIMEOUT_SECONDS,
    instance_index: int = 0,
    instance_rank: int = 0,
    atn_tp_rank: int = 0,
    atn_tp_size: int = 1,
    atn_dp_rank: int = 0,
) -> Generator[str, None, None]:
    """Yield a handle owned by one spawned AtnAgent process."""

    with controlled_atnagent_arena_process(
        workdir=workdir,
        cuda_device=cuda_device,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        atn_dp_size=atn_dp_size,
        activate_resident=activate_resident,
        atnagent_loopback_enabled=atnagent_loopback_enabled,
        observer_output_path=observer_output_path,
        startup_timeout_s=startup_timeout_s,
        instance_index=instance_index,
        instance_rank=instance_rank,
        atn_tp_rank=atn_tp_rank,
        atn_tp_size=atn_tp_size,
        atn_dp_rank=atn_dp_rank,
    ) as controller:
        yield controller.handle


@contextmanager
def transport_arena_handles(*, workdir: Path) -> Generator[Callable[..., str], None, None]:
    """Yield a factory and retain every spawned arena owner until scope exit."""

    with ExitStack() as stack:
        next_owner = 0

        def create(
            *,
            max_tokens: int = 8,
            hidden_size: int = 4,
            dtype: TensorDType = TensorDType.FP32,
            atn_dp_size: int = 1,
            activate_resident: bool = True,
            instance_index: int = 0,
            instance_rank: int = 0,
        ) -> str:
            nonlocal next_owner
            owner_workdir = workdir / f"owner-{next_owner}"
            next_owner += 1
            return stack.enter_context(
                atnagent_arena_process(
                    workdir=owner_workdir,
                    cuda_device=0,
                    max_tokens=max_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    atn_dp_size=atn_dp_size,
                    activate_resident=activate_resident,
                    instance_index=instance_index,
                    instance_rank=instance_rank,
                )
            )

        yield create
