from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path

import xpool.native
from tests.harness.native.fabric.protocol import (
    FABRIC_TIMEOUT_SECONDS,
    FabricBootstrapCommand,
    FabricBootstrapStopped,
    FabricUidCreated,
)
from tests.harness.runner.child import PythonChildProcess
from xpool.cext import ensure_native_loaded
from xpool.fabric import FabricUid
from xpool.native import RuntimeRole


def create_fabric_uid_child(connection: Connection, spec: None) -> None:
    """Initialize a daemon-role child and publish one native Fabric UID."""

    ensure_native_loaded()
    xpool.native.initialize(RuntimeRole.DAEMON)
    connection.send(FabricUidCreated(xpool.native.fabric.create_uid()))


def run_fabric_bootstrap(connection: Connection, spec: None) -> None:
    """Create a UID and keep its socket bootstrap owner alive."""

    ensure_native_loaded()
    xpool.native.initialize(RuntimeRole.DAEMON)
    connection.send(FabricUidCreated(xpool.native.fabric.create_uid()))
    command = connection.recv()
    if command is not FabricBootstrapCommand.STOP:
        raise RuntimeError(f"Fabric bootstrap expected STOP, received {command!r}")
    connection.send(FabricBootstrapStopped())


def create_fabric_uid(*, workdir: Path) -> FabricUid:
    """Create one native NVSHMEM UID in a fresh daemon-role child."""

    workdir.mkdir(parents=True, exist_ok=False)
    process = PythonChildProcess.start(
        "fabric-uid",
        create_fabric_uid_child,
        None,
        log_path=workdir / "uid.log",
    )
    try:
        created = process.receive(FabricUidCreated, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        process.wait(timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        return FabricUid(value=created.value)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()


@contextmanager
def fabric_bootstrap(*, workdir: Path) -> Generator[FabricUid, None, None]:
    """Keep the daemon-role owner of one UID alive for a Fabric generation."""

    workdir.mkdir(parents=True, exist_ok=False)
    process = PythonChildProcess.start(
        "fabric-bootstrap",
        run_fabric_bootstrap,
        None,
        log_path=workdir / "bootstrap.log",
    )
    try:
        created = process.receive(FabricUidCreated, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        yield FabricUid(value=created.value)
        process.send(FabricBootstrapCommand.STOP)
        process.receive(FabricBootstrapStopped, timeout_seconds=FABRIC_TIMEOUT_SECONDS)
        process.wait(timeout_seconds=FABRIC_TIMEOUT_SECONDS)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()
