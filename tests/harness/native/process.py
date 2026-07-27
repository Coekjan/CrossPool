"""Fresh-process execution for native integration cases."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

from tests.harness.process import SpawnedProcess
from xpool.cext import ensure_native_loaded

NATIVE_CASE_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class NativeCaseSpec:
    """One directly callable native case and its picklable arguments."""

    target: Callable[..., None]
    arguments: tuple[object, ...]


def execute_native_case(connection: Connection, spec: NativeCaseSpec) -> None:
    """Load the extension and execute one case in the spawned interpreter."""

    ensure_native_loaded()
    spec.target(*spec.arguments)


def run_native_case(
    target: Callable[..., None],
    *arguments: object,
    timeout_seconds: float = NATIVE_CASE_TIMEOUT_SECONDS,
) -> None:
    """Execute one module-level native case in a fresh spawned interpreter."""

    with tempfile.TemporaryDirectory(prefix="xpool-native-case-") as directory:
        name = getattr(target, "__name__", type(target).__name__)
        process = SpawnedProcess.start(
            name,
            execute_native_case,
            NativeCaseSpec(target=target, arguments=arguments),
            log_path=Path(directory) / "case.log",
        )
        try:
            process.wait(timeout_seconds=timeout_seconds)
        finally:
            if process.process.is_alive():
                SpawnedProcess.terminate_all((process,))
            process.close()
