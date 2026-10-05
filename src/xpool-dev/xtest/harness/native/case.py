"""Fresh-process execution for native integration cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path

from xkit.child import PythonChildProcess
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
    workdir: Path,
    timeout_seconds: float = NATIVE_CASE_TIMEOUT_SECONDS,
) -> None:
    """Execute a source callback using the invocation directory as its import root.

    The workdir contains artifacts and does not determine source resolution.
    """

    workdir.mkdir(parents=True, exist_ok=False)
    name = getattr(target, "__name__", type(target).__name__)
    process = PythonChildProcess(
        name,
        execute_native_case,
        NativeCaseSpec(target=target, arguments=arguments),
        log_path=workdir / "case.log",
        import_paths=(Path.cwd(),),
    )
    try:
        process.start()
        process.wait(timeout_seconds=timeout_seconds)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()
