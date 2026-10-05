from __future__ import annotations

from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from xkit.child import PythonChildProcess
from xkit.supervisor import prepare_task_supervision

REPO_ROOT = Path(__file__).resolve().parents[4]


def echo_spawned_value(connection: Connection, value: str) -> None:
    connection.send(value)


def fail_spawned_process(connection: Connection, value: str) -> None:
    raise RuntimeError(value)


def test_spawned_process_exchanges_typed_message_and_closes(tmp_path: Path) -> None:
    prepare_task_supervision()
    process = PythonChildProcess(
        "echo", echo_spawned_value, "hello", log_path=tmp_path / "echo.log", import_paths=(REPO_ROOT,)
    )

    try:
        process.start()
        assert process.receive(str, timeout_seconds=5.0) == "hello"
        process.wait(timeout_seconds=5.0)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()


def test_spawned_process_propagates_child_traceback(tmp_path: Path) -> None:
    prepare_task_supervision()
    process = PythonChildProcess(
        "failure",
        fail_spawned_process,
        "expected child failure",
        log_path=tmp_path / "failure.log",
        import_paths=(REPO_ROOT,),
    )
    try:
        process.start()
        with pytest.raises(RuntimeError, match="expected child failure"):
            process.wait(timeout_seconds=5.0)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()
