from __future__ import annotations

import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]

type JsonValue = str | int | float | bool | list[int] | None

type GraphEvent = dict[str, JsonValue]

type GraphSettings = tuple[bool, bool]

type LoopbackMode = Literal["shim", "transport"]

PROBE_TIMEOUT_SECONDS = 30 * 60

PROCESS_TERMINATE_TIMEOUT_SECONDS = 30

PROCESS_KILL_TIMEOUT_SECONDS = 30

PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS = 30


def terminate_process_group(process: subprocess.Popen[str]) -> str:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "process group already exited before SIGTERM"
    try:
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
        return "process group exited after SIGTERM"
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "process group exited before SIGKILL"
    try:
        process.wait(timeout=PROCESS_KILL_TIMEOUT_SECONDS)
        return "process group exited after SIGKILL"
    except subprocess.TimeoutExpired:
        return "process group still running after SIGKILL"


def collect_process_output_after_timeout(process: subprocess.Popen[str]) -> tuple[str, str, str]:
    try:
        stdout, stderr = process.communicate(timeout=PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        close_process_pipes(process)
        return (
            timeout_output_text(exc.output),
            timeout_output_text(exc.stderr),
            "stdout/stderr pipes did not close after cleanup",
        )
    return stdout, stderr, "stdout/stderr drained after cleanup"


def close_process_pipes(process: subprocess.Popen[str]) -> None:
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            with suppress(OSError, ValueError):
                pipe.close()


def timeout_output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def tail(text: str, *, limit: int = 12000) -> str:
    return text[-limit:]
