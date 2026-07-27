"""Devkit SGLang plugin that records CUDA graph capture and replay events."""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Literal, TextIO

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.piecewise_cuda_graph_runner import PiecewiseCudaGraphRunner

from xpool.config import get_global_config
from xpool.runtime import RuntimeRole

__all__ = ["install"]

runtime_roles = frozenset({RuntimeRole.INSTANCE})

logger = logging.getLogger(__name__)

type GraphRunner = CudaGraphRunner | PiecewiseCudaGraphRunner
type GraphKind = Literal["full_cuda_graph", "piecewise_cuda_graph"]
type JsonValue = str | int | float | bool | list[int] | None
type GraphEvent = dict[str, JsonValue]

event_file: Path | None = None
event_handle: TextIO | None = None
install_lock = Lock()
write_lock = Lock()
installed = False


def install() -> None:
    """Install the SGLang graph observer when enabled by xpool config.

    Raises:
        MissingRequiredConfig: If no process-global config has been installed.
        RuntimeError: If graph-observer install is invoked while the resolved
            config does not provide an output directory.
        OSError: If the configured output directory or per-process event file
            cannot be created or opened. SGLang runner import failures happen
            at module import time before this function is called.

    Side Effects:
        Creates the configured output directory if needed and monkeypatches
        SGLang's full CUDA graph and piecewise CUDA graph runner classes. Each
        event is appended to a per-process JSONL file and flushed immediately.
    """

    config = get_global_config()
    outdir = config.debug.graph_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool graph observer requires debug.graph_observer.outdir")

    global event_file, event_handle, installed
    with install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        desired_event_file = outdir / f"xpool.graph-observer.{os.getpid()}.jsonl"
        if installed:
            if event_file != desired_event_file or event_handle is None or event_handle.closed:
                if event_handle is not None and not event_handle.closed:
                    event_handle.close()
                event_handle = desired_event_file.open("a", encoding="utf-8")
            event_file = desired_event_file
            return

        if event_handle is not None and not event_handle.closed:
            event_handle.close()
        event_file = desired_event_file
        event_handle = desired_event_file.open("w", encoding="utf-8")
        event_handle.flush()

        setattr(
            CudaGraphRunner,
            "capture",
            wrap_graph_method("full_cuda_graph", "capture", CudaGraphRunner.capture),
        )
        setattr(
            CudaGraphRunner,
            "replay",
            wrap_graph_method("full_cuda_graph", "replay", CudaGraphRunner.replay),
        )
        setattr(
            PiecewiseCudaGraphRunner,
            "capture",
            wrap_graph_method("piecewise_cuda_graph", "capture", PiecewiseCudaGraphRunner.capture),
        )
        setattr(
            PiecewiseCudaGraphRunner,
            "replay",
            wrap_graph_method("piecewise_cuda_graph", "replay", PiecewiseCudaGraphRunner.replay),
        )
        installed = True


def wrap_graph_method[R](
    kind: GraphKind,
    method_name: str,
    original: Callable[..., R],
) -> Callable[..., R]:
    """Wrap one SGLang graph method with begin, error, and end events.

    Args:
        kind: Graph implementation being observed.
        method_name: SGLang method name recorded in each event.
        original: Original bound-method implementation.

    Returns:
        Wrapped method preserving the original callable metadata.
    """

    @functools.wraps(original)
    def wrapped(runner: GraphRunner, *args: object, **kwargs: object) -> R:
        write_event(kind, f"{method_name}_begin", method_name, runner)
        try:
            result = original(runner, *args, **kwargs)
        except BaseException:
            write_event(kind, f"{method_name}_error", method_name, runner)
            raise
        write_event(kind, f"{method_name}_end", method_name, runner)
        return result

    return wrapped


def write_event(kind: GraphKind, phase: str, method_name: str, runner: GraphRunner) -> None:
    """Write one graph event without disrupting SGLang execution.

    Args:
        kind: Graph implementation being observed.
        phase: Capture or replay lifecycle phase.
        method_name: SGLang method that emitted the event.
        runner: Active SGLang graph runner.

    Side Effects:
        Appends and flushes one JSON line when the observer file is available.
    """

    try:
        current_handle = event_handle
        if current_handle is None or current_handle.closed:
            return
        payload: GraphEvent = {
            "pid": os.getpid(),
            "time_ns": time.monotonic_ns(),
            "kind": kind,
            "phase": phase,
            "method": method_name,
            "runner_class": runner.__class__.__name__,
            "device": str(runner.device),
            "tp_size": runner.tp_size,
            "dp_size": runner.dp_size,
            "pp_size": runner.pp_size,
            "capture_forward_mode": runner.capture_forward_mode.name,
            "capture_bs": list(runner.capture_bs) if isinstance(runner, CudaGraphRunner) else None,
            "capture_num_tokens": (
                list(runner.capture_num_tokens) if isinstance(runner, PiecewiseCudaGraphRunner) else None
            ),
            "max_bs": runner.max_bs,
            "max_num_tokens": runner.max_num_token if isinstance(runner, CudaGraphRunner) else runner.max_num_tokens,
        }
        line = json.dumps(payload, sort_keys=True) + "\n"
        with write_lock:
            current_handle.write(line)
            current_handle.flush()
    except Exception:
        logger.warning("Failed to record xpool graph observer event", exc_info=True)
