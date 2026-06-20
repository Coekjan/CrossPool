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
from typing import Literal, TypeVar

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.piecewise_cuda_graph_runner import PiecewiseCudaGraphRunner

from xpool.config import get_global_config

LOGGER = logging.getLogger(__name__)
ReturnT = TypeVar("ReturnT")

GraphRunner = CudaGraphRunner | PiecewiseCudaGraphRunner
GraphKind = Literal["full_cuda_graph", "piecewise_cuda_graph"]
JsonValue = str | int | float | bool | list[int] | None
GraphEvent = dict[str, JsonValue]

_event_file: Path | None = None
_install_lock = Lock()
_write_lock = Lock()
_installed = False


def install() -> None:
    """Install the SGLang graph observer when enabled by xpool config.

    Raises:
        MissingRequiredConfig: If no process-global config has been installed.
        AssertionError: If graph-observer install is invoked while the resolved
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
    assert outdir is not None

    global _event_file, _installed
    with _install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        event_file = outdir / f"xpool.graph-observer.{os.getpid()}.jsonl"
        if _installed:
            _event_file = event_file
            event_file.touch(exist_ok=True)
            return

        _event_file = event_file
        with event_file.open("w", encoding="utf-8") as file:
            file.flush()

        setattr(
            CudaGraphRunner,
            "capture",
            _wrap_graph_method("full_cuda_graph", "capture", CudaGraphRunner.capture),
        )
        setattr(
            CudaGraphRunner,
            "replay",
            _wrap_graph_method("full_cuda_graph", "replay", CudaGraphRunner.replay),
        )
        setattr(
            PiecewiseCudaGraphRunner,
            "capture",
            _wrap_graph_method("piecewise_cuda_graph", "capture", PiecewiseCudaGraphRunner.capture),
        )
        setattr(
            PiecewiseCudaGraphRunner,
            "replay",
            _wrap_graph_method("piecewise_cuda_graph", "replay", PiecewiseCudaGraphRunner.replay),
        )
        _installed = True


def _wrap_graph_method(
    kind: GraphKind,
    method_name: str,
    original: Callable[..., ReturnT],
) -> Callable[..., ReturnT]:
    @functools.wraps(original)
    def wrapped(runner: GraphRunner, *args: object, **kwargs: object) -> ReturnT:
        _write_event(kind, f"{method_name}_begin", method_name, runner)
        try:
            result = original(runner, *args, **kwargs)
        except BaseException:
            _write_event(kind, f"{method_name}_error", method_name, runner)
            raise
        _write_event(kind, f"{method_name}_end", method_name, runner)
        return result

    return wrapped


def _write_event(kind: GraphKind, phase: str, method_name: str, runner: GraphRunner) -> None:
    try:
        event_file = _event_file
        if event_file is None:
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
        with _write_lock:
            with event_file.open("a", encoding="utf-8") as file:
                file.write(line)
                file.flush()
    except Exception:
        LOGGER.warning("Failed to record xpool graph observer event", exc_info=True)
