"""SGLang Devkit observer for CUDA Graph capture and execution events."""

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

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner

from xpool.config import get_global_config
from xpool.native import RuntimeRole

__all__ = ["install"]

runtime_roles = frozenset({RuntimeRole.INSTANCE})

logger = logging.getLogger(__name__)

type GraphRunner = DecodeCudaGraphRunner | PrefillCudaGraphRunner
type ForwardPhase = Literal["decode", "prefill"]
type GraphOperation = Literal["capture", "execute"]
type JsonValue = str | int
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
        SGLang's decode and prefill CUDA graph runner classes. Each
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
            DecodeCudaGraphRunner,
            "capture",
            wrap_graph_method("decode", "capture", DecodeCudaGraphRunner.capture),
        )
        setattr(
            DecodeCudaGraphRunner,
            "execute",
            wrap_graph_method("decode", "execute", DecodeCudaGraphRunner.execute),
        )
        setattr(
            PrefillCudaGraphRunner,
            "capture",
            wrap_graph_method("prefill", "capture", PrefillCudaGraphRunner.capture),
        )
        setattr(
            PrefillCudaGraphRunner,
            "execute",
            wrap_graph_method("prefill", "execute", PrefillCudaGraphRunner.execute),
        )
        installed = True


def wrap_graph_method[R](
    forward_phase: ForwardPhase,
    operation: GraphOperation,
    original: Callable[..., R],
) -> Callable[..., R]:
    """Wrap one SGLang graph method with begin, error, and end events.

    Args:
        forward_phase: Decode or prefill runner being observed.
        operation: Capture or execution operation being observed.
        original: Original unbound method implementation.

    Returns:
        Wrapped method preserving the original callable metadata.
    """

    @functools.wraps(original)
    def wrapped(runner: GraphRunner, *args: object, **kwargs: object) -> R:
        write_event(forward_phase, f"{operation}_begin", runner)
        try:
            result = original(runner, *args, **kwargs)
        except BaseException:
            write_event(forward_phase, f"{operation}_error", runner)
            raise
        write_event(forward_phase, f"{operation}_end", runner)
        return result

    return wrapped


def write_event(forward_phase: ForwardPhase, event: str, runner: GraphRunner) -> None:
    """Write one graph event without disrupting SGLang execution.

    Args:
        forward_phase: Decode or prefill runner being observed.
        event: Capture or execution lifecycle transition.
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
            "forward_phase": forward_phase,
            "backend_class": type(runner.backend).__name__,
            "event": event,
        }
        line = json.dumps(payload, sort_keys=True) + "\n"
        with write_lock:
            current_handle.write(line)
            current_handle.flush()
    except Exception:
        logger.warning("failed to record xpool graph observer event", exc_info=True)
