from __future__ import annotations

import json
import selectors
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

DEVAGENT_ARENA_SCRIPT = r"""
import json
import sys
import traceback

import torch

from xpool.abi import DebugOption, RuntimeRole
from xpool.cext import ensure_xpool_ops_loaded

try:
    cuda_device = int(sys.argv[1])
    max_tokens = int(sys.argv[2])
    hidden_size = int(sys.argv[3])
    element_size = int(sys.argv[4])
    atn_dp_size = int(sys.argv[5])
    launch_kernel = sys.argv[6] == "1"
    transport_loopback_enabled = sys.argv[7] == "1"
    observer_output_path = sys.argv[8]

    ensure_xpool_ops_loaded()
    debug_options = DebugOption.TRANSPORT_LOOPBACK if transport_loopback_enabled else DebugOption(0)
    if observer_output_path:
        debug_options |= DebugOption.TRANSPORT_OBSERVER
    torch.ops.xpool.init(cuda_device, int(RuntimeRole.DEVAGENT), int(debug_options))
    arena = torch.ops.xpool.devagent.create_transport_arena(
        cuda_device,
        max_tokens,
        hidden_size,
        element_size,
        atn_dp_size,
    )
    if launch_kernel:
        torch.ops.xpool.devagent.launch_transport_kernel(arena)
    print(json.dumps({"handle": arena}), flush=True)
    command = sys.stdin.readline().strip()
    if command == "destroy":
        sequence, dropped, records = torch.ops.xpool.devagent.destroy_transport_arena(arena)
        if observer_output_path:
            with open(observer_output_path, "w", encoding="utf-8") as output_file:
                json.dump(
                    {"sequence": sequence, "dropped": dropped, "records": records},
                    output_file,
                )
except BaseException:
    traceback.print_exc()
    sys.exit(1)
"""


@contextmanager
def devagent_arena_process(
    *,
    cuda_device: int,
    max_tokens: int,
    hidden_size: int,
    element_size: int,
    atn_dp_size: int,
    launch_kernel: bool,
    transport_loopback_enabled: bool = True,
    observer_output_path: str = "",
    startup_timeout_s: float = 30.0,
) -> Iterator[str]:
    """Start a devagent subprocess that owns one native transport arena.

    Args:
        cuda_device: CUDA device that owns the arena allocation.
        max_tokens: Maximum token rows supported by one slot.
        hidden_size: Hidden-state width supported by one slot.
        element_size: Hidden-state element size in bytes.
        atn_dp_size: Attention data-parallel world size for DP token counts.
        launch_kernel: Whether to start the persistent transport kernel before
            returning the arena.
        transport_loopback_enabled: Whether the devagent subprocess should
            launch the debug loopback transport kernel.
        observer_output_path: Optional JSON path that receives the native
            observer snapshot immediately before arena destruction.
        startup_timeout_s: Seconds to wait for the subprocess to publish arena
            arena.

    Yields:
        Native transport arena handle.

    Side Effects:
        Spawns a Python subprocess and destroys the arena before exit.
    """

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            DEVAGENT_ARENA_SCRIPT,
            str(cuda_device),
            str(max_tokens),
            str(hidden_size),
            str(element_size),
            str(atn_dp_size),
            "1" if launch_kernel else "0",
            "1" if transport_loopback_enabled else "0",
            observer_output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("devagent subprocess stdout was not captured")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(startup_timeout_s):
            terminate_process(process)
            raise RuntimeError("devagent subprocess timed out before publishing arena")
        line = process.stdout.readline()
        if not line:
            stderr = process.communicate(timeout=5)[1]
            raise RuntimeError(f"devagent subprocess exited before publishing arena: {stderr}")
        payload = json.loads(line)
        yield str(payload["handle"])
    finally:
        if process.poll() is None:
            if process.stdin is not None:
                process.stdin.write("destroy\n")
                process.stdin.flush()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                terminate_process(process)


def terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)
