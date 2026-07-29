"""Typed subprocess harness for native multi-PE Fabric tests."""

from __future__ import annotations

from multiprocessing.connection import Connection

import torch

import xpool.native
from tests.harness.native.fabric.protocol import (
    FABRIC_LOOPBACK_TIMEOUT_SECONDS,
    FabricInstanceCommand,
    FabricInstanceReady,
    FabricInstanceResult,
    FabricInstanceRunning,
    FabricInstanceSpec,
)
from tests.harness.native.fabric.trace import fabric_debug_options, torch_dtype
from xpool.abi import FfnResultCode
from xpool.cext import ensure_native_loaded
from xpool.runtime import RuntimeRole


def run_fabric_instance(connection: Connection, spec: FabricInstanceSpec) -> None:
    """Attach one Instance and return repeated FFN loopback outputs."""

    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    xpool.native.initialize(
        int(RuntimeRole.INSTANCE),
        spec.device,
        fabric_debug_options(loopback_enabled=spec.loopback_enabled),
    )
    dtype = torch_dtype(spec.dtype)
    hidden_states = (torch.arange(32, device=spec.device, dtype=dtype) + spec.payload_offset).reshape(4, 8)
    xpool.native.transport.attach_arena(spec.instance_index, 0, spec.arena)
    try:
        x_values = hidden_states[..., 0::2]
        y_values = hidden_states[..., 1::2]
        expected = torch.empty_like(hidden_states)
        expected[..., 0::2] = (x_values - y_values) / (2.0**0.5)
        expected[..., 1::2] = (x_values + y_values) / (2.0**0.5)
        connection.send(FabricInstanceReady())
        command = connection.recv()
        if command is not FabricInstanceCommand.RUN:
            raise RuntimeError(f"Fabric Instance expected RUN, received {command!r}")
        repetitions_completed = 0
        result_code = FfnResultCode.OK
        output = torch.empty_like(hidden_states)
        for repetition in range(spec.repetition_count):
            spec.start_barrier.wait(timeout=FABRIC_LOOPBACK_TIMEOUT_SECONDS)
            output = torch.ops.xpool.ffn_shim(
                hidden_states,
                None,
                0,
                int(spec.forward_mode),
                1,
                0,
            )
            repetitions_completed += 1
            if repetition == 0:
                connection.send(FabricInstanceRunning())
            torch.cuda.synchronize(spec.device)
            result_code = FfnResultCode(xpool.native.transport.read_generation_failure())
            if result_code is not FfnResultCode.OK:
                break
            if spec.stop_on_command:
                command = connection.recv()
                if command is not FabricInstanceCommand.STOP:
                    raise RuntimeError(f"Fabric Instance expected STOP, received {command!r}")
                break
        connection.send(
            FabricInstanceResult(
                atnagent_pe=spec.atnagent_pe,
                instance_index=spec.instance_index,
                forward_mode=spec.forward_mode,
                repetitions_completed=repetitions_completed,
                result_code=result_code,
                actual=tuple(output.float().cpu().flatten().tolist()),
                expected=tuple(expected.float().cpu().flatten().tolist()),
            )
        )
    finally:
        xpool.native.transport.detach_arena()
