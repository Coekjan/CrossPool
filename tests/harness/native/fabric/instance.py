from __future__ import annotations

from multiprocessing.connection import Connection

import torch

import xpool.native
import xpool.ops
from tests.harness.native.fabric.protocol import (
    FABRIC_TIMEOUT_SECONDS,
    FabricInstanceCommand,
    FabricInstanceReady,
    FabricInstanceResult,
    FabricInstanceRunning,
    FabricInstanceSpec,
)
from tests.harness.native.fabric.trace import fabric_debug_options
from tests.harness.native.ffn.qualification import EXECUTION_HIDDEN_SIZE, execution_reference
from xpool.cext import ensure_native_loaded
from xpool.native import RuntimeRole
from xpool.native.ffn import DpRowLayout, ResultCode
from xpool.transport import FfnRequestMetadata


def run_fabric_instance(connection: Connection, spec: FabricInstanceSpec) -> None:
    """Attach one Instance rank and return repeated real FFN outputs."""

    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    xpool.native.initialize(
        RuntimeRole.INSTANCE,
        spec.device,
        fabric_debug_options(),
    )
    hidden_size = EXECUTION_HIDDEN_SIZE
    hidden_states = (
        torch.arange(spec.payload_rows * hidden_size, device=spec.device, dtype=spec.payload_dtype)
        + spec.payload_offset
    ).reshape(spec.payload_rows, hidden_size)
    hidden_states.div_(32)
    xpool.native.transport.attach_arena(spec.instance_index, 0, spec.arena)
    try:
        connection.send(FabricInstanceReady())
        command = connection.recv()
        if command is not FabricInstanceCommand.RUN:
            raise RuntimeError(f"Fabric Instance rank expected RUN, received {command!r}")
        if spec.pre_admission_rejection:
            dp_rank_payload_rows = torch.tensor([spec.payload_rows], device=spec.device, dtype=torch.int32)
            metadata = FfnRequestMetadata(
                layer_ordinal=spec.layer_ordinals[0],
                forward_mode=spec.forward_mode,
                output_requirement=spec.output_requirement,
                dp_row_layout=DpRowLayout.UNIFORM_BY_RANK,
            )
            try:
                xpool.ops.ffn_shim(hidden_states, dp_rank_payload_rows, metadata)
            except RuntimeError as error:
                if "DP-one request must use NONE without a per-rank row vector" not in str(error):
                    raise
            else:
                raise AssertionError("Arena-incompatible FFN metadata was accepted")
            if xpool.native.transport.read_generation_failure() is not ResultCode.OK:
                raise AssertionError("pre-Admission rejection published a Generation failure")
        repetitions_completed = 0
        result_code = ResultCode.OK
        output = torch.empty_like(hidden_states)
        actual_values: list[float] = []
        expected_values: list[float] = []
        retain_each_repetition = spec.repetition_count > 1
        for repetition in range(spec.repetition_count):
            layer_ordinal = spec.layer_ordinals[repetition % len(spec.layer_ordinals)]
            spec.start_barrier.wait(timeout=FABRIC_TIMEOUT_SECONDS)
            torch.ops.xpool.ffn_shim(
                hidden_states,
                None,
                output,
                layer_ordinal,
                int(spec.forward_mode),
                int(spec.output_requirement),
                0,
            )
            repetitions_completed += 1
            if repetition == 0:
                connection.send(FabricInstanceRunning())
            torch.cuda.synchronize(spec.device)
            result_code = xpool.native.transport.read_generation_failure()
            if result_code is not ResultCode.OK:
                break
            if retain_each_repetition:
                expected = execution_reference(
                    hidden_states,
                    execution_kind=spec.layer_kind,
                    ffn_tp_size=spec.execution_tp_size,
                    output_count=spec.atnagent_count,
                    output_rank=spec.atnagent_pe,
                    output_requirement=spec.output_requirement,
                    layer_ordinal=layer_ordinal,
                )
                actual_values.extend(output.float().cpu().flatten().tolist())
                expected_values.extend(expected.float().cpu().flatten().tolist())
            if spec.stop_on_command:
                command = connection.recv()
                if command is not FabricInstanceCommand.STOP:
                    raise RuntimeError(f"Fabric Instance rank expected STOP, received {command!r}")
                break
        if retain_each_repetition:
            actual_result = tuple(actual_values)
            expected_result = tuple(expected_values)
        else:
            expected = execution_reference(
                hidden_states,
                execution_kind=spec.layer_kind,
                ffn_tp_size=spec.execution_tp_size,
                output_count=spec.atnagent_count,
                output_rank=spec.atnagent_pe,
                output_requirement=spec.output_requirement,
                layer_ordinal=spec.layer_ordinals[(repetitions_completed - 1) % len(spec.layer_ordinals)],
            )
            actual_result = tuple(output.float().cpu().flatten().tolist())
            expected_result = tuple(expected.float().cpu().flatten().tolist())
        connection.send(
            FabricInstanceResult(
                atnagent_pe=spec.atnagent_pe,
                instance_index=spec.instance_index,
                forward_mode=spec.forward_mode,
                repetitions_completed=repetitions_completed,
                result_code=result_code,
                actual=actual_result,
                expected=expected_result,
            )
        )
    finally:
        xpool.native.transport.detach_arena()
