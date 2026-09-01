from __future__ import annotations

import time
from multiprocessing.connection import Connection

import torch

import xpool.native
from tests.harness.native.fabric.protocol import (
    FabricArenasPublished,
    FabricParticipantCommand,
    FabricParticipantDrained,
    FabricParticipantQuiesced,
    FabricParticipantReady,
    FabricParticipantSpec,
)
from tests.harness.native.fabric.trace import fabric_debug_options, fabric_trace_report
from tests.harness.native.ffn.qualification import (
    EXECUTION_HIDDEN_SIZE,
    dense_layer_weights,
    execution_fabric_plan,
    execution_model_specs,
    moe_layer_weights,
)
from xpool.cext import ensure_native_loaded
from xpool.native import RuntimeRole
from xpool.native.ffn import LayerKind
from xpool.runtime.agent import project_fabric_arena
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.runtime.ffnagent.weights import FfnLayerWeights


def run_fabric_participant(connection: Connection, spec: FabricParticipantSpec) -> None:
    """Join one Fabric PE and serve typed lifecycle commands."""

    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    xpool.native.initialize(
        spec.role,
        spec.device,
        fabric_debug_options(
            graph_observer=True,
            routing_observer=spec.layer_kind is LayerKind.MOE,
        ),
    )
    fabric_plan = execution_fabric_plan(
        uid=spec.uid,
        atnagent_count=spec.atnagent_count,
        ffnagent_count=spec.ffnagent_count,
        ffn_tp_size=spec.execution_tp_size,
        executor_lane_count=spec.executor_lane_count,
        instance_count=len(spec.forward_modes),
        execution_kind=spec.layer_kind,
        layer_count=spec.execution_layer_count,
        decode_payload_row_capacity=spec.decode_payload_row_capacity,
        prefill_payload_row_capacity=spec.prefill_payload_row_capacity,
        payload_dtype=spec.payload_dtype,
    )
    projection = project_fabric_arena(fabric_plan)
    xpool.native.fabric.join(projection, spec.pe)
    execution_registry: FfnExecutionRegistry | None = None
    if spec.role is RuntimeRole.FFNAGENT:
        ffnagent_index = spec.pe - spec.atnagent_count
        model_specs = execution_model_specs(
            instance_count=len(spec.forward_modes),
            execution_kind=spec.layer_kind,
            layer_count=spec.execution_layer_count,
        )

        def local_layer_weights(layer_ordinal: int) -> FfnLayerWeights | None:
            if ffnagent_index >= spec.execution_tp_size:
                return None
            factory = dense_layer_weights if spec.layer_kind is LayerKind.DENSE else moe_layer_weights
            return factory(
                ffnagent_index=ffnagent_index,
                ffnagent_count=spec.execution_tp_size,
                device=spec.device,
                payload_dtype=spec.payload_dtype,
                layer_ordinal=layer_ordinal,
            )

        layer_weights = tuple(
            tuple(local_layer_weights(layer_ordinal) for layer_ordinal in range(spec.execution_layer_count))
            for _ in spec.forward_modes
        )
        execution_registry = FfnExecutionRegistry.materialize(
            fabric_plan=fabric_plan,
            model_specs=model_specs,
            ffnagent_index=ffnagent_index,
            layer_weights=layer_weights,
        )
    arenas: list[str] = []
    if spec.role is RuntimeRole.FFNAGENT:
        xpool.native.ffnagent.activate()
        connection.send(FabricParticipantReady())
    else:
        for instance_index in range(len(spec.forward_modes)):
            arena = xpool.native.transport.create_arena(
                instance_index,
                0,
                max(spec.decode_payload_row_capacity, spec.prefill_payload_row_capacity),
                EXECUTION_HIDDEN_SIZE,
                spec.payload_dtype,
                spec.pe,
                spec.atnagent_count,
                0,
                1,
            )
            arenas.append(str(arena))
        xpool.native.transport.activate()
        connection.send(FabricArenasPublished(tuple(arenas)))

    command = connection.recv()
    if command is not FabricParticipantCommand.QUIESCE:
        raise RuntimeError(f"Fabric participant expected QUIESCE, received {command!r}")
    if spec.role is RuntimeRole.FFNAGENT:
        xpool.native.ffnagent.check_health()
    if arenas:
        xpool.native.transport.check_health()
        xpool.native.transport.drain_async()
        deadline = time.monotonic() + 60.0
        while xpool.native.transport.drain_pending():
            if time.monotonic() >= deadline:
                raise RuntimeError("Transport Resident drain timed out")
            time.sleep(0.01)
        xpool.native.transport.drain_async()
        if xpool.native.transport.drain_pending():
            raise RuntimeError("completed Transport Resident drain became pending again")
    connection.send(FabricParticipantQuiesced())

    command = connection.recv()
    if command is not FabricParticipantCommand.DRAIN:
        raise RuntimeError(f"Fabric participant expected DRAIN, received {command!r}")
    xpool.native.fabric.drain_async()
    deadline = time.monotonic() + 60.0
    while xpool.native.fabric.drain_pending():
        if time.monotonic() >= deadline:
            raise RuntimeError("Fabric drain timed out")
        time.sleep(0.01)
    xpool.native.fabric.drain_async()
    if xpool.native.fabric.drain_pending():
        raise RuntimeError("completed Fabric drain became pending again")
    snapshot = xpool.native.devkit.fabric_observer.read()
    if snapshot is None:
        raise RuntimeError("Fabric trace observation was enabled but returned no snapshot")
    graph_snapshot = xpool.native.devkit.graph_observer.read() if spec.role is RuntimeRole.FFNAGENT else None
    routing_snapshot = (
        xpool.native.devkit.ffn_routing_observer.read()
        if spec.role is RuntimeRole.FFNAGENT and spec.layer_kind is LayerKind.MOE
        else None
    )
    report = fabric_trace_report(
        spec.role,
        snapshot,
        xpool.native.fabric.failure(),
        graph_snapshot,
        routing_snapshot,
    )
    xpool.native.fabric.finalize()
    del execution_registry
    for operation in (
        xpool.native.fabric.drain_async,
        xpool.native.fabric.drain_pending,
    ):
        try:
            operation()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Fabric drain operation was accepted after finalization")
    if arenas:
        xpool.native.transport.destroy_arenas(arenas)
    connection.send(FabricParticipantDrained(report))
