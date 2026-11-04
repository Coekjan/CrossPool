from __future__ import annotations

import logging
import os
import time
from multiprocessing.connection import Connection

import torch

import xpool.native
from xpool.cext import ensure_native_loaded
from xpool.fabric import FabricPlan
from xpool.memory import FfnExecutionContract
from xpool.native import RuntimeRole
from xpool.native.ffn import LayerKind
from xpool.runtime.agent import project_fabric_arena
from xpool.runtime.ffnagent import capture_workspace
from xpool.runtime.ffnagent.execution import OwnedExecutionResource
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.runtime.ffnagent.weights import FfnLayerWeights
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MpsEndpoint
from xtest.harness.native.fabric.protocol import (
    FabricArenasPublished,
    FabricParticipantCommand,
    FabricParticipantDrained,
    FabricParticipantQuiesced,
    FabricParticipantReady,
    FabricParticipantSpec,
    FabricParticipantStopped,
)
from xtest.harness.native.fabric.trace import fabric_debug_options, fabric_trace_report
from xtest.harness.native.ffn.qualification import (
    dense_layer_weights,
    execution_model_specs,
    moe_layer_weights,
)

logger = logging.getLogger(__name__)


def run_fabric_participant(connection: Connection, spec: FabricParticipantSpec) -> None:
    """Join one Fabric PE and serve typed lifecycle commands."""

    os.environ.update(spec.environment)
    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    if spec.role is RuntimeRole.ATNAGENT:
        MpsEndpoint(visible_uuids()[: spec.atnagent_count]).require_client()
    xpool.native.initialize(
        spec.role,
        spec.device,
        fabric_debug_options(
            graph_observer=True,
            routing_observer=spec.layer_kind is LayerKind.MOE,
        ),
    )
    finalized = False
    execution_registry: FfnExecutionRegistry | None = None
    external_resources: tuple[OwnedExecutionResource, ...] = ()
    execution_contract: FfnExecutionContract | None = None
    if spec.role is RuntimeRole.FFNAGENT:
        execution_contract, external_resources = capture_workspace.prepare_execution_sources()
        connection.send(execution_contract)
    fabric_plan = connection.recv()
    if fabric_plan is FabricParticipantCommand.STOP:
        connection.send(FabricParticipantStopped())
        return
    try:
        if not isinstance(fabric_plan, FabricPlan):
            raise RuntimeError(f"Fabric participant expected Plan, received {fabric_plan!r}")
        if execution_contract is not None and (
            fabric_plan.ffn_execution_contracts[spec.pe - spec.atnagent_count] != execution_contract
        ):
            raise RuntimeError("Fabric Plan disagrees with the participant's execution facts")
        projection = project_fabric_arena(fabric_plan)
        xpool.native.fabric.join(projection, spec.pe)
        if spec.role is RuntimeRole.FFNAGENT:
            ffnagent_index = spec.pe - spec.atnagent_count
            model_specs = execution_model_specs(
                instance_count=len(spec.forward_modes),
                execution_kind=spec.layer_kind,
                layer_count=spec.execution_layer_count,
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
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
                    hidden_size=spec.hidden_size,
                    intermediate_size=spec.intermediate_size,
                )

            layer_weights = tuple(
                tuple(local_layer_weights(layer_ordinal) for layer_ordinal in range(spec.execution_layer_count))
                for _ in spec.forward_modes
            )
            execution_registry = FfnExecutionRegistry(
                layer_weights=layer_weights, external_resources=external_resources
            )
            del layer_weights
            external_resources = ()
            execution_registry.materialize(
                fabric_plan=fabric_plan,
                model_specs=model_specs,
                ffnagent_index=ffnagent_index,
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
                    spec.hidden_size,
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
        if execution_registry is not None:
            execution_registry.close()
        del execution_registry
        finalized = True
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
    except BaseException:
        if finalized:
            raise
        logger.exception("Fabric participant retirement unconfirmed; retaining owners pe=%s", spec.pe)
        while True:
            time.sleep(1.0)
