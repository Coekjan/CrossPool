"""Typed subprocess harness for native multi-PE Fabric tests."""

from __future__ import annotations

import time
from multiprocessing.connection import Connection

import torch

import xpool.native
from tests.harness.native.fabric.protocol import (
    FabricArenasPublished,
    FabricParticipantActivationRejected,
    FabricParticipantCommand,
    FabricParticipantDrained,
    FabricParticipantQuiesced,
    FabricParticipantReady,
    FabricParticipantSpec,
)
from tests.harness.native.fabric.trace import fabric_debug_options, fabric_trace_report
from xpool.cext import ensure_native_loaded
from xpool.runtime import RuntimeRole


def run_fabric_participant(connection: Connection, spec: FabricParticipantSpec) -> None:
    """Join one Fabric PE and serve typed lifecycle commands."""

    ensure_native_loaded()
    torch.cuda.set_device(spec.device)
    xpool.native.initialize(
        int(spec.role),
        spec.device,
        fabric_debug_options(loopback_enabled=spec.loopback_enabled),
    )
    xpool.native.fabric.join(
        xpool.native.FabricJoinMetadata(
            uid=spec.uid,
            pe=spec.pe,
            atnagent_count=spec.atnagent_count,
            ffnagent_count=spec.ffnagent_count,
            executor_count=spec.executor_count,
            scheduler_policy=xpool.native.FfnSchedulerPolicy.fifo(),
            models=[
                xpool.native.FabricModelMetadata(
                    max_decode_rows=4,
                    max_prefill_rows=4,
                    dtype=int(spec.dtype),
                    hidden_size=8,
                    atn_tp_size=spec.atnagent_count,
                    atn_dp_size=1,
                    layers=[xpool.native.FabricLayerMetadata(layer_id=0, kind=1)],
                )
                for _ in spec.forward_modes
            ],
        )
    )
    arenas: list[str] = []
    if spec.role is RuntimeRole.FFNAGENT:
        try:
            xpool.native.fabric.activate()
        except RuntimeError as error:
            if not spec.expect_activation_rejection:
                raise
            connection.send(FabricParticipantActivationRejected(spec.pe, str(error)))
        else:
            if spec.expect_activation_rejection:
                raise RuntimeError("FfnAgent activation unexpectedly accepted an over-capacity Resident")
            connection.send(FabricParticipantReady())
    else:
        for instance_index in range(len(spec.forward_modes)):
            arena = xpool.native.transport.create_arena(
                instance_index,
                0,
                4,
                8,
                int(spec.dtype),
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
    if not spec.expect_activation_rejection:
        xpool.native.fabric.check_health()
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
    snapshot = xpool.native.fabric.read_trace()
    if snapshot is None:
        raise RuntimeError("Fabric trace observation was enabled but returned no snapshot")
    report = fabric_trace_report(spec.role, snapshot, xpool.native.fabric.failure())
    xpool.native.fabric.shutdown()
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
