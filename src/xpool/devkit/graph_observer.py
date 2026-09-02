"""Immutable FfnAgent CUDA Graph Observer snapshots."""

from __future__ import annotations

import functools
from pathlib import Path
from threading import Lock

# cuda-python exposes this binary extension without Python type stubs.
import cuda.bindings.runtime  # ty: ignore[unresolved-import]
import torch
from pydantic import BaseModel, ConfigDict, Field

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationId
from xpool.native import RuntimeRole
from xpool.runtime.ffnagent.agent import FfnAgent

runtime_roles = frozenset({RuntimeRole.FFNAGENT})
install_lock = Lock()
installed = False


class FfnPrimaryGraphSnapshot(BaseModel):
    """Actual structure and binding sites of one Primary Graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_counts: dict[str, int]
    binding_site_count: int = Field(ge=0)


class FfnLaneGraphSnapshot(BaseModel):
    """Actual recursive node and branch inventory of one Lane Graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_counts: dict[str, int]
    compute_branch_count: int = Field(ge=0)
    delivery_branch_count: int = Field(ge=0)


class FfnGraphObserverSnapshot(BaseModel):
    """One FfnAgent's immutable generation-scoped Graph evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: FabricGenerationId
    pe: int = Field(ge=0)
    cuda_device: int = Field(ge=0)
    gpu_uuid: str = Field(min_length=1)
    primary_graphs: tuple[FfnPrimaryGraphSnapshot, ...]
    lane_graphs: tuple[FfnLaneGraphSnapshot, ...] = Field(min_length=1)


def node_count_names(node_counts: dict[int, int]) -> dict[str, int]:
    """Resolve native CUDA node-kind values into JSON object keys."""

    return {cuda.bindings.runtime.cudaGraphNodeType(node_kind).name: count for node_kind, count in node_counts.items()}


def graph_snapshot(
    agent: FfnAgent,
    snapshot: xpool.native.devkit.graph_observer.Snapshot,
) -> FfnGraphObserverSnapshot:
    """Attach process identity to one native Graph snapshot."""

    plan = agent.fabric_plan
    if plan is None:
        raise RuntimeError("xpool Graph snapshot requires a retained Fabric Plan")
    primary_graphs = tuple(
        FfnPrimaryGraphSnapshot(
            node_counts=node_count_names(primary_graph.node_counts),
            binding_site_count=primary_graph.binding_site_count,
        )
        for primary_graph in snapshot.primary_graphs
    )

    lane_graphs = tuple(
        FfnLaneGraphSnapshot(
            node_counts=node_count_names(lane.node_counts),
            compute_branch_count=lane.compute_branch_count,
            delivery_branch_count=lane.delivery_branch_count,
        )
        for lane in snapshot.lane_graphs
    )
    return FfnGraphObserverSnapshot(
        generation=plan.generation,
        pe=agent.fabric_pe(),
        cuda_device=agent.cuda_device,
        gpu_uuid=str(torch.cuda.get_device_properties(agent.cuda_device).uuid),
        primary_graphs=primary_graphs,
        lane_graphs=lane_graphs,
    )


def write_graph_snapshot(agent: FfnAgent, snapshot: xpool.native.devkit.graph_observer.Snapshot) -> Path:
    """Atomically write one FfnAgent Graph Observer snapshot."""

    outdir = get_global_config().debug.graph_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool graph observer requires debug.graph_observer.outdir")
    observer_snapshot = graph_snapshot(agent, snapshot)
    generation = observer_snapshot.generation.format()
    path = outdir / f"xpool.graph-observer.{generation}.{observer_snapshot.pe}.json"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(observer_snapshot.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def install() -> None:
    """Write the native Graph snapshot after successful FFN installation."""

    outdir = get_global_config().debug.graph_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool graph observer requires debug.graph_observer.outdir")
    global installed
    with install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        if installed:
            return
        original_prepare = FfnAgent.prepare_fabric_execution

        @functools.wraps(original_prepare)
        def observed_prepare(agent: FfnAgent) -> None:
            original_prepare(agent)
            snapshot = xpool.native.devkit.graph_observer.read()
            if snapshot is not None:
                write_graph_snapshot(agent, snapshot)

        setattr(FfnAgent, "prepare_fabric_execution", observed_prepare)
        installed = True
