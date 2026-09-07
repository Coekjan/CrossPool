"""Tensor-native semantic routing snapshots for drained FfnAgents."""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from threading import Lock

import safetensors.torch
import torch

import xpool.native
from xpool.config import get_global_config
from xpool.fabric import FabricGenerationId, FabricParticipantPhase
from xpool.native import RuntimeRole
from xpool.runtime.ffnagent.agent import FfnAgent

runtime_roles = frozenset({RuntimeRole.FFNAGENT})
logger = logging.getLogger(__name__)
install_lock = Lock()
installed = False


def write_routing_snapshot(
    generation: FabricGenerationId,
    pe: int,
    snapshot: xpool.native.devkit.ffn_routing_observer.Snapshot,
) -> Path:
    """Atomically write one drained process-local routing snapshot."""

    outdir = get_global_config().debug.ffn_routing_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool FFN Routing Observer requires debug.ffn_routing_observer.outdir")
    if snapshot.dropped > snapshot.sequence or len(snapshot.records) != snapshot.sequence - snapshot.dropped:
        raise RuntimeError("xpool FFN Routing Observer snapshot counters are inconsistent")

    tensors: dict[str, torch.Tensor] = {}
    records: list[dict[str, int]] = []
    for index, record in enumerate(snapshot.records):
        if (
            record.topk_ids.device.type != "cpu"
            or record.topk_ids.dtype is not torch.int32
            or record.topk_weights.device.type != "cpu"
            or record.topk_weights.dtype is not torch.float32
            or record.topk_ids.ndim != 2
            or record.topk_ids.shape != record.topk_weights.shape
            or not record.topk_ids.is_contiguous()
            or not record.topk_weights.is_contiguous()
            or any(size <= 0 for size in record.topk_ids.shape)
        ):
            raise RuntimeError("xpool FFN Routing Observer record tensors are invalid")
        records.append(
            {
                "instance_index": record.key.instance_index,
                "invocation_sequence": record.key.invocation_sequence,
                "layer_ordinal": record.layer_ordinal,
            }
        )
        tensors[f"records.{index}.topk_ids"] = record.topk_ids
        tensors[f"records.{index}.topk_weights"] = record.topk_weights

    generation_text = generation.format()
    metadata = {
        "generation": generation_text,
        "pe": str(pe),
        "sequence": str(snapshot.sequence),
        "dropped": str(snapshot.dropped),
        "records": json.dumps(records, separators=(",", ":")),
    }
    path = outdir / f"xpool.ffn-routing-observer.{generation_text}.{pe}.safetensors"
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(safetensors.torch.save(tensors, metadata=metadata))
    temporary_path.replace(path)
    return path


def install() -> None:
    """Write native routing evidence at the drained lifecycle boundary."""

    outdir = get_global_config().debug.ffn_routing_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool FFN Routing Observer requires debug.ffn_routing_observer.outdir")
    global installed
    with install_lock:
        outdir.mkdir(parents=True, exist_ok=True)
        if installed:
            return
        original_advance = FfnAgent.advance_fabric_lifecycle

        @functools.wraps(original_advance)
        def observed_advance(agent: FfnAgent) -> None:
            plan = agent.fabric_plan
            previous_report = agent.participant_report
            previous_phase = previous_report.phase if previous_report is not None else None
            original_advance(agent)
            updated_report = agent.participant_report
            if (
                plan is None
                or previous_phase is FabricParticipantPhase.DRAINED
                or updated_report is None
                or updated_report.phase is not FabricParticipantPhase.DRAINED
            ):
                return
            try:
                snapshot = xpool.native.devkit.ffn_routing_observer.read()
                if snapshot is not None:
                    write_routing_snapshot(plan.generation, agent.fabric_pe(), snapshot)
            except Exception:
                logger.warning("failed to record xpool ffn routing observer snapshot", exc_info=True)

        setattr(FfnAgent, "advance_fabric_lifecycle", observed_advance)
        installed = True
