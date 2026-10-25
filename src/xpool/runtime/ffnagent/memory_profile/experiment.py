"""Concrete calibration definitions and their configuration/Plan projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from xpool import ffn
from xpool.config import FfnSchedulingPolicy, ModelConfig, XpoolConfig
from xpool.fabric import (
    FabricGenerationId,
    FabricInstancePlan,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FfnModelPlan,
    FifoSchedulerPolicy,
    InstanceFfnLayerProfile,
    InstanceFfnProfile,
    InstanceRankTopology,
    place_ffn_layer,
)
from xpool.runtime.ffnagent import preparation


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    """One synthetic FFN specification and its experimental placement."""

    spec: ffn.FfnModelSpec
    ffn_tp_size: int
    topology: InstanceRankTopology
    group_sum_complete_admitted: bool

    def model_plan(self) -> FfnModelPlan:
        """Place every layer on the ordered primary FFN TP group."""

        group = tuple(range(self.ffn_tp_size))
        return FfnModelPlan(
            model_spec_digest=self.spec.digest(),
            layers=tuple(
                place_ffn_layer(
                    layer,
                    group,
                    local_intermediate_size=preparation.prepare_ffn_layer(
                        self.spec,
                        layer_ordinal=ordinal,
                        payload_dtype=torch.bfloat16,
                        tp_size=self.ffn_tp_size,
                        tp_rank=0,
                    ).local_intermediate_size,
                )
                for ordinal, layer in enumerate(self.spec.layers)
            ),
        )

    def instance_plan(self) -> FabricInstancePlan:
        """Project the fixed BF16 capacities and declared attention membership."""

        return FabricInstancePlan(
            model_id=self.spec.model_id,
            ffn_profile=InstanceFfnProfile(
                payload_dtype=torch.bfloat16,
                hidden_size=self.spec.hidden_size,
                layers=tuple(
                    InstanceFfnLayerProfile(layer_id=layer.layer_id, kind=layer.kind) for layer in self.spec.layers
                ),
                decode_payload_row_capacity=2048,
                prefill_payload_row_capacity=4096,
                group_sum_complete_admitted=self.group_sum_complete_admitted,
            ),
            instance_rank_topology=self.topology,
        )

    def to_json_mapping(self) -> dict[str, object]:
        """Retain the specification and effective placement without runtime state."""

        return {
            "spec": self.spec.model_dump(mode="json"),
            "ffn_tp_size": self.ffn_tp_size,
            "instance_plan": self.instance_plan().model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class CalibrationExperiment:
    """An ordered model combination shared by configuration, execution and evidence."""

    models: tuple[CalibrationModel, ...]

    @property
    def model_specs(self) -> tuple[ffn.FfnModelSpec, ...]:
        return tuple(model.spec for model in self.models)

    def config_snapshot(self, source: XpoolConfig, *, directory: Path) -> XpoolConfig:
        """Disable calibration input while retaining the resolved complete fleet.

        Synthetic paths identify models without checkpoint I/O. Per-instance
        attention membership belongs to the Instance Plan, not ModelConfig.
        """

        return source.model_copy(
            update={
                "scheduler": source.scheduler.model_copy(
                    update={"atn_concurrency": 1, "ffn_policy": FfnSchedulingPolicy.FIFO}
                ),
                "ffn": source.ffn.model_copy(
                    update={"device_memory_calibration": None, "device_memory_extra_margin_bytes": 0}
                ),
                "models": [
                    ModelConfig(
                        id=model.spec.model_id,
                        path=directory / f"calibration-member-{index}",
                        ffn_tp_size=model.ffn_tp_size,
                    )
                    for index, model in enumerate(self.models)
                ],
            }
        )

    def fabric_plan(self, uid: str, *, config: XpoolConfig) -> FabricPlan:
        """Compose a fresh generation over the complete configured agent fleet."""

        return FabricPlan(
            generation=FabricGenerationId.create(),
            uid=FabricUid(uid),
            pe_placements=tuple(
                FabricPePlacement(role=FabricRole.ATNAGENT, device=device) for device in config.atn.devices
            )
            + tuple(FabricPePlacement(role=FabricRole.FFNAGENT, device=device) for device in config.ffn.devices),
            executor_lane_count=config.scheduler.ffn_concurrency,
            scheduler=FifoSchedulerPolicy(),
            model_plans=tuple(model.model_plan() for model in self.models),
            instance_plans=tuple(model.instance_plan() for model in self.models),
        )

    def to_json_mapping(self) -> dict[str, object]:
        return {"models": [model.to_json_mapping() for model in self.models]}
