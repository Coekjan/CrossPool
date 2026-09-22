"""Strict declarative schema for the SGLang E2E workload manifest."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tests.harness.sglang.serving.graph import SglangGraphMode
from xpool.config import LatencySloConfig

E2E_MANIFEST_PATH = Path(__file__).with_name("manifest.toml")


class E2eModel(BaseModel):
    """One model available to declarative E2E cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    architecture: str = Field(min_length=1)


class E2eModelPlacement(BaseModel):
    """One model's placement in a shared AtnAgent topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    atn_tp_size: int = Field(gt=0)
    atn_dp_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        """Reject the deferred combined attention TP-by-DP topology."""

        if self.atn_tp_size > 1 and self.atn_dp_size > 1:
            raise ValueError("E2E does not support combined attention TP-by-DP")
        return self

    @property
    def atnagent_count(self) -> int:
        """Return the physical attention-agent count for this placement."""

        return self.atn_tp_size * self.atn_dp_size


class E2eElasticKvWorkload(BaseModel):
    """One cross-Instance prefix-cache reclamation workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prefix_model_id: str = Field(min_length=1)
    prefix_tokens: int = Field(gt=0)
    pressure_model_id: str = Field(min_length=1)
    atn_device_memory_budget_bytes: int = Field(gt=0)


class E2eServingCase(BaseModel):
    """One independently supervised model-serving workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    models: tuple[E2eModelPlacement, ...]
    ffnagent_count: int = Field(gt=0)
    executor_lane_count: int = Field(gt=0)
    graph_modes: tuple[SglangGraphMode, ...] = ()
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    transport_record_capacity: int = Field(gt=0)
    fabric_record_capacity: int = Field(gt=0)
    elastic_kv: E2eElasticKvWorkload | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        """Require one unambiguous topology and complete execution capacity."""

        if not self.models:
            raise ValueError("E2E serving case models must be nonempty")
        model_ids = tuple(placement.model_id for placement in self.models)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("E2E serving case model IDs must be unique")
        topologies = {(placement.atn_tp_size, placement.atn_dp_size) for placement in self.models}
        if len(topologies) != 1:
            raise ValueError("every E2E model must use the same attention topology")
        if self.elastic_kv is not None:
            if self.graph_modes:
                raise ValueError("E2E elastic KV workload owns its graph mode")
            workload_models = {self.elastic_kv.prefix_model_id, self.elastic_kv.pressure_model_id}
            if len(workload_models) != 2 or not workload_models <= set(model_ids):
                raise ValueError("E2E elastic KV workload requires two distinct models from its serving case")
        elif not self.graph_modes or len(self.graph_modes) != len(set(self.graph_modes)):
            raise ValueError("ordinary E2E serving case graph_modes must be nonempty and unique")
        return self

    @property
    def atnagent_count(self) -> int:
        """Return the shared physical AtnAgent world size."""

        placement = self.models[0]
        return placement.atn_tp_size * placement.atn_dp_size

    @property
    def required_gpu_count(self) -> int:
        """Return the exclusive physical GPU requirement."""

        return self.atnagent_count + self.ffnagent_count


class E2eFfnInputMatrix(BaseModel):
    """Deterministic hidden-state dimensions for one FFN numerical case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0)
    row_counts: tuple[int, ...]

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        """Require nonempty positive rows in strict increasing order."""

        if not self.row_counts or any(row_count <= 0 for row_count in self.row_counts):
            raise ValueError("FFN numerical row counts must be nonempty and positive")
        if tuple(sorted(set(self.row_counts))) != self.row_counts:
            raise ValueError("FFN numerical row counts must be strictly increasing")
        return self


class E2eFfnNumericalCase(BaseModel):
    """One independent real-checkpoint FFN numerical workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    model_placement: E2eModelPlacement
    layer_ids: tuple[int, ...]
    input_matrix: E2eFfnInputMatrix
    ffnagent_count: int = Field(gt=0)
    executor_lane_count: int = Field(gt=0)
    ffn_tp_size: int = Field(gt=0)
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        """Require representative layers and an executable TP width."""

        if not self.layer_ids or any(layer_id < 0 for layer_id in self.layer_ids):
            raise ValueError("FFN numerical layer IDs must be nonempty and nonnegative")
        if tuple(sorted(set(self.layer_ids))) != self.layer_ids:
            raise ValueError("FFN numerical layer IDs must be strictly increasing")
        if self.ffn_tp_size > self.ffnagent_count:
            raise ValueError("FFN numerical TP size must fit the FfnAgent fleet")
        return self

    @property
    def production_required_gpu_count(self) -> int:
        """Return the production world's exclusive GPU count."""

        return self.model_placement.atnagent_count + self.ffnagent_count


class E2eFfnTopologyInstance(BaseModel):
    """One model-layer coordinate inside a topology qualification world."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    layer_ordinal: int = Field(ge=0)
    atn_tp_size: int = Field(gt=0)
    atn_dp_size: int = Field(gt=0)
    ffn_tp_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        """Reject the deferred combined attention TP-by-DP topology."""

        if self.atn_tp_size > 1 and self.atn_dp_size > 1:
            raise ValueError("E2E does not support combined attention TP-by-DP")
        return self


class E2eFfnTopologyRequest(BaseModel):
    """One controlled invocation in a topology qualification world."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_index: int = Field(ge=0)
    forward_mode: Literal["decode", "prefill"]
    dp_rank_payload_rows: tuple[int, ...]
    output_requirement: Literal["per_rank_complete", "group_sum_complete"]


class E2eFfnTopologyCase(BaseModel):
    """One installed real-FFN topology qualification world."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    atnagent_count: int = Field(gt=0)
    ffnagent_count: int = Field(gt=0)
    executor_lane_count: int = Field(gt=0)
    instances: tuple[E2eFfnTopologyInstance, ...]
    requests: tuple[E2eFfnTopologyRequest, ...]
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        """Require every request and placement to fit this concrete world."""

        if not self.instances or not self.requests:
            raise ValueError("FFN topology instances and requests must be nonempty")
        model_ids = tuple(instance.model_id for instance in self.instances)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("FFN topology model IDs must be unique")
        for instance in self.instances:
            if instance.atn_tp_size * instance.atn_dp_size > self.atnagent_count:
                raise ValueError("FFN topology Instance attention width exceeds the AtnAgent fleet")
            if instance.ffn_tp_size > self.ffnagent_count:
                raise ValueError("FFN topology Instance TP size exceeds the FfnAgent fleet")
        for request in self.requests:
            if request.instance_index >= len(self.instances):
                raise ValueError("FFN topology request references an unknown Instance")
            instance = self.instances[request.instance_index]
            if len(request.dp_rank_payload_rows) != instance.atn_dp_size:
                raise ValueError("FFN topology DP-rank payload rows must match attention DP size")
            if any(count < 0 for count in request.dp_rank_payload_rows) or not any(request.dp_rank_payload_rows):
                raise ValueError("FFN topology DP-rank payload rows must be nonnegative with a positive row")
        return self

    @property
    def required_gpu_count(self) -> int:
        """Return the world's exclusive physical GPU count."""

        return self.atnagent_count + self.ffnagent_count


class E2eManifest(BaseModel):
    """Shared model catalog, serving workloads, and FFN topology workloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: tuple[E2eModel, ...]
    serving_slo: LatencySloConfig
    model_serving_cases: tuple[E2eServingCase, ...]
    ffn_topology_cases: tuple[E2eFfnTopologyCase, ...]

    @classmethod
    def load(cls, path: Path) -> E2eManifest:
        """Load and strictly validate one TOML manifest."""

        with path.open("rb") as manifest_file:
            raw = tomllib.load(manifest_file)
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require unique identities and resolvable case model IDs."""

        if not self.models:
            raise ValueError("E2E manifest models must be nonempty")
        model_ids = tuple(model.model_id for model in self.models)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("E2E manifest model IDs must be unique")
        cases = (*self.model_serving_cases, *self.ffn_topology_cases)
        if not self.model_serving_cases or not self.ffn_topology_cases:
            raise ValueError("E2E manifest workload tables must be nonempty")
        case_ids = tuple(case.id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("E2E manifest case IDs must be unique")
        known_model_ids = set(model_ids)
        referenced_model_ids = {placement.model_id for case in self.model_serving_cases for placement in case.models}
        referenced_model_ids.update(
            instance.model_id for case in self.ffn_topology_cases for instance in case.instances
        )
        unknown_model_ids = sorted(referenced_model_ids - known_model_ids)
        if unknown_model_ids:
            raise ValueError(f"E2E manifest references unknown model IDs: {unknown_model_ids}")
        return self

    def model(self, model_id: str) -> E2eModel:
        """Return one catalog model by its validated full ID."""

        return next(model for model in self.models if model.model_id == model_id)
