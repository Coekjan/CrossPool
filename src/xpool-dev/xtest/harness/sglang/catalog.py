"""Portable qualification workloads selected by named catalogue tables."""

from __future__ import annotations

import json
import tomllib
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from xkit.config import DeploymentConfig, load_deployment
from xkit.deployment import resolve_deployment_path
from xkit.serving.sglang.graph import SglangGraphMode
from xpool.config import ConfigError
from xpool.model import ModelId


class E2eElasticKvWorkload(BaseModel):
    """One cross-Instance prefix-cache reclamation workload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prefix_model_id: ModelId
    prefix_tokens: int = Field(gt=0)
    pressure_model_id: ModelId
    atn_device_memory_budget_bytes: int = Field(gt=0)


class E2eServingCase(BaseModel):
    """Serving workload with portable geometry and declaration-owned graph policy.

    Catalogue loaders assign the table-name ID; source-owned cases may omit it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str | None = Field(default=None, min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = Field(min_length=1)
    module: str | None = Field(
        default=None, description="Catalogue-bound source module; explicit Python cases omit it."
    )
    deployment: Path
    models: tuple[ModelId, ...]
    graph_modes: tuple[SglangGraphMode, ...] = ()
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    elastic_kv: E2eElasticKvWorkload | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.models or len(self.models) != len(set(self.models)):
            raise ValueError("E2E serving case models must be nonempty and unique")
        if self.elastic_kv is not None:
            if self.graph_modes:
                raise ValueError("E2E elastic KV workload owns its graph mode")
            workload_models = {self.elastic_kv.prefix_model_id, self.elastic_kv.pressure_model_id}
            if len(workload_models) != 2 or not workload_models <= set(self.models):
                raise ValueError("E2E elastic KV workload requires two distinct models from its serving case")
        elif not self.graph_modes or len(self.graph_modes) != len(set(self.graph_modes)):
            raise ValueError("ordinary E2E serving case graph_modes must be nonempty and unique")
        return self

    @cached_property
    def deployment_config(self) -> DeploymentConfig:
        """Load the immutable case's portable topology without workspace resources."""
        return load_deployment(self.deployment, model_ids=self.models)

    @property
    def atnagent_count(self) -> int:
        return len(self.deployment_config.atn.devices)

    @property
    def ffnagent_count(self) -> int:
        return len(self.deployment_config.ffn.devices)

    @property
    def executor_lane_count(self) -> int:
        return self.deployment_config.ffn_concurrency

    @property
    def required_gpu_count(self) -> int:
        return self.atnagent_count + self.ffnagent_count


class E2eFfnInputMatrix(BaseModel):
    """Deterministic hidden-state dimensions for one FFN numerical case."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: int = Field(ge=0)
    row_counts: tuple[int, ...]

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        if not self.row_counts or any(row_count <= 0 for row_count in self.row_counts):
            raise ValueError("FFN numerical row counts must be nonempty and positive")
        if tuple(sorted(set(self.row_counts))) != self.row_counts:
            raise ValueError("FFN numerical row counts must be strictly increasing")
        return self


class E2eFfnNumericalCase(BaseModel):
    """Real-checkpoint numerical workload referencing its portable deployment."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    description: str = Field(min_length=1)
    deployment: Path
    model_id: ModelId
    layer_ids: tuple[int, ...]
    input_matrix: E2eFfnInputMatrix
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.layer_ids or any(layer_id < 0 for layer_id in self.layer_ids):
            raise ValueError("FFN numerical layer IDs must be nonempty and nonnegative")
        if tuple(sorted(set(self.layer_ids))) != self.layer_ids:
            raise ValueError("FFN numerical layer IDs must be strictly increasing")
        return self

    @cached_property
    def deployment_config(self) -> DeploymentConfig:
        """Resolve numerical resource metadata from the declared deployment."""
        return load_deployment(self.deployment, model_ids=(self.model_id,))

    @property
    def production_required_gpu_count(self) -> int:
        return len(self.deployment_config.atn.devices) + len(self.deployment_config.ffn.devices)


class E2eFfnTopologyInstance(BaseModel):
    """One model-layer coordinate inside a topology qualification world."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model_id: ModelId
    layer_ordinal: int = Field(ge=0)


class E2eFfnTopologyRequest(BaseModel):
    """One controlled invocation in a topology qualification world."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_index: int = Field(ge=0)
    forward_mode: Literal["decode", "prefill"]
    dp_rank_payload_rows: tuple[int, ...]
    output_requirement: Literal["per_rank_complete", "group_sum_complete"]


class E2eFfnTopologyCase(BaseModel):
    """Installed real-FFN qualification with deployment-owned parallel geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = Field(min_length=1)
    module: str = Field(min_length=1)
    deployment: Path
    instances: tuple[E2eFfnTopologyInstance, ...]
    requests: tuple[E2eFfnTopologyRequest, ...]
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)

    @cached_property
    def deployment_config(self) -> DeploymentConfig:
        """Load portable geometry for exactly this case's model coordinates."""
        return load_deployment(self.deployment, model_ids=tuple(instance.model_id for instance in self.instances))

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if not self.instances or not self.requests:
            raise ValueError("FFN topology instances and requests must be nonempty")
        model_ids = tuple(instance.model_id for instance in self.instances)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("FFN topology model IDs must be unique")
        models = self.deployment_config.model_by_id
        for request in self.requests:
            if request.instance_index >= len(self.instances):
                raise ValueError("FFN topology request references an unknown Instance")
            model = models[self.instances[request.instance_index].model_id]
            if len(request.dp_rank_payload_rows) != model.atn_dp_size:
                raise ValueError("FFN topology DP-rank payload rows must match attention DP size")
            if any(count < 0 for count in request.dp_rank_payload_rows) or not any(request.dp_rank_payload_rows):
                raise ValueError("FFN topology DP-rank payload rows must be nonnegative with a positive row")
        return self

    @property
    def atnagent_count(self) -> int:
        return len(self.deployment_config.atn.devices)

    @property
    def ffnagent_count(self) -> int:
        return len(self.deployment_config.ffn.devices)

    @property
    def executor_lane_count(self) -> int:
        return self.deployment_config.ffn_concurrency

    @property
    def required_gpu_count(self) -> int:
        return self.atnagent_count + self.ffnagent_count


class TestCatalog(BaseModel):
    """Named, source-bound qualification scenes with no machine-local policy."""

    __test__: ClassVar[bool] = False
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    serving_cases: tuple[E2eServingCase, ...] = ()
    topology_cases: tuple[E2eFfnTopologyCase, ...] = ()

    @classmethod
    def load(cls, path: Path) -> TestCatalog:
        """Resolve deployment basenames relative to the selected catalogue.

        Table keys supply case identities. Geometry and SLO come from complete
        portable deployments; loading probes neither hardware nor checkpoint files.
        """
        path = path.expanduser().resolve()
        with path.open("rb") as source:
            raw = tomllib.load(source)
        if unknown := raw.keys() - {"serving_cases", "topology_cases"}:
            raise ConfigError(f"unknown test catalogue fields: {sorted(unknown)}")
        cases: dict[str, object] = {}
        for family in ("serving_cases", "topology_cases"):
            declarations = raw.get(family, {})
            if not isinstance(declarations, dict):
                raise ConfigError(f"{family} must contain named case tables")
            values = []
            for id, declaration in declarations.items():
                if not isinstance(declaration, dict) or "id" in declaration:
                    raise ConfigError("the catalogue table name is the sole case identity")
                if not declaration.get("module"):
                    raise ConfigError(f"{id}: catalogue cases require module")
                model_ids = (
                    tuple(ModelId(value) for value in declaration["models"])
                    if family == "serving_cases"
                    else tuple(ModelId(value["model_id"]) for value in declaration["instances"])
                )
                declaration.update(
                    id=id,
                    deployment=str(resolve_deployment_path(path, model_ids, declaration["deployment"])),
                )
                value_type = E2eServingCase if family == "serving_cases" else E2eFfnTopologyCase
                case = value_type.model_validate_json(json.dumps(declaration, allow_nan=False))
                case.deployment_config
                values.append(case)
            cases[family] = tuple(values)
        return cls.model_validate(cases)

    @model_validator(mode="after")
    def validate_catalogue(self) -> Self:
        cases = (*self.serving_cases, *self.topology_cases)
        if not cases:
            raise ValueError("test catalogue cases must be nonempty")
        ids = tuple(case.id for case in cases)
        if None in ids:
            raise ValueError("test catalogue case IDs must be nonempty")
        if len(ids) != len(set(ids)):
            raise ValueError("test catalogue case IDs must be unique")
        return self
