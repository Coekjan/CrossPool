"""Strict declarative schema for the SGLang E2E workload manifest."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tests.harness.sglang.graph import SglangGraphMode
from xpool.config import LoopbackSite

E2E_MANIFEST_PATH = Path(__file__).with_name("manifest.toml")


class E2eModel(BaseModel):
    """One named model available to declarative E2E cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    max_total_tokens: int = Field(gt=0)


class E2eModelPlacement(BaseModel):
    """One model's placement in a shared AtnAgent topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    atn_tp_size: int = Field(gt=0)
    atn_dp_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        """Reject the deferred combined attention TP-by-DP topology."""

        if self.atn_tp_size > 1 and self.atn_dp_size > 1:
            raise ValueError("E2E does not support combined attention TP-by-DP")
        return self


class E2eServingCase(BaseModel):
    """One independently supervised model-serving workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    models: tuple[E2eModelPlacement, ...]
    ffnagent_count: int = Field(gt=0)
    executor_count: int = Field(gt=0)
    graph_modes: tuple[SglangGraphMode, ...]
    estimated_duration_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    transport_trace_capacity: int = Field(gt=0)
    fabric_trace_capacity: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        """Require one unambiguous topology and complete execution capacity."""

        if not self.models:
            raise ValueError("E2E serving case models must be nonempty")
        aliases = tuple(placement.model for placement in self.models)
        if len(aliases) != len(set(aliases)):
            raise ValueError("E2E serving case model aliases must be unique")
        topologies = {(placement.atn_tp_size, placement.atn_dp_size) for placement in self.models}
        if len(topologies) != 1:
            raise ValueError("every E2E model must use the same attention topology")
        if not self.graph_modes or len(self.graph_modes) != len(set(self.graph_modes)):
            raise ValueError("E2E serving case graph_modes must be nonempty and unique")
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


class E2eLoopbackServingCase(E2eServingCase):
    """One explicit debug-loopback site sweep."""

    sites: tuple[LoopbackSite, ...]

    @model_validator(mode="after")
    def validate_sites(self) -> Self:
        """Require a nonempty site set without duplicate tasks."""

        if not self.sites or len(self.sites) != len(set(self.sites)):
            raise ValueError("E2E loopback serving sites must be nonempty and unique")
        return self


class E2eManifest(BaseModel):
    """Complete model catalog and E2E workload matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    models: tuple[E2eModel, ...]
    model_serving_cases: tuple[E2eServingCase, ...]
    loopback_serving_cases: tuple[E2eLoopbackServingCase, ...]

    @classmethod
    def load(cls, path: Path) -> E2eManifest:
        """Load and strictly validate one TOML manifest."""

        with path.open("rb") as manifest_file:
            raw = tomllib.load(manifest_file)
        models = raw.get("models")
        if isinstance(models, dict):
            raw["models"] = [
                {"alias": alias, **value} if isinstance(value, dict) else {"alias": alias, "value": value}
                for alias, value in models.items()
            ]
        return cls.model_validate(raw)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require unique identities and resolvable case model aliases."""

        if not self.models:
            raise ValueError("E2E manifest models must be nonempty")
        aliases = tuple(model.alias for model in self.models)
        model_ids = tuple(model.model_id for model in self.models)
        if len(aliases) != len(set(aliases)):
            raise ValueError("E2E manifest model aliases must be unique")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("E2E manifest model IDs must be unique")
        cases = (*self.model_serving_cases, *self.loopback_serving_cases)
        if not self.model_serving_cases or not self.loopback_serving_cases:
            raise ValueError("E2E manifest serving case tables must be nonempty")
        case_ids = tuple(case.id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("E2E manifest case IDs must be unique")
        known_aliases = set(aliases)
        unknown_aliases = sorted(
            {placement.model for case in cases for placement in case.models if placement.model not in known_aliases}
        )
        if unknown_aliases:
            raise ValueError(f"E2E manifest references unknown model aliases: {unknown_aliases}")
        return self

    def model(self, alias: str) -> E2eModel:
        """Return one catalog model by its validated alias."""

        return next(model for model in self.models if model.alias == alias)
