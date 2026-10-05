"""Strict benchmark declarations with declaring-file-owned path resolution."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, model_validator

from xkit.config import assemble_config
from xkit.deployment import resolve_deployment_path
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xpool.config import XpoolConfig
from xpool.model import ModelId


class BenchValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TokenRange(BenchValue):
    """Inclusive token-count bounds used for independent random sampling."""

    min: int = Field(gt=0)
    max: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.max < self.min:
            raise ValueError("token range max must be at least min")
        return self


type TokenCount = Annotated[int, Field(gt=0, strict=True)] | TokenRange


class RandomPrompts(BenchValue):
    kind: Literal["random"]
    input_tokens: TokenCount


class JsonlPrompts(BenchValue):
    kind: Literal["jsonl"]
    path: Path


type PromptSource = Annotated[RandomPrompts | JsonlPrompts, Field(discriminator="kind")]


class PoissonArrivals(BenchValue):
    kind: Literal["poisson"]
    duration_seconds: FiniteFloat = Field(gt=0, description="Exclusive end of the planned arrival window.")
    rates: dict[ModelId, Annotated[FiniteFloat, Field(ge=0, strict=True)]] = Field(
        description="Requests per second for each Model ID; zero disables its measured arrivals."
    )

    @model_validator(mode="after")
    def validate_rates(self) -> Self:
        if not any(rate > 0 for rate in self.rates.values()):
            raise ValueError("Poisson arrivals require at least one positive rate")
        return self


class TraceArrivals(BenchValue):
    kind: Literal["jsonl"]
    path: Path
    duration_seconds: FiniteFloat | None = Field(
        default=None, ge=0, description="Arrival horizon; None uses the last trace arrival, or zero for an empty trace."
    )


type ArrivalSource = Annotated[PoissonArrivals | TraceArrivals, Field(discriminator="kind")]


class NativeSampling(BenchValue):
    temperature: FiniteFloat = Field(default=0.0, ge=0)
    stream_interval: int = Field(default=1, gt=0)
    ignore_eos: bool | None = Field(
        default=None, description="None ignores EOS for random prompts and honors EOS for file-backed prompts."
    )


class BenchTarget(BenchValue):
    model_id: ModelId
    api: Literal["sglang", "openai"] = "sglang"
    prompts: PromptSource
    output_tokens: TokenCount | None = Field(
        default=None, description="Poisson request output limit; file traces supply max_new_tokens per request."
    )
    sampling: NativeSampling = Field(default_factory=NativeSampling)

    def ignores_eos(self) -> bool:
        return (
            self.sampling.ignore_eos
            if self.sampling.ignore_eos is not None
            else isinstance(self.prompts, RandomPrompts)
        )


class OwnedTarget(BenchTarget):
    graph_mode: SglangGraphMode


class ClientTarget(BenchTarget):
    base_url: str
    model_metadata_path: Path | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.query or url.fragment:
            raise ValueError("serving base_url must be an HTTP(S) URL without query or fragment")
        url.port  # Validate the external port syntax at the declaration boundary.
        return self


class CaseSettings(BenchValue):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = Field(min_length=1)
    module: str = Field(min_length=1, description="Source module relative to the catalogue's sibling suites directory.")
    arrivals: ArrivalSource
    seed: int = 0
    max_inflight: int = Field(
        default=128, gt=0, description="Active HTTP request limit; excess arrivals queue in FIFO order."
    )
    startup_timeout_seconds: FiniteFloat = Field(
        default=1800.0, gt=0, description="Complete owned startup budget, including daemon, Agents and serving health."
    )
    request_timeout_seconds: FiniteFloat = Field(
        default=600.0,
        gt=0,
        description="Absolute timeout from HTTP dispatch, excluding time spent in the admission queue.",
    )
    warmup_requests_per_target: int = Field(default=1, ge=0)
    repetitions: int = Field(default=1, gt=0)
    bucket_seconds: FiniteFloat = Field(default=1.0, gt=0)

    def validate_targets(self, targets: tuple[BenchTarget, ...]) -> None:
        ids = tuple(target.model_id for target in targets)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("benchmark targets must be nonempty and have unique Model IDs")
        if isinstance(self.arrivals, PoissonArrivals):
            if set(self.arrivals.rates) != set(ids):
                raise ValueError("Poisson rates must cover Model IDs exactly")
            if any(target.output_tokens is None for target in targets):
                raise ValueError("Poisson targets require output_tokens")
        elif any(target.output_tokens is not None for target in targets):
            raise ValueError("file traces own each request's max_new_tokens; target output_tokens must be absent")


class OwnedBenchCase(CaseSettings):
    mode: Literal["owned"]
    deployment: Path
    runtime_config: Path | None = None
    targets: tuple[OwnedTarget, ...]

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        self.validate_targets(self.targets)
        return self


class ClientBenchCase(CaseSettings):
    mode: Literal["client"]
    targets: tuple[ClientTarget, ...]
    serving_metadata_path: Path | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        self.validate_targets(self.targets)
        for target in self.targets:
            if isinstance(target.prompts, RandomPrompts) and target.model_metadata_path is None:
                raise ValueError(f"{target.model_id}: random prompts require explicit local model_metadata_path")
        return self

    def load_serving_metadata(self) -> ServingMetadata | None:
        """Validate explicit external conditions without contacting the server."""
        if self.serving_metadata_path is None:
            return None
        contents = self.serving_metadata_path.read_bytes()
        metadata = ServingMetadata.model_validate_json(contents)
        if "schema_version" not in metadata.model_fields_set:
            raise ValueError("external serving metadata requires schema_version=1")
        if set(metadata.target_device_uuids or {}) - {target.model_id for target in self.targets}:
            raise ValueError("serving metadata references an unknown benchmark target")
        return metadata


type BenchCase = Annotated[OwnedBenchCase | ClientBenchCase, Field(discriminator="mode")]


class ServingDevice(BenchValue):
    uuid: str = Field(min_length=1)
    name: str | None = None
    total_memory_bytes: int | None = Field(default=None, gt=0)
    pci_bus_id: str | None = None
    cpu_affinity: str | None = None
    numa_affinity: str | None = None


class ServingDeviceLink(BenchValue):
    source_uuid: str = Field(min_length=1)
    destination_uuid: str = Field(min_length=1)
    link: str = Field(min_length=1)


class ServingMetadata(BenchValue):
    """Serving conditions; provenance is supplied by the tool, not the input.

    Missing fields remain unknown. Link tokens are topology observations, not
    measured bandwidth. Client files may reference only their provided devices;
    preparation additionally checks their target references against the case.
    """

    schema_version: int = Field(default=1, ge=1, le=1, strict=True, exclude=True)
    devices: tuple[ServingDevice, ...] | None = None
    links: tuple[ServingDeviceLink, ...] | None = None
    target_device_uuids: dict[ModelId, tuple[str, ...]] | None = None
    role_device_uuids: dict[Literal["atn", "ffn"], tuple[str, ...]] | None = None
    packages: dict[str, str] | None = None
    driver_version: str | None = None
    cuda_build_version: str | None = None
    cuda_build_source: str | None = None

    @model_validator(mode="after")
    def validate_device_references(self) -> Self:
        uuids = {device.uuid for device in self.devices or ()}
        if len(uuids) != len(self.devices or ()):
            raise ValueError("serving metadata device UUIDs must be unique")
        groups = (
            *(self.target_device_uuids or {}).values(),
            *(self.role_device_uuids or {}).values(),
            *((link.source_uuid, link.destination_uuid) for link in self.links or ()),
        )
        if any(set(group) - uuids for group in groups):
            raise ValueError("serving metadata references an unknown device UUID")
        return self


class CatalogDeclaration(BenchValue):
    serving_cases: dict[str, BenchCase] = Field(min_length=1)


class ResolvedDeployment(BenchValue):
    """Effective owned configuration and provenance, retained once per case."""

    runtime_config: Path | None
    cwd: Path
    effective: dict[str, JsonValue]
    sources: JsonValue


def resolve_deployment(case: OwnedBenchCase) -> XpoolConfig:
    """Prepare owned launch policy before resource acquisition or workload timing."""
    config = assemble_config(
        tuple(target.model_id for target in case.targets),
        deployment=case.deployment,
        runtime_config=case.runtime_config,
        env=os.environ,
    )
    ServingLaunch(
        config,
        {},
        Path.cwd(),
        tuple(SglangLaunchModel(target.model_id, target.graph_mode) for target in case.targets),
    )
    return config


@dataclass(frozen=True, slots=True)
class BenchCatalog:
    """Ordered validated cases whose relative paths belong to the declaring file."""

    path: Path
    cases: tuple[BenchCase, ...]

    @classmethod
    def load(cls, path: Path) -> BenchCatalog:
        path = path.expanduser().resolve()
        with path.open("rb") as source:
            raw = tomllib.load(source)
        cases = raw.get("serving_cases", {})
        if not isinstance(cases, dict):
            raise ValueError("serving_cases must contain named case tables")
        for id, case in cases.items():
            if not isinstance(case, dict) or "id" in case:
                raise ValueError("the catalogue table name is the sole case identity")
            case["id"] = id
        # JSON validation preserves strict scalars while accepting serialized paths/enums.
        declaration = CatalogDeclaration.model_validate_json(json.dumps(raw, allow_nan=False))

        def resolve(value: Path) -> Path:
            return (path.parent / value.expanduser()).resolve()

        cases: list[BenchCase] = []
        for id, case in declaration.serving_cases.items():
            source_case = raw["serving_cases"][id]
            targets = []
            for target in case.targets:
                updates: dict[str, object] = {}
                if isinstance(target.prompts, JsonlPrompts):
                    updates["prompts"] = target.prompts.model_copy(update={"path": resolve(target.prompts.path)})
                if isinstance(target, ClientTarget) and target.model_metadata_path is not None:
                    updates["model_metadata_path"] = resolve(target.model_metadata_path)
                targets.append(target.model_copy(update=updates))
            case_updates: dict[str, object] = {"targets": tuple(targets)}
            if isinstance(case.arrivals, TraceArrivals):
                case_updates["arrivals"] = case.arrivals.model_copy(update={"path": resolve(case.arrivals.path)})
            if isinstance(case, OwnedBenchCase):
                case_updates["deployment"] = resolve_deployment_path(
                    path, tuple(target.model_id for target in case.targets), source_case["deployment"]
                )
                if case.runtime_config is not None:
                    case_updates["runtime_config"] = resolve(case.runtime_config)
            elif case.serving_metadata_path is not None:
                case_updates["serving_metadata_path"] = resolve(case.serving_metadata_path)
            cases.append(case.model_copy(update=case_updates))
        return cls(path, tuple(cases))

    def select(self, ids: tuple[str, ...]) -> tuple[BenchCase, ...]:
        if not ids:
            return self.cases
        if len(ids) != len(set(ids)):
            raise ValueError("selected case IDs must be unique")
        by_id = {case.id: case for case in self.cases}
        unknown = set(ids) - by_id.keys()
        if unknown:
            raise ValueError(f"unknown benchmark cases: {sorted(unknown)}")
        return tuple(by_id[id] for id in ids)
