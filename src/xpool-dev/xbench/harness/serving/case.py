"""Strict benchmark declarations with declaring-file-owned path resolution."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue, field_validator, model_validator

from xkit.case import CaseId, Catalog
from xkit.config import DeploymentConfig, assemble_config
from xkit.deployment import resolve_deployment_path
from xkit.requirements import ResourceRequirements
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xpool.config import XpoolConfig
from xpool.model import ModelId


class BenchValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TokenRange(BenchValue):
    """Inclusive uniform token-count bounds conditioned on legal request limits."""

    min: int = Field(gt=0)
    max: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.max < self.min:
            raise ValueError("token range max must be at least min")
        return self


class LogNormalTokens(BenchValue):
    """Legal-interval-relative median and natural-log standard deviation."""

    kind: Literal["lognormal"]
    median_fraction: FiniteFloat = Field(gt=0, le=1)
    sigma: FiniteFloat = Field(gt=0)


class LogNormalOutputTokens(LogNormalTokens):
    """Generated output policy with an optional ceiling relative to input length."""

    max_output_input_ratio: FiniteFloat | None = Field(default=None, gt=0)


type TokenCount = Annotated[int, Field(gt=0, strict=True)] | TokenRange | LogNormalTokens
type OutputTokenCount = Annotated[int, Field(gt=0, strict=True)] | TokenRange | LogNormalOutputTokens


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


class Sampling(BenchValue):
    temperature: FiniteFloat = Field(default=0.0, ge=0)
    ignore_eos: bool | None = Field(
        default=None, description="None ignores EOS for random prompts and honors EOS for file-backed prompts."
    )


class BenchTarget(BenchValue):
    model_id: ModelId
    prompts: PromptSource
    output_tokens: OutputTokenCount | None = Field(
        default=None, description="Poisson request output limit; file traces supply max_new_tokens per request."
    )
    sampling: Sampling = Field(default_factory=Sampling)

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
    served_model_name: str | None = Field(default=None, min_length=1)

    @field_validator("base_url")
    @classmethod
    def validate_endpoint(cls, base_url: str) -> str:
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.path not in {"", "/"}
            or "?" in base_url
            or "#" in base_url
        ):
            raise ValueError("serving base_url must be an HTTP(S) root without credentials, query or fragment")
        url.port  # Validate the external port syntax at the declaration boundary.
        return base_url


class ServingEndpoint(BenchValue):
    """Server URL and API model name for one benchmark target."""

    base_url: str
    served_model_name: str


class CaseSettings(BenchValue):
    id: CaseId
    description: str = Field(min_length=1)
    targets: tuple[BenchTarget, ...]
    arrivals: ArrivalSource
    seed: int = 0
    max_inflight: int = Field(
        default=128, gt=0, description="Active HTTP request limit; excess arrivals queue in FIFO order."
    )
    startup_timeout_seconds: FiniteFloat = Field(
        default=1800.0, gt=0, description="Complete managed serving startup budget in seconds."
    )
    request_timeout_seconds: FiniteFloat | None = Field(
        default=None,
        gt=0,
        description="Absolute HTTP timeout excluding client queue wait; None waits until completion.",
    )
    warmup_requests_per_target: int = Field(default=1, ge=0)
    bucket_seconds: FiniteFloat = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        ids = tuple(target.model_id for target in self.targets)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("benchmark targets must be nonempty and have unique Model IDs")
        if isinstance(self.arrivals, PoissonArrivals):
            if set(self.arrivals.rates) != set(ids):
                raise ValueError("Poisson rates must cover Model IDs exactly")
            if any(target.output_tokens is None for target in self.targets):
                raise ValueError("Poisson targets require output_tokens")
        elif any(target.output_tokens is not None for target in self.targets):
            raise ValueError("file traces own each request's max_new_tokens; target output_tokens must be absent")
        return self


class OwnedBenchCase(CaseSettings):
    mode: Literal["owned"]
    deployment: Path
    runtime_config: Path | None = None
    targets: tuple[OwnedTarget, ...]


class ClientBenchCase(CaseSettings):
    mode: Literal["client"]
    targets: tuple[ClientTarget, ...]


class HostedBenchCase(CaseSettings):
    mode: Literal["hosted"]
    script: Path
    launch_settings: dict[str, JsonValue] | None = None


class HostedRequirements(BenchValue):
    """Resource count and model groups sharing individual HTTP listeners."""

    device_count: int = Field(ge=0)
    listeners: tuple[Annotated[tuple[ModelId, ...], Field(min_length=1)], ...] = Field(min_length=1)

    def validate_targets(self, targets: tuple[BenchTarget, ...]) -> None:
        expected = {target.model_id for target in targets}
        listed = tuple(model_id for group in self.listeners for model_id in group)
        if len(listed) != len(expected) or set(listed) != expected:
            raise ValueError("hosted listeners must cover each target Model ID exactly once")


class HostedLaunchDeclaration(BenchValue):
    """A hosted script's digest and validated resource requirements."""

    script_sha256: str
    requirements: HostedRequirements


type BenchCase = Annotated[OwnedBenchCase | ClientBenchCase | HostedBenchCase, Field(discriminator="mode")]


class CatalogDeclaration(BenchValue):
    serving_cases: dict[CaseId, BenchCase] = Field(min_length=1)


class ResolvedDeployment(BenchValue):
    """Effective owned configuration and provenance, retained once per case."""

    runtime_config: Path | None
    cwd: Path
    effective: dict[str, JsonValue]
    sources: JsonValue


def owned_requirements(case: OwnedBenchCase) -> ResourceRequirements:
    model_ids = tuple(target.model_id for target in case.targets)
    scene = DeploymentConfig.from_file(case.deployment, model_ids=model_ids)
    return ResourceRequirements(len(scene.atn.devices) + len(scene.ffn.devices), True, model_ids)


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
        tuple(SglangLaunchModel(target.model_id, target.graph_mode.settings()) for target in case.targets),
    )
    return config


class BenchCatalog(Catalog):
    """Ordered validated cases whose relative paths belong to the declaring file."""

    cases: tuple[BenchCase, ...]

    @classmethod
    def from_file(cls, path: Path) -> Self:
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
            source_case = raw["serving_cases"][str(id)]
            targets = []
            for target in case.targets:
                updates: dict[str, object] = {}
                if isinstance(target.prompts, JsonlPrompts):
                    updates["prompts"] = target.prompts.model_copy(update={"path": resolve(target.prompts.path)})
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
            elif isinstance(case, HostedBenchCase):
                case_updates["script"] = resolve(case.script)
            cases.append(case.model_copy(update=case_updates))
        return cls(path=path, cases=tuple(cases))

    def select(self, prefixes: Sequence[str]) -> tuple[BenchCase, ...]:
        """Select unique full identities from prefixes, preserving invocation order."""
        if not prefixes:
            return self.cases
        by_id = {case.id: case for case in self.cases}
        identities = tuple(CaseId.resolve(prefix, by_id) for prefix in prefixes)
        if len(identities) != len(set(identities)):
            raise ValueError("selected case IDs must be unique")
        return tuple(by_id[identity] for identity in identities)
