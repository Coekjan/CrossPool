"""Offline workload normalization with domain-separated deterministic RNGs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Annotated, Self, cast

from pydantic import Field, FiniteFloat, TypeAdapter, model_validator

from xbench.harness.serving.case import (
    BenchCase,
    BenchTarget,
    BenchValue,
    ClientTarget,
    JsonlPrompts,
    LogNormalOutputTokens,
    OutputTokenCount,
    OwnedBenchCase,
    PoissonArrivals,
    RandomPrompts,
    TokenCount,
    TokenRange,
)
from xkit.case import CaseId
from xpool.config import XpoolConfig
from xpool.integrations.sglang.devkit.requests import LocalModelMetadata
from xpool.model import ModelId


class PromptValue(BenchValue):
    prompt_id: str = Field(min_length=1)
    text: str | None = None
    input_ids: tuple[int, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        if (self.text is None) == (self.input_ids is None):
            raise ValueError("a prompt requires exactly one of text or input_ids")
        if self.input_ids is not None and any(token < 0 for token in self.input_ids):
            raise ValueError("prompt input_ids must be nonnegative integers")
        return self


class ResolvedPrompt(PromptValue):
    model_id: ModelId
    text_tokens: int | None = Field(default=None, ge=0, description="Local serving-tokenizer length for text input.")

    @property
    def input_tokens(self) -> int | None:
        """Exact prepared input length; unknown for un-tokenized client text."""
        return len(self.input_ids) if self.input_ids is not None else self.text_tokens


class ScheduledRequest(BenchValue):
    """One replay request whose arrival is relative to the measurement origin."""

    request_id: str = Field(min_length=1)
    model_id: ModelId
    arrival_seconds: FiniteFloat = Field(ge=0)
    prompt_id: str = Field(min_length=1)
    max_new_tokens: int = Field(gt=0)


class PreparedWorkload(BenchValue):
    """Finite replay authority reused unchanged across case repetitions.

    Content and arrival hashes cover normalized records, not a seed alone.
    Warmup has separate requests/RNGs and does not consume measured traffic.
    Configuration and local metadata belong to preparation, not replay content.
    """

    case_id: CaseId
    model_ids: tuple[ModelId, ...]
    model_contexts: dict[ModelId, Annotated[int, Field(gt=0, strict=True)]]
    prompts: tuple[ResolvedPrompt, ...]
    requests: tuple[ScheduledRequest, ...]
    warmup: tuple[ScheduledRequest, ...]
    arrival_horizon_seconds: FiniteFloat = Field(ge=0)
    bucket_seconds: FiniteFloat = Field(gt=0)
    seeds: dict[str, int]
    prompt_sha256: str
    trace_sha256: str
    warmup_sha256: str

    @classmethod
    def load(cls, directory: Path, *, case: BenchCase) -> PreparedWorkload:
        """Rehydrate resolved JSONL inputs; never reopen models or regenerate traffic."""
        metadata = TypeAdapter(dict[str, object]).validate_json((directory / "workload.json").read_bytes())
        fields = set(cls.model_fields) - {"case_id", "model_ids", "prompts", "requests", "warmup"}
        if set(metadata) - fields - {"inputs"}:
            raise ValueError("workload checkpoint contains unsupported metadata fields")
        inputs = TypeAdapter(dict[str, str]).validate_python(metadata.pop("inputs", None))
        if set(inputs) != {"prompts", "requests", "warmup"}:
            raise ValueError("workload inputs must reference prompts, requests and warmup")
        for field, value_type in (
            ("prompts", ResolvedPrompt),
            ("requests", ScheduledRequest),
            ("warmup", ScheduledRequest),
        ):
            path = (directory / inputs[field]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError(f"workload input escapes its case: {inputs[field]}")
            metadata[field] = read_jsonl(path, value_type)
        metadata.update(case_id=case.id, model_ids=tuple(target.model_id for target in case.targets))
        return cls.model_validate(metadata)

    @model_validator(mode="after")
    def validate_replay(self) -> Self:
        if set(self.model_contexts) - set(self.model_ids):
            raise ValueError("prepared model context references an unknown target")
        prompt_keys = {(prompt.model_id, prompt.prompt_id) for prompt in self.prompts}
        ids = tuple(request.request_id for request in self.requests)
        if len(ids) != len(set(ids)) or len(prompt_keys) != len(self.prompts):
            raise ValueError("prepared request and target-scoped prompt IDs must be unique")
        if any((request.model_id, request.prompt_id) not in prompt_keys for request in (*self.requests, *self.warmup)):
            raise ValueError("prepared request prompt reference is unresolved")
        if any(request.model_id not in self.model_ids for request in (*self.requests, *self.warmup)):
            raise ValueError("prepared request target is unknown")
        arrivals = tuple(request.arrival_seconds for request in self.requests)
        if arrivals != tuple(sorted(arrivals)) or any(value > self.arrival_horizon_seconds for value in arrivals):
            raise ValueError("prepared requests must be chronological within their arrival horizon")
        if self.prompt_sha256 != content_digest(self.prompts):
            raise ValueError("prepared prompt content digest does not match")
        if self.trace_sha256 != content_digest(self.requests):
            raise ValueError("prepared trace content digest does not match")
        if self.warmup_sha256 != content_digest(self.warmup):
            raise ValueError("prepared warmup content digest does not match")
        return self


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def content_digest(values: Sequence[BenchValue]) -> str:
    raw = [value.model_dump(mode="json", exclude_none=True) for value in values]
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_jsonl[V: BenchValue](path: Path, value_type: type[V]) -> tuple[V, ...]:
    values = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            try:
                values.append(value_type.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid {value_type.__name__} at {path}:{number}: {error}") from error
    return tuple(values)


def sample_tokens(
    count: TokenCount,
    rng: random.Random,
    *,
    minimum: int = 1,
    maximum: int | None,
) -> int:
    """Condition generated counts on a legal interval; never rewrite fixed values."""
    if isinstance(count, int):
        if count < minimum or (maximum is not None and count > maximum):
            raise ValueError(f"token count {count} is outside the legal request interval [{minimum}, {maximum}]")
        return count
    if isinstance(count, TokenRange):
        lower = max(count.min, minimum)
        upper = count.max if maximum is None else min(count.max, maximum)
        if upper < lower:
            raise ValueError(f"token range [{count.min}, {count.max}] has no legal request interval")
        return lower if lower == upper else rng.randint(lower, upper)
    if maximum is None:
        raise ValueError("lognormal token sampling requires a finite legal upper bound")
    if maximum < minimum:
        raise ValueError("lognormal token policy has no legal request interval")
    if maximum == minimum:
        return minimum
    median = minimum + (maximum - minimum) * count.median_fraction
    normal = NormalDist(mu=math.log(median), sigma=count.sigma)
    lower = normal.cdf(math.log(minimum - 0.5))
    upper = normal.cdf(math.log(maximum + 0.5))
    if upper <= lower:
        raise ValueError("lognormal token policy has no representable probability in its legal interval")
    # Keep inverse-CDF probabilities open despite floating-point rounding.
    probability = max(math.nextafter(0.0, 1.0), min(math.nextafter(upper, lower), rng.uniform(lower, upper)))
    return math.floor(math.exp(normal.inv_cdf(probability)) + 0.5)


def prepare_workload(case: BenchCase, *, config: XpoolConfig | None = None) -> PreparedWorkload:
    """Resolve datasets and local metadata before timed execution, with no downloads.

    Relative file paths are already bound by BenchCatalog. Every target owns its
    prompt identity and independent content, arrival, selection, output and
    warmup RNGs. File traces keep source-row ordering for arrival ties.
    """

    if isinstance(case, OwnedBenchCase) and config is None:
        raise ValueError("owned workload preparation requires its effective configuration")

    seeds: dict[str, int] = {}

    def rng(model_id: ModelId, purpose: str) -> random.Random:
        key = f"{model_id}:{purpose}"
        seed = int.from_bytes(hashlib.sha256(json.dumps([case.seed, str(model_id), purpose]).encode()).digest(), "big")
        seeds[key] = seed
        return random.Random(seed)

    metadata: dict[ModelId, LocalModelMetadata] = {}
    sources: dict[ModelId, tuple[PromptValue, ...]] = {}
    prompt_index: dict[ModelId, dict[str, PromptValue]] = {}
    for target in case.targets:
        if isinstance(target.prompts, JsonlPrompts):
            values = read_jsonl(target.prompts.path, PromptValue)
            ids = [value.prompt_id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{target.model_id}: prompt IDs must be unique within the target")
            if not values:
                raise ValueError(f"{target.model_id}: prompt source is empty")
            sources[target.model_id] = values
            prompt_index[target.model_id] = {value.prompt_id: value for value in values}
        metadata_path = (
            target.model_metadata_path
            if isinstance(target, ClientTarget)
            else config.model_path_of(target.model_id)
            if config is not None
            else None
        )
        if metadata_path is not None:
            metadata[target.model_id] = LocalModelMetadata.from_checkpoint(
                metadata_path,
                text_prompts={
                    prompt.prompt_id: prompt.text
                    for prompt in sources.get(target.model_id, ())
                    if prompt.text is not None
                },
            )
        if isinstance(target.prompts, RandomPrompts) and not metadata[target.model_id].admissible_ids:
            raise ValueError(f"{target.model_id}: random prompts require nonempty admissible local tokenizer IDs")
        for value in sources.get(target.model_id, ()):
            validate_prompt(value, metadata.get(target.model_id))

    resolved: dict[tuple[ModelId, str], ResolvedPrompt] = {}

    def prompt(
        target: BenchTarget,
        id: str,
        *,
        output_tokens: int,
        minimum_input: int = 1,
        warmup: bool = False,
    ) -> ResolvedPrompt:
        key = (target.model_id, id)
        if key in resolved:
            return resolved[key]
        local = metadata.get(target.model_id)
        if isinstance(target.prompts, RandomPrompts):
            # Random targets have local metadata by declaration and acquisition.
            local = metadata[target.model_id]
            content_rng = rng(target.model_id, f"{'warmup' if warmup else 'content'}:{id}")
            length = sample_tokens(
                target.prompts.input_tokens,
                content_rng,
                minimum=minimum_input,
                maximum=local.limits.input_budget(output_tokens),
            )
            value = PromptValue(
                prompt_id=id, input_ids=tuple(content_rng.choice(local.admissible_ids) for _ in range(length))
            )
            validate_prompt(value, metadata.get(target.model_id))
        else:
            by_id = prompt_index[target.model_id]
            if id not in by_id:
                raise ValueError(f"{target.model_id}: unresolved prompt reference {id!r}")
            value = by_id[id]
        result = ResolvedPrompt(
            model_id=target.model_id,
            text_tokens=local.text_lengths[id] if local is not None and value.text is not None else None,
            **value.model_dump(),
        )
        resolved[key] = result
        return result

    by_target = {target.model_id: target for target in case.targets}
    selection_rngs = {target.model_id: rng(target.model_id, "selection") for target in case.targets}
    output_rngs = {target.model_id: rng(target.model_id, "output") for target in case.targets}

    def request_budget(
        target: BenchTarget,
        prompt_id: str,
        policy: OutputTokenCount,
        output_rng: random.Random,
        *,
        warmup: bool = False,
        shared_output_bound: int = 0,
    ) -> int:
        local = metadata.get(target.model_id)
        limits = local.limits if local is not None else None
        ratio = policy.max_output_input_ratio if isinstance(policy, LogNormalOutputTokens) else None
        if isinstance(target.prompts, RandomPrompts):
            minimum_output = policy if isinstance(policy, int) else policy.min if isinstance(policy, TokenRange) else 1
            value = prompt(
                target,
                prompt_id,
                output_tokens=max(minimum_output, shared_output_bound),
                minimum_input=max(1, math.ceil(minimum_output / ratio)) if ratio is not None else 1,
                warmup=warmup,
            )
        else:
            value = prompt(target, prompt_id, output_tokens=0, warmup=warmup)
        maximum = None
        if limits is not None and value.input_tokens is not None:
            maximum = limits.output_budget(value.input_tokens)
            if ratio is not None:
                maximum = math.floor(min(maximum, ratio * value.input_tokens))
        output = sample_tokens(policy, output_rng, maximum=maximum)
        if limits is not None and value.input_tokens is not None:
            limits.validate_request(value.input_tokens, output)
        return output

    requests: list[ScheduledRequest] = []
    if isinstance(case.arrivals, PoissonArrivals):
        arrivals: list[tuple[float, ModelId, str]] = []
        for target in case.targets:
            rate = case.arrivals.rates[target.model_id]
            arrival_rng = rng(target.model_id, "arrivals")
            if rate == 0:
                continue
            arrival = arrival_rng.expovariate(rate)
            sequence = 0
            while arrival < case.arrivals.duration_seconds:
                arrivals.append((arrival, target.model_id, f"{target.model_id}-{sequence:08d}"))
                sequence += 1
                arrival += arrival_rng.expovariate(rate)
        # Stable sort preserves target declaration/local sequence for arrival ties.
        arrivals.sort(key=lambda value: value[0])
        for arrival, model_id, request_id in arrivals:
            target = by_target[model_id]
            prompt_id = (
                selection_rngs[model_id].choice(sources[model_id]).prompt_id
                if isinstance(target.prompts, JsonlPrompts)
                else request_id
            )
            requests.append(
                ScheduledRequest(
                    request_id=request_id,
                    model_id=model_id,
                    arrival_seconds=arrival,
                    prompt_id=prompt_id,
                    # Case validation requires a policy for every Poisson target.
                    max_new_tokens=request_budget(
                        target, prompt_id, cast(OutputTokenCount, target.output_tokens), output_rngs[model_id]
                    ),
                )
            )
        horizon = case.arrivals.duration_seconds
    else:
        requests = list(read_jsonl(case.arrivals.path, ScheduledRequest))
        ids = [request.request_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("trace request IDs must be unique")
        if any(request.model_id not in by_target for request in requests):
            raise ValueError("trace contains an unknown target ID")
        requests.sort(key=lambda request: request.arrival_seconds)
        last = requests[-1].arrival_seconds if requests else 0.0
        horizon = last if case.arrivals.duration_seconds is None else case.arrivals.duration_seconds
        if horizon < last:
            raise ValueError("trace arrival horizon must cover every request")
        shared_output_bounds: dict[tuple[ModelId, str], int] = {}
        for request in requests:
            key = (request.model_id, request.prompt_id)
            shared_output_bounds[key] = max(shared_output_bounds.get(key, 0), request.max_new_tokens)
        for request in requests:
            request_budget(
                by_target[request.model_id],
                request.prompt_id,
                request.max_new_tokens,
                output_rngs[request.model_id],
                shared_output_bound=shared_output_bounds[request.model_id, request.prompt_id],
            )

    warmup_requests: list[ScheduledRequest] = []
    for target in case.targets:
        warmup_rng = rng(target.model_id, "warmup")
        cap = (
            cast(OutputTokenCount, target.output_tokens)
            if isinstance(case.arrivals, PoissonArrivals)
            else next((request.max_new_tokens for request in requests if request.model_id == target.model_id), 1)
        )
        for index in range(case.warmup_requests_per_target):
            id = f"warmup-{target.model_id}-{index}"
            prompt_id = (
                warmup_rng.choice(sources[target.model_id]).prompt_id
                if isinstance(target.prompts, JsonlPrompts)
                else id
            )
            warmup_requests.append(
                ScheduledRequest(
                    request_id=id,
                    model_id=target.model_id,
                    arrival_seconds=0.0,
                    prompt_id=prompt_id,
                    max_new_tokens=request_budget(target, prompt_id, cap, warmup_rng, warmup=True),
                )
            )
    prompts = tuple(resolved.values())
    return PreparedWorkload(
        case_id=case.id,
        model_ids=tuple(by_target),
        model_contexts={model_id: local.limits.context_length for model_id, local in metadata.items()},
        prompts=prompts,
        requests=tuple(requests),
        warmup=tuple(warmup_requests),
        arrival_horizon_seconds=horizon,
        bucket_seconds=case.bucket_seconds,
        seeds=seeds,
        prompt_sha256=content_digest(prompts),
        trace_sha256=content_digest(requests),
        warmup_sha256=content_digest(warmup_requests),
    )


def validate_prompt(prompt: PromptValue, metadata: LocalModelMetadata | None) -> None:
    if metadata is None or prompt.input_ids is None:
        return
    if any(id >= metadata.vocab_size for id in prompt.input_ids):
        raise ValueError(f"prompt {prompt.prompt_id!r} contains IDs outside the local model vocabulary")
    if len(prompt.input_ids) >= metadata.limits.context_length:
        raise ValueError(f"prompt {prompt.prompt_id!r} exceeds the known model input limit")
