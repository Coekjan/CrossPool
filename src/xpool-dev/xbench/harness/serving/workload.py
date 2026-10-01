"""Offline workload normalization with domain-separated deterministic RNGs."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Self, cast

from pydantic import Field, FiniteFloat, TypeAdapter, model_validator
from transformers import AutoConfig, AutoTokenizer

from xbench.harness.serving.case import (
    BenchCase,
    BenchTarget,
    BenchValue,
    ClientTarget,
    JsonlPrompts,
    OwnedBenchCase,
    PoissonArrivals,
    RandomPrompts,
    TokenCount,
)
from xpool.config import XpoolConfig
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


class ScheduledRequest(BenchValue):
    """One replay request whose arrival is relative to the measurement origin."""

    request_id: str = Field(min_length=1)
    model_id: ModelId
    arrival_seconds: FiniteFloat = Field(ge=0)
    prompt_id: str = Field(min_length=1)
    max_new_tokens: int = Field(gt=0)


class LocalMetadata(BenchValue):
    vocab_size: int = Field(gt=0)
    max_input_tokens: int | None = Field(gt=0)


class PreparedWorkload(BenchValue):
    """Finite replay authority reused unchanged across case repetitions.

    Content and arrival hashes cover normalized records, not a seed alone.
    Warmup has separate requests/RNGs and does not consume measured traffic.
    Configuration and local metadata belong to preparation, not replay content.
    """

    case_id: str
    model_ids: tuple[ModelId, ...]
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


def sample_tokens(count: TokenCount, rng: random.Random) -> int:
    return count if isinstance(count, int) else rng.randint(count.min, count.max)


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

    if isinstance(case.arrivals, PoissonArrivals):
        arrivals = case.arrivals
        requests: list[ScheduledRequest] = []
        for target in case.targets:
            rate = arrivals.rates[target.model_id]
            arrival_rng = rng(target.model_id, "arrivals")
            output_rng = rng(target.model_id, "output")
            if rate == 0:
                continue
            arrival = arrival_rng.expovariate(rate)
            sequence = 0
            while arrival < arrivals.duration_seconds:
                request_id = f"{target.model_id}-{sequence:08d}"
                requests.append(
                    ScheduledRequest(
                        request_id=request_id,
                        model_id=target.model_id,
                        arrival_seconds=arrival,
                        prompt_id=request_id,
                        # Case validation requires a policy for every Poisson target.
                        max_new_tokens=sample_tokens(cast(TokenCount, target.output_tokens), output_rng),
                    )
                )
                sequence += 1
                arrival += arrival_rng.expovariate(rate)
        # Python's stable sort preserves target declaration/local sequence for ties.
        requests.sort(key=lambda request: request.arrival_seconds)
        horizon = arrivals.duration_seconds
    else:
        requests = list(read_jsonl(case.arrivals.path, ScheduledRequest))
        ids = [request.request_id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("trace request IDs must be unique")
        targets = {target.model_id for target in case.targets}
        if any(request.model_id not in targets for request in requests):
            raise ValueError("trace contains an unknown target ID")
        requests.sort(key=lambda request: request.arrival_seconds)
        last = requests[-1].arrival_seconds if requests else 0.0
        horizon = last if case.arrivals.duration_seconds is None else case.arrivals.duration_seconds
        if horizon < last:
            raise ValueError("trace arrival horizon must cover every request")

    metadata: dict[ModelId, LocalMetadata] = {}
    admissible: dict[ModelId, tuple[int, ...]] = {}
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
            metadata[target.model_id], admissible[target.model_id] = load_local_metadata(
                metadata_path, prompts=sources.get(target.model_id, ())
            )
        if isinstance(target.prompts, RandomPrompts) and not admissible.get(target.model_id):
            raise ValueError(f"{target.model_id}: random prompts require nonempty admissible local tokenizer IDs")
        for value in sources.get(target.model_id, ()):
            validate_prompt(value, metadata.get(target.model_id))

    resolved: dict[tuple[ModelId, str], ResolvedPrompt] = {}

    def prompt(target: BenchTarget, id: str, *, warmup: bool = False) -> ResolvedPrompt:
        key = (target.model_id, id)
        if key in resolved:
            return resolved[key]
        if isinstance(target.prompts, RandomPrompts):
            content_rng = rng(target.model_id, f"{'warmup' if warmup else 'content'}:{id}")
            length = sample_tokens(target.prompts.input_tokens, content_rng)
            value = PromptValue(
                prompt_id=id, input_ids=tuple(content_rng.choice(admissible[target.model_id]) for _ in range(length))
            )
            validate_prompt(value, metadata.get(target.model_id))
        else:
            by_id = prompt_index[target.model_id]
            if id not in by_id:
                raise ValueError(f"{target.model_id}: unresolved prompt reference {id!r}")
            value = by_id[id]
        result = ResolvedPrompt(model_id=target.model_id, **value.model_dump())
        resolved[key] = result
        return result

    by_target = {target.model_id: target for target in case.targets}
    selection_rngs = {target.model_id: rng(target.model_id, "selection") for target in case.targets}
    for index, request in enumerate(requests):
        target = by_target[request.model_id]
        if isinstance(case.arrivals, PoissonArrivals) and isinstance(target.prompts, JsonlPrompts):
            id = selection_rngs[target.model_id].choice(sources[target.model_id]).prompt_id
            request = request.model_copy(update={"prompt_id": id})
            requests[index] = request
        prompt(target, request.prompt_id)

    warmup_requests: list[ScheduledRequest] = []
    for target in case.targets:
        warmup_rng = rng(target.model_id, "warmup")
        cap = (
            cast(TokenCount, target.output_tokens)
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
            prompt(target, prompt_id, warmup=True)
            warmup_requests.append(
                ScheduledRequest(
                    request_id=id,
                    model_id=target.model_id,
                    arrival_seconds=0.0,
                    prompt_id=prompt_id,
                    max_new_tokens=sample_tokens(cap, warmup_rng),
                )
            )
    prompts = tuple(resolved.values())
    return PreparedWorkload(
        case_id=case.id,
        model_ids=tuple(by_target),
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


def load_local_metadata(path: Path, *, prompts: Sequence[PromptValue] = ()) -> tuple[LocalMetadata, tuple[int, ...]]:
    # Metadata acquisition is offline and deliberately excludes SGLang model discovery.
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer is None:
        raise ValueError(f"local metadata did not produce a tokenizer: {path}")
    text_config = config.get_text_config()
    vocab_size = getattr(text_config, "vocab_size", None)
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError(f"local model metadata has no positive vocab_size: {path}")
    special = set(tokenizer.all_special_ids)
    ids = tuple(sorted({id for id in tokenizer.get_vocab().values() if 0 <= id < vocab_size and id not in special}))
    max_input = getattr(text_config, "max_position_embeddings", None)
    if not isinstance(max_input, int) or isinstance(max_input, bool) or max_input <= 0:
        max_input = None
    for prompt in prompts:
        if prompt.text is not None and max_input is not None:
            tokens = tokenizer.encode(prompt.text)
            if len(tokens) > max_input:
                raise ValueError(f"prompt {prompt.prompt_id!r} exceeds the known model input limit")
            if any(token < 0 or token >= vocab_size for token in tokens):
                raise ValueError(f"prompt {prompt.prompt_id!r} contains IDs outside the local model vocabulary")
    return (
        LocalMetadata(
            vocab_size=vocab_size,
            max_input_tokens=max_input,
        ),
        ids,
    )


def validate_prompt(prompt: PromptValue, metadata: LocalMetadata | None) -> None:
    if metadata is None or prompt.input_ids is None:
        return
    if any(id >= metadata.vocab_size for id in prompt.input_ids):
        raise ValueError(f"prompt {prompt.prompt_id!r} contains IDs outside the local model vocabulary")
    if metadata.max_input_tokens is not None and len(prompt.input_ids) > metadata.max_input_tokens:
        raise ValueError(f"prompt {prompt.prompt_id!r} exceeds the known model input limit")
