"""Request-owned sampling records and development-only SGLang replay."""

from __future__ import annotations

import fcntl
import os
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter

from xkit.envs import DETERMINISTIC_MODE, DETERMINISTIC_PATH
from xpool.config import get_global_config
from xpool.model import ModelId

__all__ = ["DeterministicConfig", "SamplerReplayEvidence", "SamplingRecord", "install"]


@dataclass(frozen=True, slots=True)
class DeterministicConfig:
    """One launch's record/replay inputs; active modes require a file path."""

    mode: Literal["none", "record", "replay"] = "none"
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("none", "record", "replay"):
            raise ValueError(f"unsupported deterministic mode: {self.mode}")
        if self.mode != "none" and self.path is None:
            raise ValueError("record/replay requires XKIT_DETERMINISTIC_PATH")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> DeterministicConfig:
        """Read explicit process inputs, defaulting to inactive operation."""

        mode = TypeAdapter(Literal["none", "record", "replay"]).validate_python(
            DETERMINISTIC_MODE.get(env=environment) or "none", strict=True
        )
        value = DETERMINISTIC_PATH.get(env=environment)
        return cls(mode, Path(value).expanduser().resolve() if mode != "none" and value else None)

    def prepare(self) -> None:
        """Create one new output or validate a replay source before child startup.

        The shared serving owner calls this once, before its model processes.
        Existing recordings and failed attempts are never truncated.
        """

        if self.mode == "none":
            return
        path = cast(Path, self.path)
        if self.mode == "record":
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb"):
                pass
        else:
            for record in SamplingRecord.read(path):
                del record


@dataclass(frozen=True, slots=True)
class SamplingRecord:
    """One completed request; logprob fields retain SGLang's response format.

    request contains the public text/input_ids and requested sampling_params.
    prompt_tokens is the existing response count, not a tokenizer reconstruction.
    No scheduler-effective settings or arithmetic policy are inferred.
    """

    model_id: ModelId
    request_id: str
    request: dict[str, JsonValue]
    prompt_tokens: Annotated[int, Field(gt=0, strict=True)]
    output_ids: tuple[Annotated[int, Field(ge=0, strict=True)], ...]
    finish_reason: dict[str, JsonValue]
    logprobs: dict[str, JsonValue] | None

    def __post_init__(self) -> None:
        if not self.request_id or self.prompt_tokens <= 0 or not self.output_ids:
            raise ValueError("sampling records require a request ID, prompt and completed output")
        if any(type(token) is not int or token < 0 for token in self.output_ids):
            raise ValueError("recorded output IDs must be nonnegative integers")
        if not isinstance(self.request.get("sampling_params"), dict):
            raise ValueError("sampling records require public sampling_params")
        if ("text" in self.request) == ("input_ids" in self.request):
            raise ValueError("sampling records require exactly one public text or input_ids input")
        if "text" in self.request:
            TypeAdapter(str).validate_python(self.request["text"], strict=True)
        else:
            TypeAdapter(list[Annotated[int, Field(ge=0, strict=True)]]).validate_python(self.request["input_ids"])

    def append(self, path: Path) -> None:
        """Append one complete JSONL record under the shared local file lock.

        The launch owner has already created the file. Serialization follows
        the engine's HTTP handling of non-finite logprob values.
        """

        from sglang.srt.utils.json_response import dumps_json

        payload = dumps_json(TypeAdapter(SamplingRecord).dump_python(self, mode="json")) + b"\n"
        with path.open("r+b") as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            output.seek(0, os.SEEK_END)
            output.write(payload)
            output.flush()

    @classmethod
    def read(cls, path: Path) -> Iterator[SamplingRecord]:
        """Read validated complete records, rejecting duplicate request keys."""

        keys: set[tuple[ModelId, str]] = set()
        with path.open("rb") as source:
            for line in source:
                if not line.endswith(b"\n"):
                    raise ValueError(f"incomplete sampling record in {path}")
                record = TypeAdapter(cls).validate_json(line)
                key = record.model_id, record.request_id
                if key in keys:
                    raise ValueError(f"duplicate sampling record: {record.model_id}/{record.request_id}")
                keys.add(key)
                yield record


@dataclass(frozen=True, slots=True)
class SamplingSelection:
    """A controlled live sampler row, distinct from padded Graph capacity."""

    row: int
    record: SamplingRecord
    prefix_length: int
    forward_mode: str
    rank: int

    @property
    def token_id(self) -> int | None:
        """Recorded choice at this row's actual logical prefix."""

        step = self.prefix_length - self.record.prompt_tokens
        return self.record.output_ids[step] if 0 <= step < len(self.record.output_ids) else None


@dataclass(frozen=True, slots=True)
class SamplerReplayEvidence:
    """One rank's sampler choice; null prescribed_id marks passthrough work."""

    model_id: ModelId
    request_id: str
    rank: int
    prefix_length: int
    forward_mode: str
    original_id: int
    prescribed_id: int | None


def install() -> None:
    """Install process-local hooks only for explicit record/replay operation.

    Engine imports stay after activation so ordinary entry-point discovery has
    no sampling, arithmetic or file side effects.
    """

    settings = DeterministicConfig.from_environment(os.environ)
    if settings.mode == "none":
        return

    import fastapi
    import torch
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.logprob_processor import LogprobResult, OutputLogprobProcessor
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.managers.io_struct import (
        BatchTokenizedGenerateReqInput,
        GenerateReqInput,
        TokenizedGenerateReqInput,
    )
    from sglang.srt.managers.tokenizer_manager import ReqState, ServerStatus, TokenizerManager
    from sglang.srt.managers.utils import is_health_check_generate_req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.utils.json_response import dumps_json

    path = cast(Path, settings.path)
    records = (
        {(record.model_id, record.request_id): replace(record, logprobs=None) for record in SamplingRecord.read(path)}
        if settings.mode == "replay"
        else {}
    )
    selections: ContextVar[tuple[SamplingSelection, ...]] = ContextVar("xkit_sampler_selections", default=())

    def public_request(obj: GenerateReqInput) -> dict[str, JsonValue]:
        key = "text" if obj.text is not None else "input_ids"
        return TypeAdapter(dict[str, JsonValue]).validate_python(
            {key: obj.text if obj.text is not None else obj.input_ids, "sampling_params": obj.sampling_params or {}}
        )

    def around_dispatch(
        original: Callable[[TokenizerManager, object], None], manager: TokenizerManager, obj: object
    ) -> None:
        # Admission precedes sampling; internal startup and health work is not a workload replay.
        pending = obj.batch if isinstance(obj, BatchTokenizedGenerateReqInput) else (obj,)
        for item in pending:
            if (
                not isinstance(item, TokenizedGenerateReqInput)
                or manager.server_status is ServerStatus.Starting
                or is_health_check_generate_req(item)
            ):
                continue
            state = manager.rid_to_state[cast(str, item.rid)]
            model_id = get_global_config().model_id_of_path(manager.model_config.model_path)
            record = records.get((model_id, item.rid))
            if record is None:
                raise ValueError(f"no sampling record for {model_id}/{item.rid}")
            if not isinstance(state.obj, GenerateReqInput) or public_request(state.obj) != record.request:
                raise ValueError(f"replay request differs from its sampling record: {item.rid}")
            if state.obj.return_sampling_mask:
                raise ValueError("sampler replay does not support return_sampling_mask")
        original(manager, obj)

    async def around_response(
        original: Callable[..., AsyncIterator[dict[str, object]]],
        manager: TokenizerManager,
        obj: GenerateReqInput,
        state: ReqState,
        request: fastapi.Request | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        # SGLang yields native tuples before its HTTP JSON serialization.
        ignored = (
            not isinstance(obj, GenerateReqInput)
            or manager.server_status is ServerStatus.Starting
            or is_health_check_generate_req(obj)
        )
        async for response in original(manager, obj, state, request):
            response_meta = TypeAdapter(dict[str, object]).validate_python(response["meta_info"], strict=True)
            if not ignored and response_meta["finish_reason"] is not None:
                reason = TypeAdapter(dict[str, JsonValue]).validate_python(response_meta["finish_reason"])
                if reason["type"] != "abort":
                    model_id = get_global_config().model_id_of_path(manager.model_config.model_path)
                    if settings.mode == "replay":
                        reference = records[model_id, cast(str, obj.rid)]
                        output_ids = TypeAdapter(tuple[Annotated[int, Field(ge=0, strict=True)], ...]).validate_python(
                            response["output_ids"]
                        )
                        prompt_tokens = TypeAdapter(int).validate_python(response_meta["prompt_tokens"], strict=True)
                        if output_ids != reference.output_ids or prompt_tokens != reference.prompt_tokens:
                            raise ValueError(f"incomplete or mismatched sampler replay: {obj.rid}")
                    else:
                        payload = TypeAdapter(dict[str, JsonValue]).validate_json(dumps_json(response))
                        meta = cast(dict[str, JsonValue], payload["meta_info"])
                        SamplingRecord(
                            model_id=model_id,
                            request_id=cast(str, obj.rid),
                            request=public_request(obj),
                            prompt_tokens=TypeAdapter(int).validate_python(meta["prompt_tokens"], strict=True),
                            output_ids=TypeAdapter(tuple[int, ...]).validate_python(payload["output_ids"]),
                            finish_reason=reason,
                            logprobs={key: value for key, value in meta.items() if "logprobs" in key}
                            if obj.return_logprob
                            else None,
                        ).append(path)
            yield response

    def around_sample(
        original: Callable[[ModelRunner, LogitsProcessorOutput, ForwardBatch], torch.Tensor],
        runner: ModelRunner,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        model_id = get_global_config().model_id_of_path(runner.model_config.model_path)
        matched = [
            (row, records[model_id, rid])
            for row, rid in enumerate(forward_batch.rids or ())
            if (model_id, rid) in records
        ]
        if not matched:
            return original(runner, logits_output, forward_batch)
        positions = (
            forward_batch.positions if forward_batch.forward_mode.is_decode() else forward_batch.seq_lens - 1
        ).tolist()
        active: list[SamplingSelection] = []
        for row, record in matched:
            prefix = positions[row] + 1
            choice = SamplingSelection(row, record, prefix, forward_batch.forward_mode.name.lower(), runner.ps.tp_rank)
            if choice.token_id is not None and choice.token_id >= runner.model_config.vocab_size:
                raise ValueError(f"recorded token outside model vocabulary: {choice.token_id}")
            active.append(choice)
        token = selections.set(tuple(active))
        try:
            return original(runner, logits_output, forward_batch)
        finally:
            selections.reset(token)

    def after_logprobs(
        result: LogprobResult,
        processor: OutputLogprobProcessor,
        logprobs: torch.Tensor,
        top_logprobs_nums: list[int],
        token_ids_logprobs: list[list[int]],
        batch_next_token_ids: torch.Tensor,
    ) -> None:
        del processor, top_logprobs_nums, token_ids_logprobs, batch_next_token_ids
        for choice in selections.get():
            if choice.token_id is None:
                continue
            if result.token_logprobs is None:
                raise RuntimeError("SGLang omitted requested selected-token logprobs")
            result.token_logprobs[choice.row] = logprobs[choice.row, choice.token_id]

    def after_sampler(
        next_token_ids: torch.Tensor,
        sampler: Sampler,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: list[int],
        token_ids_logprobs: list[list[int]],
        positions: torch.Tensor,
    ) -> None:
        del sampler, logits_output, sampling_info, return_logprob, top_logprobs_nums, token_ids_logprobs, positions
        observer = get_global_config().debug.graph_observer
        evidence: list[SamplerReplayEvidence] = []
        for choice in selections.get():
            if observer.enable:
                evidence.append(
                    SamplerReplayEvidence(
                        choice.record.model_id,
                        choice.record.request_id,
                        choice.rank,
                        choice.prefix_length,
                        choice.forward_mode,
                        int(next_token_ids[choice.row].item()),
                        choice.token_id,
                    )
                )
            if choice.token_id is not None:
                next_token_ids[choice.row] = choice.token_id
        if evidence:
            target = cast(Path, observer.outdir) / f"xkit.sampler-replay.{os.getpid()}.jsonl"
            with target.open("ab") as output:
                for value in evidence:
                    output.write(dumps_json(TypeAdapter(SamplerReplayEvidence).dump_python(value, mode="json")) + b"\n")

    HookRegistry.register(
        "sglang.srt.managers.tokenizer_manager.TokenizerManager._stream_one_response", around_response, HookType.AROUND
    )
    if settings.mode == "replay":
        HookRegistry.register(
            "sglang.srt.managers.tokenizer_manager.TokenizerManager._dispatch_to_scheduler",
            around_dispatch,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.model_executor.model_runner.ModelRunner.sample", around_sample, HookType.AROUND
        )
        HookRegistry.register(
            "sglang.srt.layers.logprob_processor.OutputLogprobProcessor.compute_logprobs",
            after_logprobs,
            HookType.AFTER,
        )
        HookRegistry.register("sglang.srt.layers.sampler.Sampler.forward", after_sampler, HookType.AFTER)
