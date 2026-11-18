"""Public SGLang requests and normalized prediction-distribution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from time import monotonic_ns
from typing import Annotated

import httpx
import torch
from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from safetensors.torch import save_file

from xkit.serving.sglang.graph import SglangGraphSettings
from xkit.serving.sglang.server import SglangServerProcess
from xpool.model import ModelId

PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
NEW_TOKENS = 8
HTTP_TIMEOUT_SECONDS = 5 * 60.0


@dataclass(frozen=True, slots=True)
class SglangServerResult:
    """One request and CPU FP32 logprobs ordered by vocabulary ID.

    Row i conditions on ``(input_ids + output_ids)[:prefix_lengths[i]]``.
    Non-numerical requests have no prediction rows or logprob file.
    """

    model_id: ModelId
    resolved_graph_settings: SglangGraphSettings
    request_id: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    cached_tokens: int | None
    inference_path: Path
    logprobs_path: Path | None

    @property
    def prefix_lengths(self) -> tuple[int, ...]:
        """Conditioning lengths for the delivered generation distributions."""

        if self.logprobs_path is None:
            return ()
        return tuple(range(len(self.input_ids), len(self.input_ids) + len(self.output_ids)))


@dataclass(frozen=True, slots=True)
class SglangInferenceRecord:
    """Exact HTTP evidence and its host-monotonic request window, in nanoseconds."""

    model_id: ModelId
    url: str
    request: dict[str, JsonValue]
    response_status_code: int
    response_content_type: str | None
    response: JsonValue
    started_ns: int
    finished_ns: int

    def write(self, path: Path) -> None:
        """Exclusively retain the raw response before status or shape validation."""

        with path.open("xb") as output:
            output.write(TypeAdapter(type(self)).dump_json(self))


@dataclass(slots=True)
class SglangProbeServer:
    """Test-owned request policy over the shared serving-process owner."""

    process: SglangServerProcess

    def url(self) -> str:
        return self.process.url()

    def generate(
        self,
        request: dict[str, JsonValue],
        *,
        record_path: Path,
        request_barrier: Barrier | None = None,
    ) -> SglangServerResult:
        """Run public generation and retain evidence.

        Numerical callers request all vocabulary IDs and exact prompt IDs.
        The response boundary checks token IDs and complete vocabulary coverage,
        then stores dense CPU FP32 distributions.
        A barrier synchronizes concurrent requests immediately before HTTP POST.
        """

        record_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(base_url=self.url(), timeout=HTTP_TIMEOUT_SECONDS) as client:
            server_info = client.get("/server_info")
            server_info.raise_for_status()
            info = server_info.json()
            try:
                graph = info["cuda_graph_config"]
                settings = TypeAdapter(SglangGraphSettings).validate_python(
                    {
                        "decode_backend": graph["decode"]["backend"],
                        "prefill_backend": graph["prefill"]["backend"],
                    }
                )
            except (KeyError, TypeError, ValidationError) as error:
                raise RuntimeError("SGLang /server_info returned invalid CUDA Graph settings") from error
            if request_barrier is not None:
                request_barrier.wait(timeout=HTTP_TIMEOUT_SECONDS)
            started_ns = monotonic_ns()
            response = client.post("/generate", json=request)
            finished_ns = monotonic_ns()
            try:
                payload: JsonValue = response.json()
            except ValueError:
                payload = response.text
            SglangInferenceRecord(
                model_id=self.process.model.model_id,
                url=f"{self.url()}/generate",
                request=request,
                response_status_code=response.status_code,
                response_content_type=response.headers.get("content-type"),
                response=payload,
                started_ns=started_ns,
                finished_ns=finished_ns,
            ).write(record_path)
            response.raise_for_status()

        if not isinstance(payload, dict):
            raise RuntimeError("SGLang /generate returned a non-object response")
        token_ids = TypeAdapter(tuple[Annotated[int, Field(ge=0, strict=True)], ...])
        input_ids = token_ids.validate_python(payload["prompt_token_ids"])
        output_ids = token_ids.validate_python(payload["output_ids"])
        parameters = request["sampling_params"]
        if not isinstance(parameters, dict) or len(output_ids) != parameters["max_new_tokens"]:
            raise RuntimeError("SGLang /generate returned an unexpected output token count")
        request_id = TypeAdapter(str).validate_python(request["rid"], strict=True)
        meta = payload["meta_info"]
        if not isinstance(meta, dict):
            raise RuntimeError("SGLang /generate returned invalid meta_info")
        cached_tokens = TypeAdapter(Annotated[int, Field(ge=0, strict=True)] | None).validate_python(
            meta.get("cached_tokens")
        )
        logprobs_path = None
        if request.get("return_logprob"):
            vocabulary = token_ids.validate_python(request["token_ids_logprob"])
            if not vocabulary or vocabulary != tuple(range(len(vocabulary))):
                raise ValueError("numerical requests must select the complete vocabulary in token ID order")
            rows = TypeAdapter(
                list[
                    list[tuple[Annotated[float, Field(strict=True)], Annotated[int, Field(strict=True)], str | None]]
                    | None
                ]
            ).validate_python(meta["output_token_ids_logprobs"])
            if not output_ids or len(rows) != len(output_ids):
                raise RuntimeError("SGLang logprob rows do not cover the requested prediction positions")
            logprobs = torch.empty((len(rows), len(vocabulary)), dtype=torch.float32)
            for index, row in enumerate(rows):
                if row is None:
                    raise RuntimeError("SGLang returned a missing output-logprob row")
                ids = [entry[1] for entry in row]
                if sorted(ids) != list(vocabulary):
                    raise RuntimeError("SGLang logprobs do not cover the complete vocabulary exactly once")
                logprobs[index, ids] = torch.tensor([entry[0] for entry in row], dtype=torch.float32)
            if not torch.isfinite(logprobs).all():
                raise RuntimeError("SGLang logprobs must be finite FP32 values")
            logprobs_path = record_path.with_suffix(".logprobs.safetensors")
            save_file({"logprobs": logprobs}, logprobs_path)
        return SglangServerResult(
            model_id=self.process.model.model_id,
            resolved_graph_settings=settings,
            request_id=request_id,
            input_ids=input_ids,
            output_ids=output_ids,
            cached_tokens=cached_tokens,
            inference_path=record_path,
            logprobs_path=logprobs_path,
        )


def generation_request(
    request_id: str,
    *,
    input_ids: tuple[int, ...] | None = None,
    vocab_size: int | None = None,
    max_new_tokens: int = NEW_TOKENS,
) -> dict[str, JsonValue]:
    """Build the shared greedy workload, preserving EOS probabilities and cache hits."""

    request: dict[str, JsonValue] = {
        "rid": request_id,
        "return_prompt_token_ids": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True},
        "stream": False,
    }
    if input_ids is None:
        request["text"] = PROMPT
    else:
        request["input_ids"] = list(input_ids)
    if vocab_size is not None:
        request.update(return_logprob=True, token_ids_logprob=list(range(vocab_size)), logprob_start_len=-1)
    return request
