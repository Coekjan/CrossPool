"""Test-only deterministic requests and observer evidence over shared serving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Barrier

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError
from safetensors import SafetensorError, safe_open

from xkit.serving.sglang.graph import SglangGraphSettings
from xkit.serving.sglang.server import SglangServerProcess
from xpool.model import ModelId

PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
NEW_TOKENS = 8
HTTP_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class SglangServerResult:
    """Resolved graph behavior and output evidence from one server."""

    model_id: ModelId
    resolved_graph_settings: SglangGraphSettings
    output_ids: tuple[int, ...]
    prefill_logits_path: Path | None


@dataclass(frozen=True, slots=True)
class SglangInferenceRecord:
    """Exact public HTTP request and response retained as E2E evidence."""

    model_id: ModelId
    url: str
    request: dict[str, JsonValue]
    response_status_code: int
    response_content_type: str | None
    response: JsonValue

    def write(self, path: Path) -> None:
        """Exclusively write one versionless inference record."""

        with path.open("xb") as output:
            output.write(TypeAdapter(type(self)).dump_json(self, indent=2))


@dataclass(slots=True)
class SglangProbeServer:
    """Test-owned requests and observer evidence over a shared serving process."""

    process: SglangServerProcess
    inference_path: Path
    prefill_logit_outdir: Path | None = None

    def url(self) -> str:
        return self.process.url()

    def result(self, request_barrier: Barrier | None = None) -> SglangServerResult:
        """Read graph settings and execute one deterministic request.

        Args:
            request_barrier: Optional synchronization point reached immediately
                before sending the request, used to overlap multi-model serving.

        Raises:
            RuntimeError: If server-info Graph settings or returned token IDs
                are invalid.
            BrokenBarrierError: If peer requests do not reach the barrier before
                the HTTP timeout.
        """

        request_id = f"xpool-serving-graph-{self.process.model.model_id}"
        request: dict[str, JsonValue] = {
            "rid": request_id,
            "text": PROMPT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": NEW_TOKENS,
                "min_new_tokens": NEW_TOKENS,
                "ignore_eos": True,
            },
            "stream": False,
        }
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
            response = client.post("/generate", json=request)
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
            ).write(self.inference_path)
            response.raise_for_status()
        output_ids = payload.get("output_ids") if isinstance(payload, dict) else None
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != NEW_TOKENS
            or not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in output_ids)
        ):
            raise RuntimeError(f"{self.process.owner.name} returned invalid output_ids: {output_ids!r}")
        return SglangServerResult(
            model_id=self.process.model.model_id,
            resolved_graph_settings=settings,
            output_ids=tuple(output_ids),
            prefill_logits_path=self.read_prefill_logits_path(request_id),
        )

    def read_prefill_logits_path(self, request_id: str) -> Path | None:
        """Return one uniquely matching raw observer file, or no artifact."""

        if self.prefill_logit_outdir is None:
            return None
        matches: list[Path] = []
        for path in sorted(self.prefill_logit_outdir.glob("xpool.prefill-logits.*.safetensors")):
            try:
                with safe_open(path, framework="pt", device="cpu") as file:
                    metadata = file.metadata()
            except (OSError, SafetensorError):
                continue
            if metadata == {"rid": request_id}:
                matches.append(path)
        return matches[0] if len(matches) == 1 else None
