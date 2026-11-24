"""Native media qualification shared by the two Gemma 4 model suites."""

from __future__ import annotations

import base64
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from pydantic import JsonValue, TypeAdapter

from xkit.network import TcpPortSpace
from xkit.process import OwnedProcessGroup
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.reference.ffn_child import SGLANG_PLUGIN_SENTINEL
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import SglangInferenceRecord, SglangProbeServer, SglangServerResult
from xtest.harness.support.environment import qualification_environment
from xtest.harness.support.native.observer import assert_fabric_observer_snapshots, assert_transport_observer_snapshots
from xtest.harness.support.sglang.graph import assert_run_graph_evidence

MEDIA = Path(__file__).with_name("gemma4_media")
MEDIA_PAIRS = (("red.png", "blue.png"), ("low.wav", "high.wav"), ("red.mp4", "blue.mp4"))
RESPONSE_TOKENS = 4


@dataclass(frozen=True, slots=True)
class MediaResponse:
    tokens: tuple[int, ...]
    first_logprob: float
    started_ns: int
    finished_ns: int


def run_media_qualification(case: E2eServingCase, base_config: ResolvedConfig, workdir: Path) -> None:
    """Compare native media requests and retain serving and FFN replay evidence."""

    (model_id,) = case.models
    graph = SglangGraphMode.DECODE_FULL.settings()
    observed: dict[str, MediaResponse] = {}

    def workload(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
        (server,) = servers
        with httpx.Client(base_url=server.url(), timeout=120) as client:
            info = client.get("/server_info").raise_for_status().json()
            config = info["cuda_graph_config"]
            settings = TypeAdapter(SglangGraphSettings).validate_python(
                {"decode_backend": config["decode"]["backend"], "prefill_backend": config["prefill"]["backend"]}
            )
            observed.update(run_media_requests(client, case, workdir / "crosspool-media"))
        first_name = MEDIA_PAIRS[0][0]
        first = observed[first_name]
        return (
            SglangServerResult(
                model_id=model_id,
                resolved_graph_settings=settings,
                request_id=first_name,
                input_ids=(),
                output_ids=first.tokens,
                cached_tokens=None,
                started_ns=first.started_ns,
                finished_ns=first.finished_ns,
                logprobs_path=None,
            ),
        )

    run = run_probe(case, base_config=base_config, graph_settings=graph, workdir=workdir, workload=workload)
    (template,) = run.results
    for name, response in observed.items():
        result = replace(
            template,
            request_id=name,
            output_ids=response.tokens,
            started_ns=response.started_ns,
            finished_ns=response.finished_ns,
        )
        assert_run_graph_evidence(replace(run, results=(result,)))
    assert_transport_observer_snapshots(run.observer_outdir, expected_count=case.atnagent_count, site="atnagent")
    assert_fabric_observer_snapshots(
        run.observer_outdir,
        atnagent_count=case.atnagent_count,
        ffnagent_count=case.ffnagent_count,
    )

    reference = run_native_media_reference(case, base_config, workdir)
    for names in MEDIA_PAIRS:
        for name in names:
            assert observed[name].tokens[0] == reference[name].tokens[0], name
        first, second = (reference[name] for name in names)
        if first.tokens[0] == second.tokens[0]:
            # A media change can affect confidence while preserving the greedy answer.
            native_delta = first.first_logprob - second.first_logprob
            observed_delta = observed[names[0]].first_logprob - observed[names[1]].first_logprob
            assert abs(native_delta) > 1e-4 and native_delta * observed_delta > 0, names


def run_native_media_reference(
    case: E2eServingCase, base_config: ResolvedConfig, workdir: Path
) -> dict[str, MediaResponse]:
    """Run the same pinned SGLang wrapper directly on one assigned device."""

    (model_id,) = case.models
    checkpoint = base_config.config.model_path_of(model_id)
    deadline = time.monotonic() + 600
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1", dp_size=1, port_space=TcpPortSpace.local(), deadline=deadline
    )
    try:
        endpoint = lease.family
        command = [
            "sglang",
            "serve",
            "--model-path",
            str(checkpoint),
            "--host",
            endpoint.host,
            "--port",
            str(endpoint.http_port),
            "--nccl-port",
            str(endpoint.nccl_port),
            "--device",
            "cuda",
            "--base-gpu-id",
            "0",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--attention-backend",
            "triton",
            "--chunked-prefill-size",
            "-1",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-backend-decode",
            "full",
            "--context-length",
            "4096",
            "--mem-fraction-static",
            "0.75",
        ]
        if str(model_id) == "google/gemma-4-12B-it":
            # The PP1 BF16 native reference needs the missing SGLang 0.5.20
            # tied-head flag in the spawned scheduler as well as the CLI parent.
            launcher = workdir / "native-unified.py"
            launcher.write_text(
                "from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration\n"
                "Gemma4UnifiedForConditionalGeneration.lm_head_is_tied = True\n"
                "if __name__ == '__main__':\n"
                "    from sglang.cli.main import main\n"
                "    main()\n",
                encoding="utf-8",
            )
            command[:1] = [sys.executable, str(launcher)]
        environment = {**qualification_environment(), "SGLANG_PLUGINS": SGLANG_PLUGIN_SENTINEL}
        lease.release_tcp_for_spawn()
        owner = OwnedProcessGroup.spawn_logged(
            "native Gemma 4", command, cwd=workdir, env=environment, log_path=workdir / "native-sglang.log"
        )
        try:
            with httpx.Client(base_url=f"http://{endpoint.host}:{endpoint.http_port}", timeout=120) as client:
                while time.monotonic() < deadline and owner.process.poll() is None:
                    with suppress(httpx.RequestError):
                        if client.get("/health", timeout=2).is_success:
                            return run_media_requests(client, case, workdir / "native-media")
                    time.sleep(0.2)
                raise RuntimeError(f"native Gemma 4 failed to become healthy:\n{owner.tail()}")
        finally:
            try:
                owner.terminate()
            finally:
                owner.close()
    finally:
        lease.close()


def run_media_requests(client: httpx.Client, case: E2eServingCase, evidence_dir: Path) -> dict[str, MediaResponse]:
    evidence_dir.mkdir()
    (model_id,) = case.models
    results = {}
    for pair in MEDIA_PAIRS:
        for name in pair:
            encoded = base64.b64encode((MEDIA / name).read_bytes()).decode("ascii")
            suffix = Path(name).suffix
            if suffix == ".wav":
                part: dict[str, JsonValue] = {
                    "type": "input_audio",
                    "input_audio": {"data": encoded, "format": "wav"},
                }
                prompt = "Is the tone high pitched or low pitched? Answer in one word."
            else:
                field, media_type = ("image_url", "image/png") if suffix == ".png" else ("video_url", "video/mp4")
                part = {"type": field, field: {"url": f"data:{media_type};base64,{encoded}"}}
                prompt = "What is the dominant color? Answer in one word."
            request: dict[str, JsonValue] = {
                "messages": [{"role": "user", "content": [part, {"type": "text", "text": prompt}]}],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_tokens": RESPONSE_TOKENS,
                "min_tokens": RESPONSE_TOKENS,
                "ignore_eos": True,
                "return_token_ids": True,
                "logprobs": True,
                "stream": False,
            }
            started_ns = time.monotonic_ns()
            response = client.post("/v1/chat/completions", json=request)
            finished_ns = time.monotonic_ns()
            SglangInferenceRecord(
                model_id=model_id,
                url=str(response.url),
                started_ns=started_ns,
                finished_ns=finished_ns,
                request=request,
                response_status_code=response.status_code,
                response_content_type=response.headers.get("content-type"),
                response=response.json() if "json" in response.headers.get("content-type", "") else response.text,
            ).write(evidence_dir / f"{name}.json")
            response.raise_for_status()
            (choice,) = response.json()["choices"]
            tokens = tuple(choice["response_token_ids"])
            assert len(tokens) == RESPONSE_TOKENS
            assert choice["message"]["content"].strip()
            results[name] = MediaResponse(
                tokens=tokens,
                first_logprob=float(choice["logprobs"]["content"][0]["logprob"]),
                started_ns=started_ns,
                finished_ns=finished_ns,
            )
    return results
