"""Cross-Instance elastic KV Cache serving evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx
from pydantic import TypeAdapter

import xtest
from xkit.config import resolve_model_weights
from xkit.results import write_json
from xkit.serving.sglang.graph import SglangGraphSettings
from xpool.integrations.sglang.devkit.requests import LocalModelMetadata
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving import qualification
from xtest.harness.sglang.serving.alignment import (
    PREFILL_DISTRIBUTION_KL_LIMIT,
    DistributionComparison,
    compare_distributions,
)
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import NEW_TOKENS, SglangProbeServer, SglangServerResult, generation_request

pytest_plugins = ("xtest.harness.support.config",)

HTTP_TIMEOUT_SECONDS = 5 * 60.0
SUSTAINED_REQUEST_COUNT = 4


@xtest.parameterize("case")
@xtest.requirements(qualification.requirements_of)
def test_e2e_elastic_kv(
    case: E2eServingCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
) -> None:
    """Observe competing prefix hits, peer-pressure cache loss, and renewed hits."""

    workload = case.elastic_kv
    assert workload is not None
    vocabularies = {
        model_id: LocalModelMetadata.from_checkpoint(
            resolve_model_weights(e2e_base_config.config, model_id), text_prompts={}
        ).vocab_size
        for model_id in case.models
    }

    def generate(
        server: SglangProbeServer, request_id: str, token_count: int, *, input_token_id: int = 1, logprobs: bool = True
    ) -> SglangServerResult:
        return server.generate(
            generation_request(
                request_id,
                input_ids=(input_token_id,) * token_count,
                vocab_size=vocabularies[server.process.model.model_id] if logprobs else None,
                max_new_tokens=1 if logprobs else NEW_TOKENS,
            ),
            record_path=tmp_path / "requests" / f"{request_id}.json",
        )

    def requests(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
        request_barrier = Barrier(len(servers))

        def initial_request(server: SglangProbeServer) -> SglangServerResult:
            model_id = server.process.model.model_id
            return server.generate(
                generation_request(f"xpool-serving-{model_id}"),
                record_path=tmp_path / "requests" / f"{model_id.uri_encode()}.json",
                request_barrier=request_barrier,
            )

        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            results = tuple(executor.map(initial_request, servers))

        by_model_id = {server.process.model.model_id: server for server in servers}
        prefix_server = by_model_id[workload.prefix_model_id]
        pressure_server = by_model_id[workload.pressure_model_id]
        response = httpx.get(f"{pressure_server.url()}/server_info", timeout=HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        server_info = response.json()
        max_input_tokens = server_info.get("max_req_input_len") if isinstance(server_info, dict) else None
        if (
            not isinstance(max_input_tokens, int)
            or isinstance(max_input_tokens, bool)
            or max_input_tokens <= NEW_TOKENS
        ):
            raise AssertionError(f"SGLang returned invalid max_req_input_len: {max_input_tokens!r}")
        pressure_tokens = max_input_tokens - NEW_TOKENS
        primary_initial = generate(
            prefix_server,
            "xpool-elastic-kv-primary-fill",
            workload.prefix_tokens,
        )
        competing_initial = generate(
            prefix_server,
            "xpool-elastic-kv-competing-fill",
            workload.prefix_tokens,
            input_token_id=2,
        )
        primary_hit = generate(
            prefix_server,
            "xpool-elastic-kv-primary-hit",
            workload.prefix_tokens,
        )
        competing_hit = generate(
            prefix_server,
            "xpool-elastic-kv-competing-hit",
            workload.prefix_tokens,
            input_token_id=2,
        )
        generate(pressure_server, "xpool-elastic-kv-pressure", pressure_tokens, logprobs=False)
        primary_after_pressure = generate(
            prefix_server,
            "xpool-elastic-kv-primary-after-pressure",
            workload.prefix_tokens,
        )
        competing_after_pressure = generate(
            prefix_server,
            "xpool-elastic-kv-competing-after-pressure",
            workload.prefix_tokens,
            input_token_id=2,
        )

        cache_evidence = TypeAdapter(dict[str, int]).validate_python(
            {
                "primary_initial": primary_initial.cached_tokens,
                "primary_hit": primary_hit.cached_tokens,
                "primary_after_pressure": primary_after_pressure.cached_tokens,
                "competing_initial": competing_initial.cached_tokens,
                "competing_hit": competing_hit.cached_tokens,
                "competing_after_pressure": competing_after_pressure.cached_tokens,
            },
            strict=True,
        )
        if cache_evidence["primary_after_pressure"] < cache_evidence["primary_hit"]:
            victim_before_pressure = cache_evidence["primary_hit"]
            victim_after_pressure = cache_evidence["primary_after_pressure"]
            restored_reference = primary_initial
            restored_token_id = 1
        elif cache_evidence["competing_after_pressure"] < cache_evidence["competing_hit"]:
            victim_before_pressure = cache_evidence["competing_hit"]
            victim_after_pressure = cache_evidence["competing_after_pressure"]
            restored_reference = competing_initial
            restored_token_id = 2
        else:
            raise AssertionError("peer pressure did not reclaim either confirmed prefix")
        restored_fill = generate(
            prefix_server,
            "xpool-elastic-kv-restored-fill",
            workload.prefix_tokens,
            input_token_id=restored_token_id,
        )
        restored = generate(
            prefix_server,
            "xpool-elastic-kv-restored-hit",
            workload.prefix_tokens,
            input_token_id=restored_token_id,
        )

        def sustain(alias: str, server: SglangProbeServer, token_count: int) -> int:
            for index in range(SUSTAINED_REQUEST_COUNT):
                generate(server, f"xpool-elastic-kv-{alias}-{index}", token_count, logprobs=False)
            return SUSTAINED_REQUEST_COUNT

        with ThreadPoolExecutor(max_workers=2) as pressure_executor:
            futures = (
                pressure_executor.submit(sustain, "prefix-pressure", prefix_server, workload.prefix_tokens),
                pressure_executor.submit(sustain, "peer-pressure", pressure_server, pressure_tokens),
            )
            completed = tuple(future.result() for future in futures)

        restored_counts = TypeAdapter(dict[str, int]).validate_python(
            {"restored": restored.cached_tokens, "restored_fill": restored_fill.cached_tokens}, strict=True
        )
        cache_evidence.update(
            **restored_counts,
            pressure_tokens=pressure_tokens,
            victim_after_pressure=victim_after_pressure,
            victim_before_pressure=victim_before_pressure,
        )
        write_json(tmp_path / "elastic-kv.json", cache_evidence)
        print("XPOOL_ELASTIC_KV_CACHE=" + json.dumps(cache_evidence, sort_keys=True))
        assert cache_evidence["primary_initial"] < cache_evidence["primary_hit"]
        assert cache_evidence["competing_initial"] < cache_evidence["competing_hit"]
        assert cache_evidence["restored"] > 0
        assert cache_evidence["restored"] >= cache_evidence["restored_fill"]
        assert completed == (SUSTAINED_REQUEST_COUNT, SUSTAINED_REQUEST_COUNT)
        comparisons = [
            compare_distributions(reference, actual, prefix_lengths=(len(reference.input_ids),))
            for reference, actual in (
                (primary_initial, primary_hit),
                (primary_initial, primary_after_pressure),
                (competing_initial, competing_hit),
                (competing_initial, competing_after_pressure),
                (restored_reference, restored_fill),
                (restored_reference, restored),
            )
        ]
        write_json(
            tmp_path / "elastic-kv.numerical.json",
            TypeAdapter(list[DistributionComparison]).dump_python(comparisons, mode="json"),
        )
        for comparison in comparisons:
            comparison.assert_within(PREFILL_DISTRIBUTION_KL_LIMIT)
        return results

    run = run_probe(
        case,
        base_config=e2e_base_config,
        graph_settings=SglangGraphSettings(decode_backend="full", prefill_backend=workload.prefill_backend),
        workdir=tmp_path,
        workload=requests,
    )
    qualification.assert_serving_evidence(case, run)
