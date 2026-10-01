"""Cross-Instance elastic KV Cache serving evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from threading import Barrier

import httpx

import xtest
from xkit.serving.sglang.graph import SglangGraphSettings
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving import qualification
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import NEW_TOKENS, SglangProbeServer, SglangServerResult
from xtest.harness.support.native.observer import (
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)
from xtest.harness.support.sglang.graph import assert_run_graph_evidence

pytest_plugins = ("xtest.harness.support.config",)

HTTP_TIMEOUT_SECONDS = 5 * 60.0
SUSTAINED_REQUEST_COUNT = 4


def generate(
    server: SglangProbeServer,
    request_id: str,
    token_count: int,
    *,
    input_token_id: int = 1,
) -> tuple[int, tuple[int, ...]]:
    """Run one deterministic request and return its cache hit and output IDs."""

    response = httpx.post(
        f"{server.url()}/generate",
        json={
            "rid": request_id,
            "input_ids": [input_token_id] * token_count,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": NEW_TOKENS,
                "min_new_tokens": NEW_TOKENS,
                "ignore_eos": True,
            },
            "stream": False,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    output_ids = payload.get("output_ids")
    meta_info = payload.get("meta_info")
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != NEW_TOKENS
        or not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in output_ids)
        or not isinstance(meta_info, dict)
    ):
        raise AssertionError(f"SGLang returned invalid elastic KV response: {payload!r}")
    cached_tokens = meta_info.get("cached_tokens")
    if not isinstance(cached_tokens, int):
        raise AssertionError(f"SGLang returned invalid cached_tokens: {cached_tokens!r}")
    return cached_tokens, tuple(output_ids)


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

    def requests(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
        request_barrier = Barrier(len(servers))
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            results = tuple(executor.map(SglangProbeServer.result, servers, repeat(request_barrier)))

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
        primary_initial, primary_expected_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-primary-fill",
            workload.prefix_tokens,
        )
        competing_initial, competing_expected_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-competing-fill",
            workload.prefix_tokens,
            input_token_id=2,
        )
        primary_hit, primary_hit_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-primary-hit",
            workload.prefix_tokens,
        )
        competing_hit, competing_hit_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-competing-hit",
            workload.prefix_tokens,
            input_token_id=2,
        )
        generate(pressure_server, "xpool-elastic-kv-pressure", pressure_tokens)
        primary_after_pressure, primary_after_pressure_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-primary-after-pressure",
            workload.prefix_tokens,
        )
        competing_after_pressure, competing_after_pressure_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-competing-after-pressure",
            workload.prefix_tokens,
            input_token_id=2,
        )

        if primary_after_pressure < primary_hit:
            victim_before_pressure = primary_hit
            victim_after_pressure = primary_after_pressure
            restored_expected_output_ids = primary_expected_output_ids
            restored_token_id = 1
        elif competing_after_pressure < competing_hit:
            victim_before_pressure = competing_hit
            victim_after_pressure = competing_after_pressure
            restored_expected_output_ids = competing_expected_output_ids
            restored_token_id = 2
        else:
            raise AssertionError("peer pressure did not reclaim either confirmed prefix")
        restored_fill, restored_fill_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-restored-fill",
            workload.prefix_tokens,
            input_token_id=restored_token_id,
        )
        restored, restored_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-restored-hit",
            workload.prefix_tokens,
            input_token_id=restored_token_id,
        )

        def sustain(alias: str, server: SglangProbeServer, token_count: int) -> int:
            for index in range(SUSTAINED_REQUEST_COUNT):
                generate(server, f"xpool-elastic-kv-{alias}-{index}", token_count)
            return SUSTAINED_REQUEST_COUNT

        with ThreadPoolExecutor(max_workers=2) as pressure_executor:
            futures = (
                pressure_executor.submit(sustain, "prefix-pressure", prefix_server, workload.prefix_tokens),
                pressure_executor.submit(sustain, "peer-pressure", pressure_server, pressure_tokens),
            )
            completed = tuple(future.result() for future in futures)

        print(
            "XPOOL_ELASTIC_KV_CACHE="
            + json.dumps(
                {
                    "competing_after_pressure": competing_after_pressure,
                    "competing_hit": competing_hit,
                    "competing_initial": competing_initial,
                    "pressure_tokens": pressure_tokens,
                    "primary_after_pressure": primary_after_pressure,
                    "primary_hit": primary_hit,
                    "primary_initial": primary_initial,
                    "restored": restored,
                    "restored_fill": restored_fill,
                    "victim_after_pressure": victim_after_pressure,
                    "victim_before_pressure": victim_before_pressure,
                },
                sort_keys=True,
            )
        )
        assert primary_initial < primary_hit
        assert competing_initial < competing_hit
        assert restored > 0
        assert restored >= restored_fill
        assert completed == (SUSTAINED_REQUEST_COUNT, SUSTAINED_REQUEST_COUNT)
        assert primary_hit_output_ids == primary_after_pressure_output_ids == primary_expected_output_ids
        assert competing_hit_output_ids == competing_after_pressure_output_ids == competing_expected_output_ids
        assert restored_fill_output_ids == restored_output_ids == restored_expected_output_ids
        return results

    run = run_probe(
        case,
        base_config=e2e_base_config,
        graph_settings=SglangGraphSettings(decode_backend="full", prefill_backend="breakable"),
        workdir=tmp_path,
        workload=requests,
    )
    assert_transport_observer_snapshots(
        run.observer_outdir,
        expected_count=case.atnagent_count * len(case.models),
        site="atnagent",
    )
    assert_fabric_observer_snapshots(
        run.observer_outdir,
        atnagent_count=case.atnagent_count,
        ffnagent_count=case.ffnagent_count,
        expected_topologies=tuple(
            (case.atnagent_count // model.atn_dp_size, model.atn_dp_size)
            for model_id in case.models
            for model in (case.deployment_config.model_by_id[model_id],)
        ),
    )
    assert_two_model_executor_overlap(run.observer_outdir)
    assert_run_graph_evidence(run)
