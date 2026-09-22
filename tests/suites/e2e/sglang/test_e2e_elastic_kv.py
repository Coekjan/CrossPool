"""Cross-Instance elastic KV Cache serving evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from threading import Barrier

import httpx
import pytest
from _pytest.mark.structures import ParameterSet

from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eManifest, E2eServingCase
from tests.harness.sglang.serving.graph import SglangGraphSettings
from tests.harness.sglang.serving.probe import run_probe
from tests.harness.sglang.serving.server import NEW_TOKENS, SglangServerProcess, SglangServerResult
from tests.harness.support.native.observer import (
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)
from tests.harness.support.sglang.graph import assert_run_graph_evidence
from xpool.config import XpoolConfig

pytest_plugins = ("tests.harness.support.config",)

MANIFEST = E2eManifest.load(E2E_MANIFEST_PATH)
HTTP_TIMEOUT_SECONDS = 5 * 60.0
SUSTAINED_REQUEST_COUNT = 4


def case_parameter(case: E2eServingCase) -> ParameterSet:
    """Attach the resources required by one elastic KV serving case."""

    model_ids = tuple(placement.model_id for placement in case.models)
    marks = [
        pytest.mark.requires_cuda(min_devices=case.required_gpu_count),
        pytest.mark.requires_config,
        pytest.mark.requires_mps,
        pytest.mark.timeout(case.timeout_seconds),
        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
    ]
    marks.extend(pytest.mark.requires_model_weights(model_id) for model_id in model_ids)
    return pytest.param(case, id=case.id, marks=marks)


def generate(
    server: SglangServerProcess,
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


@pytest.mark.parametrize(
    "case",
    tuple(case_parameter(case) for case in MANIFEST.model_serving_cases if case.elastic_kv),
)
def test_e2e_elastic_kv(
    case: E2eServingCase,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
) -> None:
    """Observe competing prefix hits, peer-pressure cache loss, and renewed hits."""

    workload = case.elastic_kv
    assert workload is not None

    def requests(servers: list[SglangServerProcess]) -> tuple[SglangServerResult, ...]:
        request_barrier = Barrier(len(servers))
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            results = tuple(executor.map(SglangServerProcess.result, servers, repeat(request_barrier)))

        by_model_id = {server.model.model_id: server for server in servers}
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
            restored_hit = primary_hit
            restored_expected_output_ids = primary_expected_output_ids
            restored_token_id = 1
        elif competing_after_pressure < competing_hit:
            restored_hit = competing_hit
            restored_expected_output_ids = competing_expected_output_ids
            restored_token_id = 2
        else:
            raise AssertionError("peer pressure did not reclaim either confirmed prefix")
        _, restored_fill_output_ids = generate(
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

        def sustain(alias: str, server: SglangServerProcess, token_count: int) -> int:
            for index in range(SUSTAINED_REQUEST_COUNT):
                generate(server, f"xpool-elastic-kv-{alias}-{index}", token_count)
            return SUSTAINED_REQUEST_COUNT

        with ThreadPoolExecutor(max_workers=2) as pressure_executor:
            futures = (
                pressure_executor.submit(sustain, "prefix-pressure", prefix_server, workload.prefix_tokens),
                pressure_executor.submit(sustain, "peer-pressure", pressure_server, pressure_tokens),
            )
            completed = tuple(future.result() for future in futures)

        assert primary_initial < primary_hit
        assert competing_initial < competing_hit
        assert restored == restored_hit
        assert completed == (SUSTAINED_REQUEST_COUNT, SUSTAINED_REQUEST_COUNT)
        assert primary_hit_output_ids == primary_after_pressure_output_ids == primary_expected_output_ids
        assert competing_hit_output_ids == competing_after_pressure_output_ids == competing_expected_output_ids
        assert restored_fill_output_ids == restored_output_ids == restored_expected_output_ids
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
                },
                sort_keys=True,
            )
        )
        return results

    run = run_probe(
        case,
        models=tuple(MANIFEST.model(placement.model_id) for placement in case.models),
        serving_slo=MANIFEST.serving_slo,
        base_config=e2e_base_config,
        graph_settings=SglangGraphSettings(decode_backend="full", prefill_backend="breakable"),
        workdir=tmp_path,
        workload=requests,
    )
    assert_transport_observer_snapshots(
        run.launch.observer_outdir,
        expected_count=case.atnagent_count * len(case.models),
        site="atnagent",
    )
    assert_fabric_observer_snapshots(
        run.launch.observer_outdir,
        atnagent_count=case.atnagent_count,
        ffnagent_count=case.ffnagent_count,
        expected_topologies=tuple((model.atn_tp_size, model.atn_dp_size) for model in case.models),
    )
    assert_two_model_executor_overlap(run.launch.observer_outdir)
    assert_run_graph_evidence(run)
