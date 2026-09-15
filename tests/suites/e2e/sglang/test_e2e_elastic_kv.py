"""Cross-Instance elastic KV-cache serving evidence."""

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


def case_parameter(case: E2eServingCase) -> ParameterSet:
    """Attach the resources required by one elastic KV serving case."""

    model_ids = tuple(MANIFEST.model(placement.model).model_id for placement in case.models)
    marks = [
        pytest.mark.requires_cuda(min_devices=case.required_gpu_count),
        pytest.mark.requires_config,
        pytest.mark.requires_mps,
        pytest.mark.timeout(case.timeout_seconds),
        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
    ]
    marks.extend(pytest.mark.requires_model_weights(model_id) for model_id in model_ids)
    return pytest.param(case, id=case.id, marks=marks)


def generate(server: SglangServerProcess, request_id: str, token_count: int) -> tuple[int, tuple[int, ...]]:
    """Run one deterministic request and return its cache hit and output IDs."""

    response = httpx.post(
        f"{server.url()}/generate",
        json={
            "rid": request_id,
            "input_ids": [1] * token_count,
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
    """Prove prefix hit, cross-Instance reclamation, and repopulation."""

    workload = case.elastic_kv
    assert workload is not None

    def requests(servers: list[SglangServerProcess]) -> tuple[SglangServerResult, ...]:
        request_barrier = Barrier(len(servers))
        with ThreadPoolExecutor(max_workers=len(servers)) as executor:
            results = tuple(executor.map(SglangServerProcess.result, servers, repeat(request_barrier)))

        by_alias = {server.model.alias: server for server in servers}
        prefix_server = by_alias[workload.prefix_model]
        pressure_server = by_alias[workload.pressure_model]
        initial, expected_output_ids = generate(prefix_server, "xpool-elastic-kv-fill", workload.prefix_tokens)
        hit, hit_output_ids = generate(prefix_server, "xpool-elastic-kv-hit", workload.prefix_tokens)
        generate(pressure_server, "xpool-elastic-kv-pressure", workload.pressure_tokens)
        after_pressure, after_pressure_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-after-pressure",
            workload.prefix_tokens,
        )
        repopulated, repopulated_output_ids = generate(
            prefix_server,
            "xpool-elastic-kv-repopulated",
            workload.prefix_tokens,
        )

        assert initial < hit
        assert after_pressure < hit
        assert repopulated == hit
        assert hit_output_ids == after_pressure_output_ids == repopulated_output_ids == expected_output_ids
        print(
            "XPOOL_ELASTIC_KV_CACHE="
            + json.dumps(
                {
                    "after_pressure": after_pressure,
                    "hit": hit,
                    "initial": initial,
                    "repopulated": repopulated,
                },
                sort_keys=True,
            )
        )
        return results

    run = run_probe(
        MANIFEST,
        case,
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
