"""Declarative SGLang E2E manifest schema and catalog behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.harness.sglang.manifest import (
    E2E_MANIFEST_PATH,
    E2eElasticKvWorkload,
    E2eFfnInputMatrix,
    E2eManifest,
    E2eModelPlacement,
    E2eServingCase,
)
from tests.harness.sglang.serving.graph import SglangGraphMode


def test_manifest_loads_declared_serving_cases() -> None:
    manifest = E2eManifest.load(E2E_MANIFEST_PATH)

    assert manifest.models and manifest.model_serving_cases
    assert (manifest.serving_slo.ttft_ms, manifest.serving_slo.tbt_ms) == (1000, 50)
    assert all(not case.graph_modes for case in manifest.model_serving_cases if case.elastic_kv is not None)
    assert all(
        case.required_gpu_count == case.atnagent_count + case.ffnagent_count for case in manifest.model_serving_cases
    )


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "e2e.toml"
    manifest_path.write_text(
        """
[serving_slo]
ttft_ms = 1000
tbt_ms = 50

[models.synthetic]
model_id = "organization/model"
architecture = "SyntheticForCausalLM"
unknown = true

[[model_serving_cases]]
id = "synthetic"
models = [{ model = "synthetic", atn_tp_size = 1, atn_dp_size = 1 }]
ffnagent_count = 1
executor_lane_count = 1
graph_modes = ["eager"]
estimated_duration_seconds = 1
timeout_seconds = 1
transport_record_capacity = 1
fabric_record_capacity = 1

""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown"):
        E2eManifest.load(manifest_path)


def test_serving_case_allows_fewer_executors_than_ffnagents() -> None:
    case = E2eServingCase(
        id="shared-executor",
        models=(E2eModelPlacement(model="synthetic", atn_tp_size=1, atn_dp_size=1),),
        ffnagent_count=2,
        executor_lane_count=1,
        graph_modes=(SglangGraphMode.EAGER,),
        estimated_duration_seconds=1,
        timeout_seconds=1,
        transport_record_capacity=1,
        fabric_record_capacity=1,
    )

    assert case.ffnagent_count == 2
    assert case.executor_lane_count == 1


def test_model_placement_rejects_combined_attention_tp_by_dp() -> None:
    with pytest.raises(ValidationError, match="combined attention TP-by-DP"):
        E2eModelPlacement(model="synthetic", atn_tp_size=2, atn_dp_size=2)


def test_ffn_input_matrix_rejects_unordered_rows() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        E2eFfnInputMatrix(seed=17, row_counts=(32, 1))


def test_serving_case_rejects_heterogeneous_attention_topology() -> None:
    with pytest.raises(ValidationError, match="same attention topology"):
        serving_case(
            models=(
                E2eModelPlacement(model="first", atn_tp_size=1, atn_dp_size=1),
                E2eModelPlacement(model="second", atn_tp_size=2, atn_dp_size=1),
            )
        )


def test_serving_case_graph_modes_belong_to_the_matching_test_path() -> None:
    models = (
        E2eModelPlacement(model="first", atn_tp_size=1, atn_dp_size=1),
        E2eModelPlacement(model="second", atn_tp_size=1, atn_dp_size=1),
    )
    workload = E2eElasticKvWorkload(
        prefix_model="first",
        prefix_tokens=1,
        pressure_model="second",
        pressure_tokens=1,
        atn_device_memory_utilization=0.5,
    )
    with pytest.raises(ValidationError, match="ordinary E2E serving case graph_modes"):
        serving_case(models=models, graph_modes=())
    with pytest.raises(ValidationError, match="elastic KV workload owns its graph mode"):
        serving_case(models=models, elastic_kv=workload)


def serving_case(
    *,
    models: tuple[E2eModelPlacement, ...],
    graph_modes: tuple[SglangGraphMode, ...] = (SglangGraphMode.EAGER,),
    elastic_kv: E2eElasticKvWorkload | None = None,
) -> E2eServingCase:
    return E2eServingCase(
        id="synthetic-case",
        models=models,
        ffnagent_count=1,
        executor_lane_count=1,
        graph_modes=graph_modes,
        elastic_kv=elastic_kv,
        estimated_duration_seconds=1,
        timeout_seconds=1,
        transport_record_capacity=1,
        fabric_record_capacity=1,
    )
