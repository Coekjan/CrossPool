"""Declarative SGLang E2E manifest schema and catalog behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from tests import TEST_CATALOG_PATH

from xkit.serving.sglang.graph import SglangGraphMode
from xpool.model import ModelId
from xtest.harness.sglang.catalog import (
    E2eElasticKvWorkload,
    E2eFfnInputMatrix,
    E2eServingCase,
    TestCatalog,
)


def test_catalogue_collects_deployment_resources_without_workspace_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    catalogue = TestCatalog.load(TEST_CATALOG_PATH)
    serving = {case.id: case for case in catalogue.serving_cases}
    topology = {case.id: case for case in catalogue.topology_cases}
    assert serving["serving-001"].required_device_count == 2
    assert serving["serving-006"].required_device_count == 4
    assert topology["topology-005"].required_device_count == 6
    assert all(
        case.deployment is not None and case.deployment.is_file()
        for case in (*catalogue.serving_cases, *catalogue.topology_cases)
    )


def test_catalogue_rejects_unknown_fields() -> None:
    declaration = TestCatalog.load(TEST_CATALOG_PATH).model_dump()
    declaration["serving_cases"][0]["unknown"] = True
    with pytest.raises(ValidationError, match="unknown"):
        TestCatalog.model_validate(declaration)


def test_serving_case_allows_fewer_executors_than_ffnagents() -> None:
    case = E2eServingCase(
        description="Two FFN ranks share one executor lane.",
        deployment=Path("configs/deployments/Qwen%2FQwen3-0.6B/atn1-ffn2-lanes1.toml"),
        models=(ModelId("Qwen/Qwen3-0.6B"),),
        graph_modes=(SglangGraphMode.EAGER,),
        estimated_duration_seconds=1,
        timeout_seconds=1,
    )

    assert case.ffnagent_count == 2
    assert case.executor_lane_count == 1


def test_ffn_input_matrix_rejects_unordered_rows() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        E2eFfnInputMatrix(seed=17, row_counts=(32, 1))


def test_serving_case_graph_modes_belong_to_the_matching_test_path() -> None:
    models = (
        ModelId("organization/first"),
        ModelId("organization/second"),
    )
    workload = E2eElasticKvWorkload(
        prefix_model_id=ModelId("organization/first"),
        prefix_tokens=1,
        pressure_model_id=ModelId("organization/second"),
        atn_device_memory_budget_bytes=1024,
    )
    with pytest.raises(ValidationError, match="ordinary E2E serving case graph_modes"):
        serving_case(models=models, graph_modes=())
    with pytest.raises(ValidationError, match="elastic KV workload owns its graph mode"):
        serving_case(models=models, elastic_kv=workload)


def serving_case(
    *,
    models: tuple[ModelId, ...],
    graph_modes: tuple[SglangGraphMode, ...] = (SglangGraphMode.EAGER,),
    elastic_kv: E2eElasticKvWorkload | None = None,
) -> E2eServingCase:
    return E2eServingCase(
        description="Validate workload and graph-mode declarations.",
        deployment=Path("deployment.toml"),
        models=models,
        graph_modes=graph_modes,
        elastic_kv=elastic_kv,
        estimated_duration_seconds=1,
        timeout_seconds=1,
    )
