"""Explicit SGLang numerical and graph qualification for Qwen/Qwen3-0.6B."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from tests import TEST_CATALOG_PATH

import xtest
from xkit.deployment import resolve_deployment_path
from xkit.serving.sglang.graph import SglangGraphMode
from xpool.model import ModelId
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import numerical
from xtest.harness.sglang.catalog import (
    E2eFfnInputMatrix,
    E2eFfnNumericalCase,
    E2eServingCase,
)
from xtest.harness.sglang.serving import qualification

pytest_plugins = ("xtest.harness.support.config",)

MODEL = ModelId("Qwen/Qwen3-0.6B")
NUMERICAL_CASES: tuple[E2eFfnNumericalCase, ...] = (
    E2eFfnNumericalCase(
        description="Representative real-checkpoint FFN numerical parity against the original SGLang implementation.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn2-lanes1"),
        model_id=MODEL,
        layer_ids=(0, 27),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 32, 4096)),
        estimated_duration_seconds=300,
        timeout_seconds=1800,
    ),
)
SERVING_CASES: tuple[E2eServingCase, ...] = (
    E2eServingCase(
        description="Installed serving graph qualification with attention TP 1, DP 1 and FFN TP 1.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn1-lanes1"),
        models=(MODEL,),
        graph_modes=(
            SglangGraphMode.EAGER,
            SglangGraphMode.DECODE_FULL,
            SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE,
        ),
        estimated_duration_seconds=300,
        timeout_seconds=1800,
    ),
)


@xtest.parameterize(
    "case",
    tuple(numerical.case_parameter(case) for case in NUMERICAL_CASES),
)
@xtest.requirements(numerical.requirements_of)
def test_ffn_numerical(
    case: E2eFfnNumericalCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Compare installed FFN with an independent SGLang reference."""

    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir)


@xtest.parameterize(
    ("case", "graph_mode"),
    SERVING_CASES,
    rows=partial(qualification.graph_rows, compare_modes=True),
)
@xtest.requirements(qualification.requirements_of)
def test_serving_graph(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove installed graph modes and their cross-mode alignment."""

    qualification.run_serving_case(
        case,
        graph_mode,
        e2e_base_config,
        tmp_path,
        task_artifact_dir,
        compare_modes=True,
    )
