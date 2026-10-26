"""Original MXFP4 numerical and hybrid-attention Graph qualification."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
from tests import TEST_CATALOG_PATH

import xtest
from xkit.deployment import resolve_deployment_path
from xkit.serving.sglang.graph import SglangGraphMode
from xpool.model import ModelId
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import attention, numerical
from xtest.harness.sglang.catalog import E2eFfnInputMatrix, E2eFfnNumericalCase, E2eServingCase
from xtest.harness.sglang.serving import qualification
from xtest.harness.support.sglang.runtime import published_sglang_config

pytest_plugins = ("xtest.harness.support.config",)

MODEL = ModelId("openai/gpt-oss-20b")
NUMERICAL_CASES: tuple[E2eFfnNumericalCase, ...] = (
    E2eFfnNumericalCase(
        description="Original packed Experts and biased Router against the independent SGLang FFN implementation.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn2-lanes1"),
        model_id=MODEL,
        layer_ids=(0, 23),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 32, 4096)),
        estimated_duration_seconds=900,
        timeout_seconds=1800,
    ),
)
SERVING_CASES: tuple[E2eServingCase, ...] = (
    E2eServingCase(
        description="Original Full/SWA hybrid storage with attention TP one and FFN TP two.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn2-lanes1"),
        models=(MODEL,),
        graph_modes=(
            SglangGraphMode.EAGER,
            SglangGraphMode.DECODE_FULL,
            SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE,
        ),
        estimated_duration_seconds=800,
        timeout_seconds=3600,
    ),
)


@xtest.parameterize("case", tuple(numerical.case_parameter(case) for case in NUMERICAL_CASES))
@xtest.requirements(numerical.requirements_of)
def test_ffn_numerical(
    case: E2eFfnNumericalCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Compare installed true-TP FFN execution with the original model's FFN."""

    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir)


@xtest.requirements(device_count=1, requires_config=True, model_ids=(MODEL,))
@pytest.mark.usefixtures(published_sglang_config.__name__)
def test_attention_numerical(
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    attention.run_attention_numerical(MODEL, e2e_base_config, task_artifact_dir or tmp_path)


@xtest.parameterize(("case", "graph_mode"), SERVING_CASES, rows=partial(qualification.graph_rows, compare_modes=True))
@xtest.requirements(qualification.requirements_of)
def test_serving_graph(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove installed Graph modes and their cross-mode numerical alignment."""

    qualification.run_serving_case(case, graph_mode, e2e_base_config, tmp_path, task_artifact_dir, compare_modes=True)
