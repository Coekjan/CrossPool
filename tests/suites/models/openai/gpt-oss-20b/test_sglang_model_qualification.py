"""Explicit GPT-OSS MXFP4 numerical and serving qualification on A100."""

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
from xtest.harness.sglang.catalog import E2eFfnInputMatrix, E2eFfnNumericalCase, E2eServingCase
from xtest.harness.sglang.serving import qualification

pytest_plugins = ("xtest.harness.support.config",)
MODEL = ModelId("openai/gpt-oss-20b")
DEPLOYMENT = resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn2-ffn5-tp1-lanes1")
NUMERICAL_CASES = (
    E2eFfnNumericalCase(
        description="Resident MXFP4 FFN against the independent original SGLang layer reference.",
        deployment=DEPLOYMENT,
        model_id=MODEL,
        layer_ids=(0, 23),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 32, 512)),
        estimated_duration_seconds=900,
        timeout_seconds=3600,
    ),
)
SERVING_CASES = (
    E2eServingCase(
        description="A100 installed GPT-OSS graph qualification with attention TP2 and FFN TP1.",
        deployment=DEPLOYMENT,
        models=(MODEL,),
        graph_modes=(SglangGraphMode.EAGER, SglangGraphMode.DECODE_FULL, SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
        estimated_duration_seconds=900,
        timeout_seconds=3600,
    ),
)


@xtest.parameterize("case", tuple(numerical.case_parameter(case) for case in NUMERICAL_CASES))
@xtest.requirements(numerical.requirements_of)
def test_ffn_numerical(
    case: E2eFfnNumericalCase, e2e_base_config: ResolvedConfig, tmp_path: Path, task_artifact_dir: Path | None
) -> None:
    """Compare real-checkpoint installed FFN with original SGLang."""

    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir)


@xtest.parameterize(("case", "graph_mode"), SERVING_CASES, rows=partial(qualification.graph_rows, compare_modes=True))
@xtest.requirements(qualification.requirements_of)
def test_serving_graph(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Qualify Eager, Decode Full and Breakable Prefill on the same checkpoint."""

    qualification.run_serving_case(case, graph_mode, e2e_base_config, tmp_path, task_artifact_dir, compare_modes=True)


@xtest.requirements(qualification.requirements_of)
@xtest.parameterize("case", SERVING_CASES)
def test_sliding_window_and_prefix_reuse(case: E2eServingCase, e2e_base_config: ResolvedConfig, tmp_path: Path) -> None:
    """Exercise a 4097-token prefix with ordinary Elastic KV storage."""

    from tests.suites.models.openai.gpt_oss_support import qualify_sliding_window

    qualify_sliding_window(case, e2e_base_config, tmp_path)
