"""Manifest-driven evidence for every reachable debug loopback site."""

from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.mark.structures import ParameterSet

from tests.harness.sglang.graph import SglangGraphMode, assert_graph_events
from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eLoopbackServingCase, E2eManifest
from tests.harness.sglang.observer import (
    assert_fabric_observer_snapshots,
    assert_no_fabric_invocations,
    assert_no_transport_observer_snapshots,
    assert_transport_observer_snapshots,
)
from tests.harness.sglang.parity import TOKEN_PARITY_ARTIFACT_FILENAME
from tests.harness.sglang.probe import ProbeRun, run_probe
from xpool.config import LoopbackSite, XpoolConfig

MANIFEST = E2eManifest.load(E2E_MANIFEST_PATH)


def case_parameter(
    case: E2eLoopbackServingCase,
    site: LoopbackSite,
    graph_mode: SglangGraphMode,
) -> ParameterSet:
    """Attach every manifest-derived requirement to one loopback-site task."""

    model_ids = tuple(MANIFEST.model(placement.model).model_id for placement in case.models)
    marks = [
        pytest.mark.requires_cuda(min_devices=case.required_gpu_count),
        pytest.mark.requires_config,
        pytest.mark.requires_mps,
        pytest.mark.timeout(case.timeout_seconds),
        pytest.mark.estimated_duration(seconds=case.estimated_duration_seconds),
    ]
    if len(case.graph_modes) >= 2:
        marks.append(
            pytest.mark.token_parity_group(
                name=f"{case.id}-{site.value}",
                expected_case_count=len(case.graph_modes),
            )
        )
    marks.extend(pytest.mark.requires_model_weights(model_id) for model_id in model_ids)
    return pytest.param(case, site, graph_mode, id=f"{case.id}-{site.value}-{graph_mode.value}", marks=marks)


@pytest.mark.parametrize(
    ("case", "site", "graph_mode"),
    tuple(
        case_parameter(case, site, graph_mode)
        for case in MANIFEST.loopback_serving_cases
        for site in case.sites
        for graph_mode in case.graph_modes
    ),
)
def test_e2e_loopback_serving(
    case: E2eLoopbackServingCase,
    site: LoopbackSite,
    graph_mode: SglangGraphMode,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove one explicit debug site through installed xpool and SGLang commands."""

    run = run_probe(
        MANIFEST,
        case,
        base_config=e2e_base_config,
        graph_settings=graph_mode.settings(),
        workdir=tmp_path,
        loopback_site=site,
    )
    assert_site_evidence(site, run)
    result = run.results[0]
    assert result.resolved_graph_settings.cuda_graph is run.graph_settings.cuda_graph
    resolved_piecewise = result.resolved_graph_settings.piecewise_cuda_graph
    assert resolved_piecewise is run.graph_settings.piecewise_cuda_graph
    assert_graph_events(run.graph_settings, run.events, resolved_piecewise_cuda_graph=resolved_piecewise)
    if task_artifact_dir is not None and len(case.graph_modes) >= 2:
        run.token_parity_artifact(f"{case.id}-{site.value}").write(task_artifact_dir / TOKEN_PARITY_ARTIFACT_FILENAME)


def assert_site_evidence(site: LoopbackSite, run: ProbeRun) -> None:
    """Require the highest observer boundary reachable at one loopback site."""

    outdir = run.launch.observer_outdir
    if site is LoopbackSite.INSTANCE:
        assert_no_transport_observer_snapshots(outdir)
        assert_no_fabric_invocations(outdir)
    elif site is LoopbackSite.ATNAGENT:
        assert_transport_observer_snapshots(outdir, expected_count=1)
        assert_no_fabric_invocations(outdir)
    else:
        assert_transport_observer_snapshots(outdir, expected_count=1)
        assert_fabric_observer_snapshots(
            outdir,
            atnagent_count=1,
            ffnagent_count=1,
            expected_topologies=((1, 1),),
        )
