"""Assertions for requested and observed SGLang graph execution."""

from __future__ import annotations

from tests.harness.sglang.attempt import ProbeRun
from tests.harness.sglang.graph import GraphEvent, SglangGraphSettings


def assert_run_graph_evidence(run: ProbeRun) -> None:
    """Validate resolved modes and matching graph-observer evidence."""

    assert len(run.results) == len(run.launch.models)
    for result, model in zip(run.results, run.launch.models, strict=True):
        assert result.model_id == model.model_id
        assert result.resolved_graph_settings.cuda_graph is run.graph_settings.cuda_graph
        if run.graph_settings.piecewise_cuda_graph:
            assert result.resolved_graph_settings.piecewise_cuda_graph is (model.atn_dp_size == 1)
        else:
            assert result.resolved_graph_settings.piecewise_cuda_graph is False
    resolved_piecewise = any(result.resolved_graph_settings.piecewise_cuda_graph for result in run.results)
    assert_graph_events(run.graph_settings, run.events, resolved_piecewise_cuda_graph=resolved_piecewise)


def assert_graph_events(
    graph_settings: SglangGraphSettings,
    events: list[GraphEvent],
    *,
    resolved_piecewise_cuda_graph: bool,
) -> None:
    """Require capture and replay evidence for every resolved graph mode."""

    full_graph_phases = graph_phases(events, kind="full_cuda_graph")
    piecewise_graph_phases = graph_phases(events, kind="piecewise_cuda_graph")
    expected_phases = {"capture_begin", "capture_end", "replay_begin", "replay_end"}
    if graph_settings.cuda_graph:
        assert expected_phases <= full_graph_phases
    else:
        assert full_graph_phases == set()
    if resolved_piecewise_cuda_graph:
        assert expected_phases <= piecewise_graph_phases
    else:
        assert piecewise_graph_phases == set()


def graph_phases(events: list[GraphEvent], *, kind: str) -> set[str]:
    """Collect observed phases for one graph-observer event kind."""

    return {phase for event in events if event.get("kind") == kind and isinstance((phase := event.get("phase")), str)}
