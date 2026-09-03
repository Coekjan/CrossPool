"""Assertions for requested and observed SGLang graph execution."""

from __future__ import annotations

from tests.harness.sglang.serving.attempt import ProbeRun
from tests.harness.sglang.serving.graph import GraphEvent, SglangGraphSettings


def assert_run_graph_evidence(run: ProbeRun) -> None:
    """Validate resolved modes and matching graph-observer evidence."""

    assert len(run.results) == len(run.launch.models)
    for result, model in zip(run.results, run.launch.models, strict=True):
        assert result.model_id == model.model_id
        assert result.resolved_graph_settings == run.graph_settings
    assert_graph_events(run.graph_settings, run.events)


def assert_graph_events(
    graph_settings: SglangGraphSettings,
    events: list[GraphEvent],
) -> None:
    """Require capture and execution evidence for every resolved graph mode."""

    decode_events = graph_events(events, forward_phase="decode")
    prefill_events = graph_events(events, forward_phase="prefill")
    expected_events = {"capture_begin", "capture_end", "execute_begin", "execute_end"}
    if graph_settings.decode_backend == "full":
        assert expected_events <= {event.get("event") for event in decode_events}
        assert {event.get("backend_class") for event in decode_events} == {"FullCudaGraphBackend"}
    else:
        assert decode_events == []
    if graph_settings.prefill_backend == "breakable":
        assert expected_events <= {event.get("event") for event in prefill_events}
        assert {event.get("backend_class") for event in prefill_events} == {"BreakableCudaGraphBackend"}
    else:
        assert prefill_events == []


def graph_events(events: list[GraphEvent], *, forward_phase: str) -> list[GraphEvent]:
    """Collect observed events for one graph runner phase."""

    return [event for event in events if event.get("forward_phase") == forward_phase]
