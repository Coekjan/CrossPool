"""Assertions for requested and observed SGLang graph execution."""

from __future__ import annotations

from collections import Counter

from xkit.serving.sglang.graph import SglangGraphSettings
from xpool.integrations.sglang.devkit import SglangGraphEvent
from xtest.harness.sglang.serving.attempt import ProbeRun


def assert_run_graph_evidence(run: ProbeRun) -> None:
    """Validate resolved modes and matching graph-observer evidence."""

    assert len(run.results) == len(run.model_ids)
    for result, model_id in zip(run.results, run.model_ids, strict=True):
        assert result.model_id == model_id
        assert result.resolved_graph_settings == run.graph_settings
        model_events = [event for event in run.events if event["model_id"] == model_id]
        measured = [event for event in model_events if result.started_ns <= event["time_ns"] <= result.finished_ns]
        decode = [event for event in measured if event["forward_phase"] == "decode"]
        prefill = [event for event in measured if event["forward_phase"] == "prefill"]
        if run.graph_settings.decode_backend == "full":
            replays_by_pid = Counter(event["pid"] for event in decode if event["event"] == "execute_end")
            assert max(replays_by_pid.values(), default=0) >= len(result.output_ids) - 1
        else:
            assert not decode
        if run.graph_settings.prefill_backend == "breakable":
            assert any(event["event"] == "execute_end" for event in prefill)
        else:
            assert not prefill
        assert_graph_events(run.graph_settings, model_events)


def assert_graph_events(
    graph_settings: SglangGraphSettings,
    events: list[SglangGraphEvent],
) -> None:
    """Require capture and execution evidence for every resolved graph mode."""

    decode_events = [event for event in events if event["forward_phase"] == "decode"]
    prefill_events = [event for event in events if event["forward_phase"] == "prefill"]
    expected_events = {"capture_begin", "capture_end", "execute_begin", "execute_end"}
    if graph_settings.decode_backend == "full":
        assert expected_events <= {event["event"] for event in decode_events}
        assert {event["backend_class"] for event in decode_events} == {"FullCudaGraphBackend"}
    else:
        assert decode_events == []
    if graph_settings.prefill_backend == "breakable":
        assert expected_events <= {event["event"] for event in prefill_events}
        assert {event["backend_class"] for event in prefill_events} == {"BreakableCudaGraphBackend"}
    else:
        assert prefill_events == []
