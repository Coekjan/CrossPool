from __future__ import annotations

from typing import Literal

import pytest

from xbench.harness.serving.measure import (
    RequestRecord,
    StreamEvent,
    distribution,
    summarize,
    token_intervals,
    unavailable_summary,
)
from xbench.harness.serving.workload import PreparedWorkload, ResolvedPrompt, ScheduledRequest, content_digest
from xtest.harness.support.config import TEST_CASE_ID, TEST_MODEL_ID


def test_first_coalesced_chunk_has_no_invented_intervals_and_two_ttft_origins() -> None:
    events = (data("r", 0, 0.1, 3), data("r", 1, 0.16, 6))
    request = (
        terminal("r", 0.18, 6)
        .model_copy(
            update={
                "enqueued_at_seconds": 0.0,
                "http_started_at_seconds": 0.02,
                "first_token_at_seconds": 0.1,
                "last_token_at_seconds": 0.16,
                "first_completion_tokens": 3,
            }
        )
        .measured(events)
    )
    assert request.metrics.http_ttft_seconds == pytest.approx(0.08)
    assert request.metrics.arrival_ttft_seconds == pytest.approx(0.1)
    assert request.metrics.queue_wait_seconds == pytest.approx(0.02)
    assert request.metrics.mean_itl_seconds == pytest.approx(0.02)
    assert request.metrics.itl_sample_count == request.metrics.estimated_itl_sample_count == 3
    assert request.metrics.tpot_seconds == pytest.approx(0.016)
    assert request.metrics.itl_coverage == pytest.approx(3 / 5)


def test_duplicate_and_empty_text_progress_preserve_positive_anchor_and_terminal_increment() -> None:
    events = tuple(
        data("r", index, when, count)
        for index, (when, count) in enumerate(((0.08, 1), (0.10, 2), (0.12, 2), (0.13, 3), (0.15, 3), (0.17, 4)))
    )
    intervals = token_intervals(events)
    assert tuple(interval.weight for interval in intervals) == (1, 1, 1)
    assert tuple(interval.kind for interval in intervals) == ("observed", "observed", "observed")
    assert tuple(interval.value_seconds for interval in intervals) == pytest.approx((0.02, 0.03, 0.04))


def test_weighted_itl_differs_from_request_tpot_and_percentiles_are_linear() -> None:
    distribution_itl = distribution(((0.010, 1), (0.030, 9)))
    distribution_tpot = distribution(((0.010, 1), (0.030, 1)))
    assert distribution_itl.mean == pytest.approx(0.028)
    assert distribution_tpot.mean == pytest.approx(0.020)
    assert distribution_tpot.p90 == pytest.approx(0.028)
    assert distribution_itl.p90 == pytest.approx(0.030)
    assert distribution(((0.010, 1_000_000),)).sample_count == 1_000_000
    assert distribution(()).model_dump() == {
        "sample_count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "median": None,
        "p90": None,
        "p95": None,
        "p99": None,
    }


def test_buckets_keep_idle_horizon_final_edge_width_cached_input_and_failed_partial() -> None:
    workload = prepared(("success", "failure"), horizon=2.0)
    records = (
        terminal("success", 2.5, 6).model_copy(
            update={
                "first_token_at_seconds": 0.8,
                "last_token_at_seconds": 2.5,
                "first_completion_tokens": 2,
                "prompt_tokens": 5,
                "cached_tokens": 4,
            }
        ),
        terminal("failure", 1.1, 2, outcome="failed").model_copy(
            update={
                "first_token_at_seconds": 1.0,
                "last_token_at_seconds": 1.0,
                "first_completion_tokens": 2,
            }
        ),
    )
    events = (
        *lifecycle(records[0], (data("success", 0, 0.8, 2), data("success", 1, 2.5, 6))),
        *lifecycle(records[1], (data("failure", 0, 1.0, 2),)),
    )
    records = tuple(
        record.measured(tuple(event for event in events if event.request_id == record.request_id)) for record in records
    )
    summary = summarize(workload, records, events, window_end_seconds=2.5, window_kind="complete")
    buckets = tuple(bucket for bucket in summary.throughput if bucket.target_id == "aggregate")
    assert tuple(bucket.width_seconds for bucket in buckets) == (1.0, 1.0, 0.5)
    assert tuple(bucket.output_tokens for bucket in buckets) == (2, 0, 4)
    assert tuple(bucket.partial_output_tokens for bucket in buckets) == (0, 2, 0)
    assert buckets[-1].input_tokens_per_second == 10
    assert buckets[-1].output_tokens_per_second == 8
    assert summary.targets["aggregate"].output_tokens_per_second == pytest.approx(2.4)
    assert summary.outcomes["failed"] == summary.outcomes["success"] == 1
    assert summary.execution_complete
    assert summary.targets["aggregate"].distributions["itl_combined"].sample_count == 4
    assert summary.targets["aggregate"].distributions["http_ttft_seconds"].sample_count == 1
    assert all(point.cumulative_probability <= 1 for point in summary.cdf)


def test_zero_token_success_and_empty_schedule_do_not_invent_latency() -> None:
    request = terminal("r", 0.2, 0)
    summary = summarize(
        prepared(("r",), horizon=1.0),
        (request,),
        lifecycle(request, (data("r", 0, 0.1, 0),)),
        window_end_seconds=1.0,
        window_kind="complete",
    )
    assert summary.measurement_available
    assert summary.targets["aggregate"].distributions["http_ttft_seconds"].sample_count == 0
    assert summary.targets["aggregate"].distributions["itl_combined"].mean is None
    unavailable = unavailable_summary(prepared((), horizon=20), ())
    assert unavailable.window_end_seconds is None
    assert unavailable.arrival_horizon_seconds == 20
    assert not unavailable.measurement_available


@pytest.mark.parametrize("changed", [{"arrival_seconds": 0.1}, {"max_new_tokens": 19}])
@pytest.mark.parametrize("timed", [True, False])
def test_summary_rejects_request_policy_that_differs_from_prepared_workload(
    changed: dict[str, float | int], timed: bool
) -> None:
    workload = prepared(("r",), horizon=1.0)
    request = RequestRecord(**workload.requests[0].model_dump(), outcome="not_sent").model_copy(update=changed)
    with pytest.raises(ValueError, match="identity differs"):
        if timed:
            summarize(workload, (request,), (), window_end_seconds=1.0, window_kind="complete")
        else:
            unavailable_summary(workload, (request,))


def test_summary_rejects_success_without_native_done_evidence() -> None:
    request = terminal("r", 0.2, 0)
    events = lifecycle(request, (data("r", 0, 0.1, 0),))
    with pytest.raises(ValueError, match=r"success.*DONE"):
        summarize(
            prepared(("r",), horizon=1.0), (request,), events[:-1], window_end_seconds=1.0, window_kind="complete"
        )


@pytest.mark.parametrize("damage", ["sequence", "chronology", "lifecycle", "past_window", "terminal_tail"])
def test_summary_rejects_inconsistent_event_history(damage: str) -> None:
    request = terminal("r", 0.2, 0)
    events = list(lifecycle(request, (data("r", 0, 0.1, 0), data("r", 1, 0.15, 0))))
    if damage == "sequence":
        events[3] = events[3].model_copy(update={"sequence": 99})
    elif damage == "chronology":
        events[3] = events[3].model_copy(update={"observed_at_seconds": 0.05})
    elif damage == "lifecycle":
        events[0] = events[0].model_copy(update={"kind": "http_start"})
    elif damage == "past_window":
        events[3] = events[3].model_copy(update={"observed_at_seconds": 1.1})
    else:
        events.append(data("r", len(events), 0.3, 0))
    with pytest.raises(ValueError, match="stream event"):
        summarize(prepared(("r",), horizon=1.0), (request,), events, window_end_seconds=1.0, window_kind="complete")


@pytest.mark.parametrize("outcome", ["not_sent", "failed"])
def test_unavailable_origin_rejects_claims_of_known_http_state(outcome: Literal["not_sent", "failed"]) -> None:
    workload = prepared(("r",), horizon=1.0)
    request = RequestRecord(
        **workload.requests[0].model_dump(),
        outcome=outcome,
        error_kind="evidence_missing" if outcome == "failed" else "before_measurement",
        http_status=200,
    )
    with pytest.raises(ValueError, match="cannot claim HTTP state"):
        unavailable_summary(workload, (request,))


def lifecycle(request: RequestRecord, observations: tuple[StreamEvent, ...]) -> tuple[StreamEvent, ...]:
    assert request.ended_at_seconds is not None
    events = (
        StreamEvent(request_id=request.request_id, sequence=0, observed_at_seconds=0.0, kind="enqueue"),
        StreamEvent(request_id=request.request_id, sequence=1, observed_at_seconds=0.0, kind="http_start"),
        *observations[:-1],
        observations[-1].model_copy(
            update={
                "prompt_tokens": request.prompt_tokens,
                "cached_tokens": request.cached_tokens,
                "finish_reason": request.finish_reason,
            }
        ),
        StreamEvent(
            request_id=request.request_id,
            sequence=0,
            observed_at_seconds=request.ended_at_seconds,
            kind="done" if request.outcome == "success" else "error",
        ),
    )
    return tuple(event.model_copy(update={"sequence": sequence}) for sequence, event in enumerate(events))


def data(id: str, sequence: int, when: float, count: int) -> StreamEvent:
    return StreamEvent(request_id=id, sequence=sequence, observed_at_seconds=when, kind="data", completion_tokens=count)


def terminal(
    id: str, end: float, count: int, *, outcome: Literal["success", "failed", "cancelled", "not_sent"] = "success"
) -> RequestRecord:
    return RequestRecord(
        request_id=id,
        model_id=TEST_MODEL_ID,
        arrival_seconds=0.0,
        prompt_id="prompt",
        max_new_tokens=20,
        outcome=outcome,
        enqueued_at_seconds=0.0,
        http_started_at_seconds=0.0,
        ended_at_seconds=end,
        completion_tokens=count,
        prompt_tokens=5,
        finish_reason="length" if outcome == "success" else None,
        http_status=200,
    )


def prepared(ids: tuple[str, ...], *, horizon: float) -> PreparedWorkload:
    prompts = (ResolvedPrompt(prompt_id="prompt", model_id=TEST_MODEL_ID, input_ids=(1, 2)),)
    requests = tuple(
        ScheduledRequest(
            request_id=id, model_id=TEST_MODEL_ID, arrival_seconds=0.0, prompt_id="prompt", max_new_tokens=20
        )
        for id in ids
    )
    return PreparedWorkload(
        case_id=TEST_CASE_ID,
        model_ids=(TEST_MODEL_ID,),
        model_contexts={},
        prompts=prompts,
        requests=requests,
        warmup=(),
        arrival_horizon_seconds=horizon,
        bucket_seconds=1.0,
        seeds={},
        prompt_sha256=content_digest(prompts),
        trace_sha256=content_digest(requests),
        warmup_sha256=content_digest(()),
    )
