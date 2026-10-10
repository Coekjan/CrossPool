"""Client-observed timing, token-weighted distributions and throughput buckets."""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, FiniteFloat, JsonValue

from xbench.harness.serving.case import (
    BenchCase,
    BenchValue,
    ResolvedDeployment,
    ServingMetadata,
)
from xbench.harness.serving.workload import PreparedWorkload, ScheduledRequest, file_digest, read_jsonl
from xkit.case import CaseId
from xkit.config import ToolConfigRecord

REPETITION_DIRECTORY_PATTERN = re.compile(r"repetition-([0-9]{4,})(?:\.attempt-([0-9]{4,}))?")


def retained_file(directory: Path, reference: str) -> Path:
    """Resolve a retained file within its owning directory, rejecting escapes."""
    path = (directory / reference).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"benchmark artifact is missing or escapes its owner: {reference}")
    return path


class BenchRunManifest(BenchValue):
    """Invocation selection and runner-finalized outcome, independent of reports."""

    run_id: str
    tool_config: ToolConfigRecord
    selected_cases: tuple[CaseId, ...]
    case_directories: tuple[str, ...] = ()
    finished: bool = Field(default=False, description="All admitted case execution and finalization has ended.")
    result_code: int | None = Field(default=None, description="Current invocation outcome; None until finalization.")
    infrastructure_error: str | None = None


class BenchCaseManifest(BenchValue):
    """Case declaration, replay identities, deployment and effective attempt references."""

    case: BenchCase
    prompt_sha256: str
    trace_sha256: str
    warmup_sha256: str
    deployment: ResolvedDeployment | None = None
    serving_metadata: ServingMetadata | None = None
    repetitions: tuple[str, ...]

    def effective_attempts(self, directory: Path) -> dict[int, Path]:
        """Resolve one physical attempt per logical repetition inside its case."""
        attempts = {}
        for reference in self.repetitions:
            match = REPETITION_DIRECTORY_PATTERN.fullmatch(reference)
            path = (directory / reference).resolve()
            if (
                match is None
                or int(match[1]) < 1
                or (match[2] is not None and int(match[2]) < 1)
                or path.parent != directory.resolve()
                or path.name != reference
            ):
                raise ValueError("benchmark repetition reference escapes its case or is not a physical attempt")
            number = int(match[1])
            if number in attempts:
                raise ValueError("benchmark case declares multiple effective attempts for one repetition")
            attempts[number] = path
        return attempts

    def load_workload(self, directory: Path) -> PreparedWorkload:
        """Read the original replay and check its case-owned identities and timing."""
        workload = PreparedWorkload.load(directory, case=self.case)
        if (
            workload.prompt_sha256 != self.prompt_sha256
            or workload.trace_sha256 != self.trace_sha256
            or workload.warmup_sha256 != self.warmup_sha256
        ):
            raise ValueError("retained replay content identities disagree")
        if workload.bucket_seconds != self.case.bucket_seconds or (
            self.case.arrivals.duration_seconds is not None
            and workload.arrival_horizon_seconds != self.case.arrivals.duration_seconds
        ):
            raise ValueError("retained workload declarations disagree with the case")
        return workload


class RepetitionManifest(BenchValue):
    """One attempt's timing and outcome, sealed by the runner after domain drain."""

    repetition: int = Field(gt=0)
    window_end_seconds: FiniteFloat | None = Field(
        default=None, ge=0, description="Common window end relative to T0; None when a usable window is unavailable."
    )
    window_kind: Literal["complete", "interrupted", "observed_prefix"] | None = Field(
        default=None, description="Measurement-owner end or recovered observation prefix; None without a usable window."
    )
    finished: bool = Field(default=False, description="Runner finalization has ended; this is not a success verdict.")
    result_code: int | None = Field(default=None, description="Original repetition outcome; None until finalization.")
    cleanup_verified: bool | None = Field(
        default=None, description="Outer runner's descendant-drain proof; None until verified or rejected."
    )
    raw_evidence_complete: bool | None = Field(
        default=None, description="Raw recording and terminal request accounting are intact; None before finalization."
    )
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    infrastructure_error: str | None = None

    @property
    def sealed(self) -> bool:
        """Whether this attempt has its own final outcome and cleanup observation."""
        return self.finished and self.result_code is not None and self.cleanup_verified is not None

    @classmethod
    def from_directory(cls, directory: Path) -> RepetitionManifest:
        """Read an attempt checkpoint whose logical number matches its path."""
        match = REPETITION_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None or (match[2] is not None and int(match[2]) < 1):
            raise ValueError(f"invalid benchmark attempt directory: {directory.name}")
        manifest = cls.model_validate_json(retained_file(directory, "repetition.json").read_bytes())
        if manifest.repetition != int(match[1]):
            raise ValueError("benchmark attempt number disagrees with its directory")
        return manifest


class MeasurementOrigin(BenchValue):
    """Common native-client T0; wall time identifies the run rather than latency."""

    monotonic_origin_seconds: FiniteFloat = Field(ge=0)
    wall_clock_origin_seconds: FiniteFloat = Field(ge=0)


def load_measurement(directory: Path, workload: PreparedWorkload) -> tuple[RepetitionManifest, BenchSummary]:
    """Validate saved attempt measurements independently of the parent run's seal.

    Continuation can reuse a successful sealed attempt in an interrupted parent.
    Reporting applies the parent publication boundary separately.
    """
    manifest = RepetitionManifest.from_directory(directory)
    required = {"events.jsonl", "requests.jsonl"}
    if (
        manifest.window_end_seconds is not None
        or "measurement.json" in manifest.artifact_sha256
        or (directory / "measurement.json").is_file()
    ):
        required.add("measurement.json")
    if not required <= manifest.artifact_sha256.keys():
        raise ValueError("benchmark repetition lacks required evidence digests")
    for reference in required:
        artifact = retained_file(directory, reference)
        if file_digest(artifact) != manifest.artifact_sha256[reference]:
            raise ValueError(f"retained benchmark artifact digest mismatch: {reference}")
    if "measurement.json" in required:
        MeasurementOrigin.model_validate_json(retained_file(directory, "measurement.json").read_bytes())
    requests = read_jsonl(directory / "requests.jsonl", RequestRecord)
    events = read_jsonl(directory / "events.jsonl", StreamEvent)
    if manifest.window_end_seconds is None and events:
        raise ValueError("benchmark events require an available measurement window")
    if (manifest.window_end_seconds is None) != (manifest.window_kind is None):
        raise ValueError("measurement window end and kind must be available together")
    summary = (
        summarize(
            workload,
            requests,
            events,
            window_end_seconds=manifest.window_end_seconds,
            window_kind=manifest.window_kind,
        )
        if manifest.window_end_seconds is not None and manifest.window_kind is not None
        else unavailable_summary(workload, requests)
    ).model_copy(
        update={"cleanup_verified": manifest.cleanup_verified, "infrastructure_error": manifest.infrastructure_error}
    )
    if manifest.result_code == 0 and (
        not summary.execution_complete
        or not summary.measurement_available
        or not summary.evidence_complete
        or not summary.cleanup_verified
        or manifest.raw_evidence_complete is False
        or summary.infrastructure_error is not None
        or any(summary.outcomes[kind] for kind in ("failed", "cancelled", "not_sent"))
    ):
        raise ValueError("successful benchmark checkpoint contradicts its completeness or outcomes")
    if not manifest.sealed or manifest.raw_evidence_complete is not True:
        summary = summary.model_copy(update={"evidence_complete": False})
    if manifest.raw_evidence_complete is False:
        summary = summary.model_copy(update={"execution_complete": False})
    return manifest, summary


class StreamEvent(BenchValue):
    """Timestamped lifecycle or protocol observation, including rejected frames."""

    request_id: str
    sequence: int = Field(ge=0)
    observed_at_seconds: FiniteFloat = Field(ge=0)
    kind: Literal["enqueue", "http_start", "data", "done", "error"]
    completion_tokens: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    accepted: bool = True
    reported_meta: JsonValue = None
    error_kind: str | None = None
    error_message: str | None = None


class RequestMetrics(BenchValue):
    """Per-request projections; unavailable timings and token intervals stay None."""

    queue_wait_seconds: FiniteFloat | None = Field(default=None, ge=0)
    arrival_lateness_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_ttft_seconds: FiniteFloat | None = Field(default=None, ge=0)
    arrival_ttft_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_latency_seconds: FiniteFloat | None = Field(default=None, ge=0)
    arrival_latency_seconds: FiniteFloat | None = Field(default=None, ge=0)
    mean_itl_seconds: FiniteFloat | None = Field(default=None, ge=0)
    tpot_seconds: FiniteFloat | None = Field(default=None, ge=0)
    itl_coverage: FiniteFloat | None = Field(default=None, ge=0)
    itl_sample_count: int = 0
    observed_itl_sample_count: int = 0
    estimated_itl_sample_count: int = 0


class RequestRecord(ScheduledRequest):
    """Terminal request facts checked against replay and the retained event prefix."""

    outcome: Literal["success", "failed", "cancelled", "not_sent"]
    enqueued_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_started_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    first_token_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    last_token_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    ended_at_seconds: FiniteFloat | None = Field(default=None, ge=0)
    http_status: int | None = None
    error_kind: str | None = None
    error_message: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    first_completion_tokens: int | None = Field(default=None, gt=0)
    metrics: RequestMetrics = Field(default_factory=RequestMetrics)

    def validate_evidence(self, events: Sequence[StreamEvent]) -> None:
        """Reject terminal claims unsupported by the retained native event prefix.

        Event order is checked by the summary boundary. Recovered owner-loss
        records retain unknown timestamps rather than inferring dispatch state.
        """

        timestamps = (
            self.enqueued_at_seconds,
            self.http_started_at_seconds,
            self.first_token_at_seconds,
            self.last_token_at_seconds,
            self.ended_at_seconds,
        )
        if (self.outcome == "not_sent" or self.error_kind == "evidence_missing") and any(
            value is not None
            for value in (
                self.http_status,
                self.finish_reason,
                self.prompt_tokens,
                self.cached_tokens,
                self.completion_tokens,
                self.first_completion_tokens,
            )
        ):
            raise ValueError("not-sent/evidence-missing request cannot claim HTTP state or token usage")
        if self.error_kind == "evidence_missing":
            if self.outcome != "failed" or any(value is not None for value in timestamps):
                raise ValueError("evidence-missing request cannot claim known termination or timestamps")
            return
        enqueued = events[0].observed_at_seconds if events else None
        started = events[1].observed_at_seconds if len(events) > 1 else None
        if self.enqueued_at_seconds != enqueued or self.http_started_at_seconds != started:
            raise ValueError("request dispatch timestamps differ from lifecycle events")
        if self.outcome == "not_sent":
            if len(events) > 1 or any(value is not None for value in timestamps[1:]):
                raise ValueError("not-sent request cannot have HTTP or termination observations")
        elif self.outcome == "success":
            if (
                not events
                or events[-1].kind != "done"
                or not events[-1].accepted
                or self.http_status != 200
                or self.finish_reason not in {"stop", "length"}
                or self.prompt_tokens is None
                or self.completion_tokens is None
                or self.error_kind is not None
            ):
                raise ValueError("success requires HTTP 200, normal final usage and accepted DONE evidence")
        elif not events or events[-1].kind != "error":
            raise ValueError("failed/cancelled request requires a terminal error observation")
        if self.outcome != "not_sent" and self.ended_at_seconds != events[-1].observed_at_seconds:
            raise ValueError("request termination time differs from its terminal event")
        usage: dict[str, int | str | float | None] = dict.fromkeys(
            (
                "prompt_tokens",
                "cached_tokens",
                "completion_tokens",
                "finish_reason",
                "first_token_at_seconds",
                "last_token_at_seconds",
                "first_completion_tokens",
            )
        )
        previous = 0
        prompt: int | None = None
        cached: int | None = None
        for event in events:
            if event.kind != "data" or not event.accepted:
                continue
            if event.completion_tokens is not None:
                if event.completion_tokens < previous:
                    raise ValueError("accepted completion counts must be monotonic")
                if event.completion_tokens > previous:
                    if usage["first_token_at_seconds"] is None:
                        usage["first_token_at_seconds"] = event.observed_at_seconds
                        usage["first_completion_tokens"] = event.completion_tokens
                    usage["last_token_at_seconds"] = event.observed_at_seconds
                previous = event.completion_tokens
            if event.finish_reason is not None and (
                event.finish_reason not in {"stop", "length"}
                or event.prompt_tokens is None
                or event.completion_tokens is None
            ):
                raise ValueError("accepted native finish requires normal reason and final usage")
            for name in ("prompt_tokens", "cached_tokens", "completion_tokens", "finish_reason"):
                value = getattr(event, name)
                if value is not None:
                    usage[name] = value
            if event.prompt_tokens is not None:
                prompt = event.prompt_tokens
            if event.cached_tokens is not None:
                cached = event.cached_tokens
            if prompt is not None and cached is not None and cached > prompt:
                raise ValueError("accepted cached tokens exceed logical prompt usage")
        if any(getattr(self, name) != value for name, value in usage.items()):
            raise ValueError("request usage or token timestamps differ from accepted stream evidence")

    def measured(self, events: Sequence[StreamEvent]) -> RequestRecord:
        """Project raw observations to nullable metrics, retaining partial evidence."""

        def difference(end: float | None, start: float | None) -> float | None:
            return None if end is None or start is None else end - start

        intervals = token_intervals(events)
        count = sum(interval.weight for interval in intervals)
        first = self.first_token_at_seconds
        completed = self.completion_tokens
        first_count = self.first_completion_tokens
        metrics = RequestMetrics(
            queue_wait_seconds=difference(self.http_started_at_seconds, self.enqueued_at_seconds),
            arrival_lateness_seconds=difference(self.enqueued_at_seconds, self.arrival_seconds),
            http_ttft_seconds=difference(first, self.http_started_at_seconds),
            arrival_ttft_seconds=difference(first, self.enqueued_at_seconds),
            http_latency_seconds=difference(self.ended_at_seconds, self.http_started_at_seconds),
            arrival_latency_seconds=difference(self.ended_at_seconds, self.enqueued_at_seconds),
            mean_itl_seconds=math.fsum(interval.value_seconds * interval.weight for interval in intervals) / count
            if count
            else None,
            tpot_seconds=(self.ended_at_seconds - first) / (completed - 1)
            if first is not None and self.ended_at_seconds is not None and completed is not None and completed > 1
            else None,
            itl_coverage=(completed - first_count) / (completed - 1)
            if completed is not None and completed > 1 and first_count is not None
            else None,
            itl_sample_count=count,
            observed_itl_sample_count=sum(interval.weight for interval in intervals if interval.kind == "observed"),
            estimated_itl_sample_count=sum(interval.weight for interval in intervals if interval.kind == "estimated"),
        )
        return self.model_copy(update={"metrics": metrics})


@dataclass(frozen=True, slots=True)
class TokenInterval:
    value_seconds: float
    weight: int
    kind: Literal["observed", "estimated"]


def token_intervals(events: Sequence[StreamEvent]) -> tuple[TokenInterval, ...]:
    """Calculate intervals from accepted counts and chronological observations."""

    intervals = []
    previous_count = 0
    previous_time = None
    for event in events:
        if event.kind != "data" or not event.accepted or event.completion_tokens is None:
            continue
        count = event.completion_tokens
        delta = count - previous_count
        if delta == 0:
            continue
        if previous_time is not None:
            gap = event.observed_at_seconds - previous_time
            intervals.append(TokenInterval(gap / delta, delta, "observed" if delta == 1 else "estimated"))
        previous_count = count
        previous_time = event.observed_at_seconds
    return tuple(intervals)


class Distribution(BenchValue):
    """Weighted population statistics; an empty population has nullable values."""

    sample_count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None


def distribution(samples: Sequence[tuple[float, int]]) -> Distribution:
    """Aggregate validated finite samples with positive token or request weights."""

    if not samples:
        return Distribution(sample_count=0)
    ordered = sorted(samples)
    weights = []
    cumulative = 0
    for _, weight in ordered:
        cumulative += weight
        weights.append(cumulative)
    mean = math.fsum(value * weight for value, weight in ordered) / cumulative

    def percentile(p: float) -> float:
        rank = (cumulative - 1) * p / 100
        low, high = math.floor(rank), math.ceil(rank)
        left = ordered[bisect_right(weights, low)][0]
        right = ordered[bisect_right(weights, high)][0]
        return left + (right - left) * (rank - low)

    return Distribution(
        sample_count=cumulative,
        mean=mean,
        std=math.sqrt(math.fsum((value - mean) ** 2 * weight for value, weight in ordered) / cumulative),
        min=ordered[0][0],
        max=ordered[-1][0],
        median=percentile(50),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
    )


class CdfPoint(BenchValue):
    """One empirical step with request or token-interval weight."""

    target_id: str = Field(description="Population label: full Model ID or aggregate.")
    metric: str
    sample_kind: str
    value_seconds: FiniteFloat = Field(ge=0)
    cumulative_probability: float
    weight: int


class ThroughputBucket(BenchValue):
    """Aligned common-window counts; the final bucket uses its actual width."""

    target_id: str = Field(description="Population label: full Model ID or aggregate.")
    start_seconds: FiniteFloat = Field(ge=0)
    end_seconds: FiniteFloat = Field(ge=0)
    width_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    partial_output_tokens: int = 0
    input_tokens_per_second: float = 0.0
    output_tokens_per_second: float = 0.0
    partial_output_tokens_per_second: float = 0.0


class TargetSummary(BenchValue):
    outcomes: dict[str, int]
    distributions: dict[str, Distribution]
    input_tokens: int
    output_tokens: int
    partial_output_tokens: int
    input_tokens_per_second: float | None
    output_tokens_per_second: float | None


class BenchSummary(BenchValue):
    """Recomputed measurement populations with separate execution and evidence facts."""

    latency_units: Literal["seconds"] = "seconds"
    case_id: CaseId
    execution_complete: bool
    measurement_available: bool
    cleanup_verified: bool | None = None
    evidence_complete: bool
    arrival_horizon_seconds: FiniteFloat = Field(ge=0)
    window_end_seconds: FiniteFloat | None = Field(
        ge=0, description="Common window end relative to T0; None when a usable window is unavailable."
    )
    window_kind: Literal["complete", "interrupted", "observed_prefix"] | None
    provisional_requests: int
    outcomes: dict[str, int]
    targets: dict[str, TargetSummary]
    cdf: tuple[CdfPoint, ...]
    throughput: tuple[ThroughputBucket, ...]
    infrastructure_error: str | None = None


def validate_request_identity(
    workload: PreparedWorkload, requests: Sequence[RequestRecord]
) -> dict[str, ScheduledRequest]:
    """Match unique retained identities and requested policy to their replay authority."""

    expected = {request.request_id: request for request in workload.requests}
    records = {request.request_id: request for request in requests}
    if len(records) != len(requests) or set(records) - expected.keys():
        raise ValueError("request records contain duplicate or unplanned IDs")
    for request in requests:
        if request.model_dump(include=set(ScheduledRequest.model_fields)) != expected[request.request_id].model_dump():
            raise ValueError("retained request identity differs from the prepared workload")
    return expected


def validate_measurement(
    workload: PreparedWorkload,
    requests: Sequence[RequestRecord],
    events: Sequence[StreamEvent],
    *,
    window_end_seconds: float | None,
    window_kind: Literal["complete", "interrupted", "observed_prefix"] | None,
) -> dict[str, list[StreamEvent]]:
    """Validate replay and raw lifecycle evidence without building statistics.

    An unavailable window supports only not-sent or unknown owner-loss outcomes.
    Timed observations must follow the declared arrival and fit the window.
    """

    expected = validate_request_identity(workload, requests)
    if (window_end_seconds is None) != (window_kind is None):
        raise ValueError("measurement window end and kind must be available together")
    if window_end_seconds is None:
        if events or any(
            request.outcome not in {"not_sent", "failed"}
            or (request.outcome == "failed" and request.error_kind != "evidence_missing")
            for request in requests
        ):
            raise ValueError("unavailable measurement window cannot support dispatched observations")
        for request in requests:
            request.validate_evidence(())
        return {}
    if not math.isfinite(window_end_seconds) or window_end_seconds <= 0:
        raise ValueError("timed measurement window must be positive and finite")
    by_request: dict[str, list[StreamEvent]] = defaultdict(list)
    for event in events:
        if event.request_id not in expected:
            raise ValueError("stream event refers to an unplanned request")
        by_request[event.request_id].append(event)
    for request_id, history in by_request.items():
        previous_time = expected[request_id].arrival_seconds
        for sequence, event in enumerate(history):
            if event.sequence != sequence:
                raise ValueError("stream event sequence must be contiguous from zero")
            if not previous_time <= event.observed_at_seconds <= window_end_seconds:
                raise ValueError("stream event timestamps must be chronological within the measurement window")
            previous_time = event.observed_at_seconds
            if (
                (sequence == 0 and event.kind != "enqueue")
                or (sequence == 1 and event.kind != "http_start")
                or (sequence > 1 and event.kind in {"enqueue", "http_start"})
                or (not event.accepted and event.kind not in {"data", "done"})
            ):
                raise ValueError("stream event lifecycle must start with enqueue then HTTP dispatch")
            if sequence:
                previous = history[sequence - 1]
                if (
                    previous.kind == "error"
                    or (previous.kind == "done" and previous.accepted)
                    or (not previous.accepted and event.kind != "error")
                ):
                    raise ValueError("stream event follows terminal or rejected native evidence")
    for request in requests:
        request.validate_evidence(by_request[request.request_id])
    if window_kind == "complete" and window_end_seconds < workload.arrival_horizon_seconds:
        raise ValueError("completed measurement window must cover the arrival horizon")
    return by_request


def summarize(
    workload: PreparedWorkload,
    requests: Sequence[RequestRecord],
    events: Sequence[StreamEvent],
    *,
    window_end_seconds: float,
    window_kind: Literal["complete", "interrupted", "observed_prefix"],
) -> BenchSummary:
    """Aggregate successful samples and separate failed partial output.

    The common monotonic window includes arrival idle time and queue drain.
    Output increments use observed chunk times; input includes logical cache-hit
    tokens and is attributed at successful request termination. Normalized
    intervals carry token weights, never artificially spaced token timestamps.
    Cleanup verification belongs to the outer runner.
    """

    by_request = validate_measurement(
        workload, requests, events, window_end_seconds=window_end_seconds, window_kind=window_kind
    )
    samples: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    cdf: list[CdfPoint] = []
    buckets: dict[str, list[list[int]]] = {
        id: [[0, 0, 0] for _ in range(math.ceil(window_end_seconds / workload.bucket_seconds))]
        for id in (*(str(model_id) for model_id in workload.model_ids), "aggregate")
    }

    def attribute(id: str, when: float, slot: int, tokens: int) -> None:
        index = min(int(when / workload.bucket_seconds), len(buckets[id]) - 1)
        for key in (id, "aggregate"):
            buckets[key][index][slot] += tokens

    for request in requests:
        target = str(request.model_id)
        request_events = by_request[request.request_id]
        success = request.outcome == "success"
        if success:
            for name, value in request.metrics.model_dump().items():
                if name.endswith("_seconds") or name == "itl_coverage":
                    if value is not None:
                        for key in (target, "aggregate"):
                            samples[key][name].append((value, 1))
            for interval in token_intervals(request_events):
                for key in (target, "aggregate"):
                    samples[key][f"itl_{interval.kind}"].append((interval.value_seconds, interval.weight))
                    samples[key]["itl_combined"].append((interval.value_seconds, interval.weight))
            attribute(target, cast(float, request.ended_at_seconds), 0, cast(int, request.prompt_tokens))
        previous = 0
        for event in request_events:
            if event.kind == "data" and event.accepted and event.completion_tokens is not None:
                increment = event.completion_tokens - previous
                attribute(target, event.observed_at_seconds, 1 if success else 2, increment)
                previous = event.completion_tokens
    throughput: list[ThroughputBucket] = []
    targets: dict[str, TargetSummary] = {}
    metric_names = (*RequestMetrics.model_fields.keys(), "itl_observed", "itl_estimated", "itl_combined")
    metric_names = tuple(name for name in metric_names if not name.endswith("sample_count"))
    for id, values in buckets.items():
        for index, (input_tokens, output_tokens, partial) in enumerate(values):
            start = index * workload.bucket_seconds
            end = min(start + workload.bucket_seconds, window_end_seconds)
            width = end - start
            throughput.append(
                ThroughputBucket(
                    target_id=id,
                    start_seconds=start,
                    end_seconds=end,
                    width_seconds=width,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    partial_output_tokens=partial,
                    input_tokens_per_second=input_tokens / width,
                    output_tokens_per_second=output_tokens / width,
                    partial_output_tokens_per_second=partial / width,
                )
            )
        target_requests = tuple(request for request in requests if id == "aggregate" or str(request.model_id) == id)
        outcomes = Counter(request.outcome for request in target_requests)
        distributions = {name: distribution(samples[id][name]) for name in metric_names}
        for metric, population in samples[id].items():
            if metric == "itl_coverage":
                continue
            counts: dict[float, int] = defaultdict(int)
            for value, weight in population:
                counts[value] += weight
            total = sum(counts.values())
            cumulative = 0
            for value, weight in sorted(counts.items()):
                cumulative += weight
                cdf.append(
                    CdfPoint(
                        target_id=id,
                        metric=metric,
                        sample_kind=metric.removeprefix("itl_") if metric.startswith("itl_") else "request",
                        value_seconds=value,
                        cumulative_probability=cumulative / total,
                        weight=weight,
                    )
                )
        input_total, output_total, partial_total = (sum(value[slot] for value in values) for slot in range(3))
        targets[id] = TargetSummary(
            outcomes={kind: outcomes[kind] for kind in ("success", "failed", "cancelled", "not_sent")},
            distributions=distributions,
            input_tokens=input_total,
            output_tokens=output_total,
            partial_output_tokens=partial_total,
            input_tokens_per_second=input_total / window_end_seconds,
            output_tokens_per_second=output_total / window_end_seconds,
        )
    complete = (
        window_kind == "complete"
        and len(requests) == len(workload.requests)
        and all(
            request.outcome in {"success", "failed"} and request.error_kind != "evidence_missing"
            for request in requests
        )
    )
    return BenchSummary(
        case_id=workload.case_id,
        execution_complete=complete,
        measurement_available=any(request.outcome == "success" for request in requests),
        evidence_complete=len(requests) == len(workload.requests)
        and all(request.error_kind != "evidence_missing" for request in requests),
        arrival_horizon_seconds=workload.arrival_horizon_seconds,
        window_end_seconds=window_end_seconds,
        window_kind=window_kind,
        provisional_requests=len(workload.requests) - len(requests),
        outcomes=targets["aggregate"].outcomes,
        targets=targets,
        cdf=tuple(cdf),
        throughput=tuple(throughput),
    )


def unavailable_summary(workload: PreparedWorkload, requests: Sequence[RequestRecord]) -> BenchSummary:
    """Retain unavailable-window/no-data outcomes without inventing elapsed time."""

    validate_measurement(workload, requests, (), window_end_seconds=None, window_kind=None)
    outcomes = Counter(request.outcome for request in requests)
    return BenchSummary(
        case_id=workload.case_id,
        execution_complete=not workload.requests,
        measurement_available=False,
        evidence_complete=len(requests) == len(workload.requests)
        and all(request.error_kind != "evidence_missing" for request in requests),
        arrival_horizon_seconds=workload.arrival_horizon_seconds,
        window_end_seconds=None,
        window_kind=None,
        provisional_requests=len(workload.requests) - len(requests),
        outcomes={kind: outcomes[kind] for kind in ("success", "failed", "cancelled", "not_sent")},
        targets={},
        cdf=(),
        throughput=(),
    )
