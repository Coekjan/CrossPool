"""Isolation and conflicting identities are enforced at public reconstruction boundaries."""

import copy
from dataclasses import replace

import pytest
from tests.support.observability import FixtureData, events_for, fixture, scenario

from xpool.observability import ErrorCode, Metadata, ObservationError, reconstruct


def test_run_isolation_preserves_identical_local_ids() -> None:
    data = scenario("normal_dense")
    original = events_for(data)
    other_run = "99999999-9999-4999-8999-999999999999"
    other = tuple(
        replace(
            event,
            source=replace(event.source, identity=replace(event.source.identity, run_id=other_run)),
            metadata=event.metadata.model_copy(update={"run_id": other_run}),
        )
        for event in original
    )
    result = reconstruct((*original, *other))
    assert len(result.of_kind("operation")) == 2
    assert len(result.of_kind("invocation")) == 2
    assert all(edge.origin.run_id == edge.destination.run_id for edge in result.edges)


def test_generation_recreation_never_merges_invocations_or_leases() -> None:
    data = scenario("normal_dense")
    second = copy.deepcopy(data)
    metadata = fixture("metadata")
    generation = copy.deepcopy(metadata["generations"][0])
    generation.update(id=2, fabric_generation="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    for item in second["events"]:
        if item["record"]["generation"]:
            item["record"]["generation"] = 2
        item["record"]["event_sequence"] += 100
        if item["record"]["endpoint_sequence"]:
            item["record"]["endpoint_sequence"] = 2
        if item["record"]["graph_execution"]:
            item["record"]["graph_execution"] += 1
    endpoint = copy.deepcopy(metadata["endpoints"][0])
    endpoint.update(endpoint_sequence=2, generation=2, supersedes_endpoint_sequence=1)
    # New event metadata must include both generations and Endpoint identities.
    base_events = events_for(data)
    new_events = events_for(second)
    documents = []
    for event in (*base_events, *new_events):
        raw = event.metadata.model_dump()
        raw["generations"].append(generation)
        raw["endpoints"].append(endpoint)
        documents.append(Metadata.model_validate(raw))
    # The original and new traces also reuse template lifecycle boundaries;
    # keep only Fabric events here to isolate Generation semantics.
    events = tuple(
        replace(event, metadata=document)
        for event, document in zip((*base_events, *new_events), documents, strict=True)
        if 200 <= event.record.kind <= 408
    )
    result = reconstruct(events)
    assert len(result.of_kind("invocation")) == len(result.of_kind("lease")) == 2


def test_same_pid_new_producer_and_process_are_distinct() -> None:
    event = next(event for event in events_for(scenario("normal_dense")) if event.record.kind == 100)
    raw = event.metadata.model_dump()
    process = raw["process"] | {"creation_id": "66666666-6666-4666-8666-666666666666"}
    domain = copy.deepcopy(next(domain for domain in raw["clock_domains"] if domain["id"] == 2))
    domain.update(id=6, process_creation_id=process["creation_id"])
    producer = {
        "id": 6,
        "source": "device",
        "process": process,
        "process_creation_id": process["creation_id"],
        "clock_domain_id": 6,
    }
    raw["producers"].append(producer)
    raw["clock_domains"].append(domain)
    raw["endpoints"][0]["sites"].append({"producer": 6, "site": "instance"})
    shared = Metadata.model_validate(raw)
    other_metadata = Metadata.model_validate(raw | {"producer_id": 6, "process": process, "clock_domain_id": 6})
    first = replace(event, metadata=shared)
    second = replace(
        event,
        metadata=other_metadata,
        record=event.record.model_copy(update={"clock_domain_id": 6}),
        source=replace(
            event.source,
            identity=replace(event.source.identity, producer=6),
            process_creation_id=process["creation_id"],
            clock_domain_id=6,
        ),
    )
    result = reconstruct((first, second))
    assert len(result.events) == 2
    assert len(result.of_kind("operation")[0].observations) == 2


def test_changed_process_creation_for_same_producer_is_a_conflict() -> None:
    event = events_for(scenario("normal_dense"))[0]
    raw = event.metadata.model_dump()
    raw["revision"] = 2
    raw["producers"][0]["process"]["pid"] += 1
    changed = Metadata.model_validate(raw | {"process": raw["producers"][0]["process"]})
    with pytest.raises(ObservationError) as caught:
        reconstruct((event,), metadata=(changed,))
    assert caught.value.code == ErrorCode.IDENTITY_CONFLICT


def test_fabric_entry_does_not_prove_submission_publication() -> None:
    result = reconstruct(events_for(scenario("missing_publication")))
    assert any(edge.relation == "operation_to_invocation" for edge in result.edges)
    assert not any(event.record.kind == 201 for event in result.events.values())
    assert "event_missing" in result.quality.codes


@pytest.mark.parametrize("case", fixture("calibration")["wire_sample_cases"], ids=lambda case: case["id"])
def test_sample_input_preserves_failed_read_as_absent(case: FixtureData) -> None:
    events = events_for({"events": [case["input"]["event"]]})
    result = reconstruct(events)
    sample = result.calibration_samples[0]
    expected = case["expected"]
    assert sample.attempt_sequence == expected["attempt_sequence"]
    assert sample.host_before_ns == expected["H_before"]
    assert sample.device_time_ns == expected["G"]
    assert sample.host_after_ns == expected["H_after"]
    assert sample.outcome == expected["outcome"]
    assert sample.failure_code == expected.get("failure_code")
    assert [sample.source.identity.run_id, sample.source.identity.producer, sample.device_clock_domain_id] == expected[
        "sampler"
    ]
