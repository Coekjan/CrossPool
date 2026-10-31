import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from tests.support.observability import FixtureData, events_for, fixture, metadata_for

from xpool.devkit.fabric_observer import FabricSnapshotJson
from xpool.observability import (
    ErrorCode,
    Metadata,
    ObservationError,
    SnapshotRequest,
    project_legacy_snapshots,
    reconstruct,
)
from xpool.observability.codec import validate_event


def membership_document(data: FixtureData) -> Metadata:
    return Metadata.parse_artifact(
        json.dumps(metadata_for(data)).encode(), Path("metadata.json"), ErrorCode.METADATA_UNAVAILABLE
    )


@pytest.mark.parametrize(
    "case",
    [case for case in fixture("identity-lifecycle")["cases"] if "membership_metadata" in case["input"]],
    ids=lambda case: case["id"],
)
def test_fixed_membership_contract(case: FixtureData) -> None:
    data, expected = case["input"], case["expected"]

    def check() -> None:
        document = membership_document(data)
        if "prior_metadata" in data:
            reconstruct((), metadata=(membership_document({}), document))
        if "membership_queries" in data:
            assert [
                document.pe_for(query["generation"], query["producer"]) for query in data["membership_queries"]
            ] == expected["pes"]
        events = events_for(data | {"events": data.get("events", [])})
        for event in events:
            validate_event(event)
        if "resolved_pes" in expected:
            assert [
                document.pe_for(event.record.generation, event.source.identity.producer) for event in events
            ] == expected["resolved_pes"]
        if "local_trace_ids" in expected:
            assert sorted({event.record.local_trace_id for event in events}) == expected["local_trace_ids"]
            assert {event.source.identity.producer for event in events} == {expected["producer"]}
            assert {document.pe_for(event.record.generation, event.source.identity.producer) for event in events} == {
                expected["pe"]
            }
        if "registered_process_roles" in expected:
            assert {producer.process.runtime_role for producer in document.producers} == set(
                expected["registered_process_roles"]
            )
            # Launcher allocation is outside this offline library. An empty inventory
            # cannot manufacture a Daemon source or a missing Daemon artifact.
            result = reconstruct((), metadata=(document,))
            assert {
                finding.producer for finding in result.quality.findings if finding.code == "process_artifact_missing"
            } == {producer.id for producer in document.producers}

    if "error" in expected:
        with pytest.raises(ObservationError) as caught:
            check()
        assert caught.value.code == expected["error"]
        for field in ("producer", "event_sequence"):
            if field in expected:
                assert getattr(caught.value, field) == expected[field]
        assert caught.value.metadata_field_path is not None or caught.value.sources
    else:
        check()


def test_absent_generation_and_nonmember_are_distinct_from_unresolved_reference() -> None:
    document = membership_document({})
    assert document.pe_for(None, 5) is None
    assert document.pe_for(1, 3) is None
    with pytest.raises(ObservationError) as caught:
        document.pe_for(99, 5)
    assert caught.value.code == ErrorCode.INVALID_EVENT
    assert (caught.value.generation, caught.value.producer) == (99, 5)


@pytest.mark.parametrize("producer,pe", [(2, 0), (5, 1)])
@pytest.mark.parametrize("attempts", [0, 2, None])
def test_snapshot_resolves_pe_without_events(producer: int, pe: int, attempts: int | None) -> None:
    document = membership_document({})
    result = reconstruct((), metadata=(document,))
    projection = project_legacy_snapshots(result, (SnapshotRequest(document.run_id, producer, 1, attempts),))[0]
    assert projection.evidence_report is not None
    assert projection.records == ()
    if attempts is None:
        assert projection.snapshot is None
    else:
        assert projection.snapshot is not None
        assert TypeAdapter(FabricSnapshotJson).validate_python(projection.snapshot)["pe"] == pe
        assert projection.snapshot["sequence"] == attempts
        assert projection.evidence_report.unobserved_operations == attempts


def test_duplicate_binding_precedes_coverage_failure() -> None:
    data = metadata_for({})
    data["generations"][0]["pe_members"] = [data["generations"][0]["pe_members"][0]] * 2
    with pytest.raises(ObservationError) as caught:
        Metadata.parse_artifact(json.dumps(data).encode(), Path("members.json"), ErrorCode.METADATA_UNAVAILABLE)
    assert caught.value.code == ErrorCode.IDENTITY_CONFLICT
    assert caught.value.metadata_field_path == "generations[0].pe_members[1].pe"
    assert caught.value.generation == 1
    assert caught.value.pe == 0


@pytest.mark.parametrize("field,value", [("pe", -1), ("pe", 2**31), ("device_producer", 0), ("extra", 1)])
def test_member_schema_failure_retains_metadata_context(field: str, value: int) -> None:
    data = metadata_for({})
    data["generations"][0]["pe_members"][0][field] = value
    with pytest.raises(ObservationError) as caught:
        Metadata.parse_artifact(json.dumps(data).encode(), Path("members.json"), ErrorCode.METADATA_UNAVAILABLE)
    error = caught.value
    assert error.code == ErrorCode.INVALID_EVENT
    assert error.generation == 1
    assert error.pe == (value if field == "pe" else 0)
    assert error.metadata_field_path == f"generations[0].pe_members[0].{field}"
    assert error.path == Path("members.json")
    assert error.__cause__ is not None


def test_snapshot_uses_generation_from_later_metadata_revision() -> None:
    original = membership_document({})
    later = original.model_dump()
    generation = dict(later["generations"][0], id=2, fabric_generation="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    later["generations"].append(generation)
    later["revision"] = 2
    result = reconstruct((), metadata=(original, Metadata.model_validate(later)))
    projection = project_legacy_snapshots(result, (SnapshotRequest(original.run_id, 5, 2, 0),))[0]
    assert projection.snapshot is not None
    snapshot = TypeAdapter(FabricSnapshotJson).validate_python(projection.snapshot)
    assert snapshot["generation"] == generation["fabric_generation"]
    assert snapshot["pe"] == 1
