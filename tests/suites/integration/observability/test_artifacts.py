"""Real published files connect strict parsing, reconstruction and attachments."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from safetensors import safe_open
from tests.support.observability import (
    FixtureData,
    events_for,
    expand,
    fixture,
    fixture_root,
    metadata_for,
    scenario,
)

from xpool.observability import (
    ArtifactInputs,
    AttachmentIndex,
    ChunkHeader,
    EntityIdentity,
    Metadata,
    ObservationError,
    RoutingAttachment,
    SnapshotRequest,
    encode_chunk,
    parse_chunk,
    project_legacy_snapshots,
    read_artifacts,
    reconstruct,
)
from xpool.observability.models import Artifact


def write_inventory(
    root: Path, data: FixtureData, *, artifacts: tuple[Artifact, ...] | None = None, revision: int = 1
) -> ArtifactInputs:
    events = events_for(data)
    chunks: list[Path] = []
    metadata_paths: dict[int, Path] = {}
    metadata_bytes: dict[int, bytes] = {}
    for sequence, event in enumerate(events, 1):
        producer = event.source.identity.producer
        if producer not in metadata_paths:
            metadata_path = root / f"metadata.{producer}.{revision}.json"
            metadata_paths[producer] = metadata_path
            document = event.metadata.model_copy(update={"revision": revision})
            if artifacts is not None:
                document = document.model_copy(update={"artifacts": list(artifacts)})
            metadata_bytes[producer] = document.model_dump_json().encode()
            metadata_path.write_bytes(metadata_bytes[producer])
        header = ChunkHeader(event.metadata.run_id, producer, sequence, revision, 0, 10000)
        chunk_path = root / f"{producer}.{sequence}.xpool"
        chunk_path.write_bytes(encode_chunk(header, (event.record,), metadata_bytes[producer]))
        chunks.append(chunk_path)
    if not events:
        path = root / "metadata.3.1.json"
        path.write_text(json.dumps(metadata_for(data)))
        metadata_paths[3] = path
    return ArtifactInputs(tuple(chunks), tuple(metadata_paths.values()))


@pytest.fixture
def routing_entry() -> RoutingAttachment:
    return RoutingAttachment.model_validate(fixture("routing")["cases"][0]["input"]["attachments"][0])


def write_referenced_routing_inventory(
    root: Path, entry: RoutingAttachment, *, reference: int = 2, revision: int = 1
) -> ArtifactInputs:
    data = copy.deepcopy(fixture("routing")["cases"][0]["input"])
    data["events"][0]["record"]["data2"] = reference
    artifact = Artifact(id=2, kind="routing", status="published", path=entry.path, sha256=entry.sha256)
    inputs = write_inventory(root, data, artifacts=(artifact,), revision=revision)
    (root / "routing.safetensors").write_bytes((fixture_root / "routing.safetensors").read_bytes())
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, (entry,), True)
    return ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=root)


@pytest.mark.parametrize("reference", [0, 2])
def test_routing_reference_resolves_the_matching_attachment(
    tmp_path: Path, routing_entry: RoutingAttachment, reference: int
) -> None:
    inputs = write_referenced_routing_inventory(tmp_path, routing_entry, reference=reference)
    result = reconstruct(read_artifacts(inputs))
    evidence = result.attachments[0]
    assert (evidence.run_id, evidence.owner, evidence.sequence) == (fixture("metadata")["run_id"], 5, 1)
    assert evidence.available and evidence.status == "published"
    assert evidence.path == tmp_path / "routing.safetensors"
    assert not any(code.startswith("attachment_") for code in result.quality.codes)


def test_routing_index_pe_must_match_publisher_membership(tmp_path: Path, routing_entry: RoutingAttachment) -> None:
    data = {"events": [], "metadata_overrides": scenario("invocation_multiple_graph_owners")["metadata_overrides"]}
    inputs = write_inventory(tmp_path, data)
    entry = routing_entry.model_copy(update={"pe": 2})
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, (entry,), True)
    with pytest.raises(ObservationError) as caught:
        read_artifacts(
            ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
        )
    assert caught.value.code == "identity_conflict"
    assert caught.value.producer == 5
    assert caught.value.generation == 1
    assert caught.value.pe == 2
    assert caught.value.metadata_field_path == "generations[0].pe_members"


@pytest.mark.parametrize("conflict", ["path", "sha256", "status"])
def test_routing_reference_must_match_the_final_index(
    tmp_path: Path, routing_entry: RoutingAttachment, conflict: str
) -> None:
    inputs = write_referenced_routing_inventory(tmp_path, routing_entry)
    changes_by_conflict: dict[str, dict[str, str | None]] = {
        "path": {"path": "other.safetensors"},
        "sha256": {"sha256": "0" * 64},
        "status": {"status": "publish_failed", "path": None, "sha256": None},
    }
    changes = changes_by_conflict[conflict]
    entry = RoutingAttachment.model_validate(routing_entry.model_dump() | changes)
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, (entry,), True)
    with pytest.raises(ObservationError) as caught:
        read_artifacts(
            ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
        )
    assert caught.value.code == "identity_conflict"
    assert caught.value.path == inputs.chunks[0]
    assert caught.value.offset == 128
    assert caught.value.producer == 5
    assert caught.value.event_sequence == 10


@pytest.mark.parametrize("reverse_metadata", [False, True])
def test_routing_reference_uses_the_events_metadata_revision(
    tmp_path: Path, routing_entry: RoutingAttachment, reverse_metadata: bool
) -> None:
    inputs = write_referenced_routing_inventory(tmp_path, routing_entry, revision=2)
    data = json.loads(inputs.metadata[0].read_bytes())
    data["revision"] = 1
    data["artifacts"] = []
    earlier = tmp_path / "metadata.5.1.json"
    earlier.write_text(json.dumps(data))
    metadata = (earlier, *inputs.metadata) if reverse_metadata else (*inputs.metadata, earlier)
    result = reconstruct(
        read_artifacts(
            ArtifactInputs(
                inputs.chunks, metadata, attachment_indices=inputs.attachment_indices, attachment_root=tmp_path
            )
        )
    )
    assert result.attachments[0].available
    assert result.attachments[0].path == tmp_path / "routing.safetensors"


@pytest.mark.parametrize("reverse_metadata", [False, True])
def test_routing_reference_does_not_allow_artifact_redefinition(
    tmp_path: Path, routing_entry: RoutingAttachment, reverse_metadata: bool
) -> None:
    inputs = write_referenced_routing_inventory(tmp_path, routing_entry)
    data = json.loads(inputs.metadata[0].read_bytes())
    data["revision"] = 2
    data["artifacts"][0]["path"] = "other.safetensors"
    later = tmp_path / "metadata.5.2.json"
    later.write_text(json.dumps(data))
    metadata = (later, *inputs.metadata) if reverse_metadata else (*inputs.metadata, later)
    artifacts = read_artifacts(
        ArtifactInputs(inputs.chunks, metadata, attachment_indices=inputs.attachment_indices, attachment_root=tmp_path)
    )
    assert artifacts.attachments[0].available
    with pytest.raises(ObservationError) as caught:
        reconstruct(artifacts)
    assert caught.value.code == "identity_conflict"
    assert caught.value.producer == 5


@pytest.mark.parametrize("closing_complete,code", [(False, "tail_unknown"), (True, "attachment_missing")])
def test_routing_reference_does_not_replace_the_final_index(
    tmp_path: Path, routing_entry: RoutingAttachment, closing_complete: bool, code: str
) -> None:
    inputs = write_referenced_routing_inventory(tmp_path, routing_entry)
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, (), closing_complete)
    result = reconstruct(
        read_artifacts(
            ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
        )
    )
    assert not result.attachments[0].available
    assert result.attachments[0].entry is None
    assert result.attachments[0].status == ("missing" if closing_complete else "unknown")
    assert any(
        finding.code == code and finding.sources == result.attachments[0].sources for finding in result.quality.findings
    )


def test_routing_reference_keeps_missing_files_explicit(tmp_path: Path, routing_entry: RoutingAttachment) -> None:
    inputs = write_referenced_routing_inventory(tmp_path, routing_entry)
    (tmp_path / "routing.safetensors").unlink()
    result = reconstruct(read_artifacts(inputs))
    assert not result.attachments[0].available
    assert result.attachments[0].status == "missing"
    assert "attachment_missing" in result.quality.codes


@pytest.mark.parametrize("case", fixture("routing")["cases"], ids=lambda case: case["id"])
def test_routing_frozen_artifacts(tmp_path: Path, case: FixtureData) -> None:
    data = copy.deepcopy(case["input"])
    inputs = write_inventory(tmp_path, data)
    target = tmp_path / "routing.safetensors"
    if case["id"] != "routing_missing":
        target.write_bytes((fixture_root / "routing.safetensors").read_bytes())
    entries = TypeAdapter(list[RoutingAttachment]).validate_python(data.get("attachments", []))
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, tuple(entries), data.get("closing_index_complete", False))
    inputs = ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
    artifacts = read_artifacts(inputs)
    result = reconstruct(artifacts)
    expected = case["expected"]
    assert len(result.attachments) == 1
    evidence = result.attachments[0]
    assert evidence.available == expected.get("attachment_available", False)
    assert bool(evidence.sources) == expected.get("event_available", bool(data.get("events")))
    if "attachment_status" in expected:
        assert evidence.status == expected["attachment_status"]
    assert set(expected.get("quality", [])) <= result.quality.codes
    if evidence.available:
        with safe_open(target, framework="numpy") as handle:
            assert handle.get_tensor("records.0.topk_ids").tolist() == [[0, 1], [1, 0]]
            assert handle.get_tensor("records.0.topk_weights").tolist() == [[0.75, 0.25], [0.5, 0.5]]
    if not expected.get("event_available", True):
        assert not any(event.record.kind == 550 for event in result.events.values())


def test_published_files_out_of_order_duplicates_and_provenance(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    inputs = ArtifactInputs((*reversed(inputs.chunks), inputs.chunks[0]), inputs.metadata)
    result = reconstruct(read_artifacts(inputs))
    assert len(result.of_kind("operation")) == len(result.of_kind("invocation")) == 1
    assert "duplicate_event" in result.quality.codes
    assert any(len(sources) == 2 for sources in result.event_sources.values())
    assert all(
        source.path in inputs.chunks and source.offset == 128 for edge in result.edges for source in edge.sources
    )


@pytest.mark.parametrize(
    "case_id", ["execution_never_returns", "execution_returns_failure", "invocation_multiple_graph_owners"]
)
def test_contract_v4_evidence_survives_file_round_trip(tmp_path: Path, case_id: str) -> None:
    result = reconstruct(read_artifacts(write_inventory(tmp_path, scenario(case_id))))
    expected = next(case["expected"] for case in fixture("causality")["cases"] if case["id"] == case_id)
    assert all(document.contract_version == 4 for document in result.metadata)
    relation = "invocation_to_graph_activity" if "graph_owners" in expected else "operation_to_invocation"
    actual = [
        [list(edge.origin.components), list(edge.destination.components)]
        for edge in result.edges
        if edge.relation == relation
        and isinstance(edge.origin, EntityIdentity)
        and isinstance(edge.destination, EntityIdentity)
    ]
    assert actual == expected[relation]
    assert set(expected.get("quality", [])) <= result.quality.codes


@pytest.mark.parametrize("failure", ["checksum", "path_escape", "identity"])
def test_routing_integrity_failures_are_structured(tmp_path: Path, failure: str) -> None:
    data = copy.deepcopy(fixture("routing")["cases"][0]["input"])
    inputs = write_inventory(tmp_path, data)
    target = tmp_path / "routing.safetensors"
    target.write_bytes((fixture_root / "routing.safetensors").read_bytes())
    entry = data["attachments"][0]
    expected = "payload_checksum"
    if failure == "checksum":
        target.write_bytes(target.read_bytes() + b"corrupt")
    elif failure == "path_escape":
        entry["path"] = "../routing.safetensors"
        expected = "invalid_event"
    else:
        entry["lease_sequence"] += 1
        expected = "identity_conflict"
    index = AttachmentIndex(fixture("metadata")["run_id"], 5, (RoutingAttachment.model_validate(entry),), True)
    with pytest.raises(ObservationError) as caught:
        read_artifacts(
            ArtifactInputs(inputs.chunks, inputs.metadata, attachment_indices=(index,), attachment_root=tmp_path)
        )
    assert caught.value.code == expected


def test_quality_files_and_missing_required_metadata(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    quality_path = tmp_path / "quality.3.1.json"
    checkpoint = fixture("quality")["checkpoint"]
    checkpoint["counters"].update(attempted=6, committed=6, copied=6, exported=6, dropped_full=0, abandoned=0)
    checkpoint["last_attempted_sequence"] = 6
    quality_path.write_text(json.dumps(checkpoint))
    result = reconstruct(read_artifacts(ArtifactInputs(inputs.chunks, inputs.metadata, (quality_path,))))
    assert result.quality.checkpoints[(fixture("metadata")["run_id"], 3)].counters.attempted == 6
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs(inputs.chunks, ()))
    assert caught.value.code == "metadata_unavailable"


def test_original_artifacts_are_unchanged(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in (*inputs.chunks, *inputs.metadata)}
    reconstruct(read_artifacts(inputs))
    assert before == {path: hashlib.sha256(path.read_bytes()).digest() for path in before}


def test_io_errors_keep_the_original_cause_and_temporary_files_are_not_published(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs((), (missing,)))
    assert caught.value.path == missing
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    temporary = tmp_path / "1.xpool.tmp"
    temporary.write_bytes(inputs.chunks[0].read_bytes())
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs((temporary,), inputs.metadata))
    assert caught.value.code == "invalid_layout"


def test_conflicting_chunk_publication_is_not_deduplicated(tmp_path: Path) -> None:
    inputs = write_inventory(tmp_path, scenario("normal_dense"))
    duplicate = tmp_path / "conflicting.xpool"
    original = read_artifacts(inputs).chunks[0]
    producer = original.header.producer
    metadata_path = next(path for path in inputs.metadata if path.name == f"metadata.{producer}.1.json")
    metadata = metadata_path.read_bytes()
    changed = original.events[0].record.model_copy(update={"raw_time_ns": 11})
    duplicate.write_bytes(encode_chunk(original.header, (changed,), metadata))
    assert parse_chunk(duplicate.read_bytes(), metadata).events[0].record.raw_time_ns == 11
    with pytest.raises(ObservationError) as caught:
        read_artifacts(ArtifactInputs((*inputs.chunks, duplicate), inputs.metadata))
    assert caught.value.code == "identity_conflict"


@pytest.mark.parametrize(
    "case_id",
    [
        "opening_event_lost_scalar_facts_recovered",
        "auxiliary_evidence_missing_required_scalars",
        "unknown_tail_keeps_counts_unknown",
        "unobserved_fabric_id_does_not_infer_role",
    ],
)
def test_v4_snapshot_reports_survive_published_file_round_trip(tmp_path: Path, case_id: str) -> None:
    case = next(c for c in fixture("snapshots")["cases"] if c["id"] == case_id)
    data = expand(case["input"])
    inputs = write_inventory(tmp_path, data)
    # Preserve the reference metadata revision explicitly provided by the fixture.
    metadata = Metadata.model_validate(metadata_for(data))
    reference = tmp_path / "reference-metadata.json"
    reference.write_text(metadata.model_dump_json())
    inventory = read_artifacts(ArtifactInputs(tuple(reversed(inputs.chunks)), (reference, *inputs.metadata)))
    selection, counts = data["select"], data["operation_counts"]
    request = SnapshotRequest(
        metadata.run_id,
        selection["producer"],
        1,
        counts["sequence"],
        counts["capacity"],
        (2, 1) if selection["observer"] == "transport" else None,
    )
    projection = project_legacy_snapshots(reconstruct(inventory), (request,))[0]
    expected = case["expected"]
    assert projection.evidence_report is not None
    report = projection.evidence_report.model_dump(mode="json")
    if "evidence_report" in expected:
        assert report == expected["evidence_report"]
    else:
        assert (
            next(
                entry
                for entry in report["entries"]
                if entry["local_trace_id"] == expected["evidence_entry"]["local_trace_id"]
            )
            == expected["evidence_entry"]
        )
    if "legacy_record" in expected:
        local_id = expected.get("evidence_entry", {}).get("local_trace_id")
        records = [
            record
            for record in projection.records
            if local_id is None or record.get("trace_id", record.get("local_trace_id")) == local_id
        ]
        assert records == ([] if expected["legacy_record"] is None else [expected["legacy_record"]])
    else:
        assert list(projection.records) == expected["legacy_records"]
    assert all(e.source.path in inputs.chunks for e in inventory.events)
