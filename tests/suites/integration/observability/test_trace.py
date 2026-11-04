"""Real pinned Trace Processor acceptance; run xpool trace setup before this suite."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace
from tests.support.observability import calibration_events, events_for, fixture, fixture_root, scenario
from tests.support.trace import publish_inputs

from xpool.cli.main import main
from xpool.config import XpoolConfig
from xpool.observability import (
    QualityCheckpoint,
    Sampler,
    SamplerDeclaration,
    build_calibration,
    reconstruct,
)
from xpool.observability.models import Artifact, RoutingAttachment
from xpool.observability.trace import (
    QueryName,
    TraceError,
    TraceErrorCode,
    TraceInputs,
    export_traces,
    load_inputs,
    query_bundle,
    verify_processor,
    write_queries,
)
from xpool.observability.trace.models import AttachmentInput


@pytest.fixture(scope="module")
def processor() -> Path:
    path = XpoolConfig.resolve_cache_root()[0] / "perfetto/trace_processor_shell"
    try:
        return verify_processor(path)
    except TraceError as error:
        if error.code == TraceErrorCode.DEPENDENCY_MISSING:
            pytest.skip("run uv run xpool trace setup before Trace Processor integration tests")
        raise


def test_processor_queries_all_families_and_raw_sources(tmp_path: Path, processor: Path) -> None:
    data = scenario("normal_dense")
    routing = fixture("routing")["cases"][1]["input"]
    addition = routing["events"][0]
    addition["record"]["event_sequence"] = 1000
    data["events"].append(addition)
    data["capabilities"] = {"routing": "enabled"}
    view = reconstruct(events_for(data))
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    result = query_bundle(output, processor)
    categories = {row[1] for item in result for row in item.rows}
    assert {"graph", "transport", "fabric", "routing", "lease", "invocation", "quality", "calibration"} <= categories
    identities = [row[0] for item in result if item.query == QueryName.INVOCATION for row in item.rows]
    assert len(identities) == len(set(identities))
    events = [row[2] for item in result for row in item.rows if isinstance(row[2], dict) and "record" in row[2]]
    assert events
    for event in events:
        assert isinstance(event, dict)
        assert event["sources"] and event["unit"] == "ns"
    flows = query_bundle(output, processor, sql="SELECT count(*) AS n FROM flow")
    counts = [row[0] for item in flows for row in item.rows]
    assert all(isinstance(count, int) for count in counts)
    assert sum(count for count in counts if isinstance(count, int)) > 0
    routing_rows = [row[2] for item in result if item.query == QueryName.ROUTING for row in item.rows]
    assert any(isinstance(payload, dict) and payload.get("status") == "not_collected" for payload in routing_rows)


@pytest.mark.parametrize("case", ["known_offset", "known_rate", "asymmetric_windows"])
def test_a2_mapping_matches_processor_axis_and_snapshots(tmp_path: Path, processor: Path, case: str) -> None:
    data = next(case_data["input"] for case_data in fixture("calibration")["cases"] if case_data["id"] == case)
    samples = calibration_events(data)
    original = tuple(event for event in events_for(scenario("normal_dense")) if event.record.clock_domain_id == 2)[:2]
    device = tuple(
        replace(event, record=event.record.model_copy(update={"raw_time_ns": timestamp}))
        for event, timestamp in zip(original, (55, 65), strict=True)
    )
    view = reconstruct((*samples, *device))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
        observation_interval=(0, 200),
    )
    index = build_calibration(view, (declaration,))
    output = tmp_path / "bundle"
    bundle = export_traces(view, index, output)
    unified = next(member for member in bundle.traces if member.kind == "unified")
    trace = Trace.FromString((output / unified.path).read_bytes())
    snapshots = [packet.clock_snapshot for packet in trace.packet if packet.HasField("clock_snapshot")]
    assert len(snapshots) == 2
    for snapshot in snapshots:
        raw = next(clock.timestamp for clock in snapshot.clocks if clock.clock_id == 64)
        displayed = next(clock.timestamp for clock in snapshot.clocks if clock.clock_id == 11)
        assert displayed + unified.origin_ns == index.map_raw(data["run_id"], 2, raw).estimate_ns
    results = query_bundle(
        output,
        processor,
        sql=(
            "SELECT ts, payload FROM xpool_evidence WHERE category = 'transport' "
            "AND name != 'xpool.interval' ORDER BY ts"
        ),
    )
    rows = next(item.rows for item in results if item.trace == unified.path)
    assert len(rows) == 2
    for timestamp, payload in rows:
        event = json.loads(str(payload))
        assert isinstance(timestamp, int)
        assert timestamp + unified.origin_ns == event["mapping"]["estimate_ns"]
        assert event["mapping"]["model_id"] is not None
    intervals = query_bundle(output, processor, names=(QueryName.TRANSPORT,))
    payloads = [row[2] for item in intervals for row in item.rows if isinstance(row[2], dict) and "duration" in row[2]]
    assert any(
        isinstance(payload, dict) and payload["duration"]["status"] == "estimated"
        for payload in payloads
        if payload["duration"]
    )


def test_cli_paths_input_preservation_and_offline_queries(
    tmp_path: Path, processor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    raw = tmp_path / "raw"
    manifest = publish_inputs(raw, scenario("normal_dense"))
    before = {path: path.read_bytes() for path in raw.iterdir()}
    arguments = ["--input-dir", str(raw), "--manifest", str(manifest)]
    assert main(["trace", "validate", *arguments, "--output-dir", str(tmp_path / "validation")]) == 0
    bundle = tmp_path / "bundle"
    assert main(["trace", "export", *arguments, "--output-dir", str(bundle)]) == 0
    query = tmp_path / "queries.sql"
    query.write_text("SELECT 9223372036854775807 AS exact_integer, NULL AS missing")
    assert (
        main(
            [
                "trace",
                "query",
                "--input-dir",
                str(bundle),
                "--output-dir",
                str(tmp_path / "result"),
                "--processor",
                str(processor),
                "--sql",
                str(query),
            ]
        )
        == 0
    )
    results = json.loads((tmp_path / "result/query.json").read_text())
    assert all(item["rows"] == [[9223372036854775807, None]] for item in results)
    assert before == {path: path.read_bytes() for path in raw.iterdir()}


def test_user_sql_error_keeps_query_context_and_no_complete_result(tmp_path: Path, processor: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    sql = tmp_path / "bad.sql"
    sql.write_text("SELECT * FROM nonexistent_table")
    with pytest.raises(TraceError) as caught:
        write_queries(output, processor, tmp_path / "results", sql_file=sql)
    assert caught.value.code == TraceErrorCode.QUERY
    assert caught.value.query == "sql"
    assert caught.value.__cause__ is not None
    assert not (tmp_path / "results/query.json").exists()


def test_overlapping_slices_are_never_nested(tmp_path: Path, processor: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    interval = next(item for item in view.intervals if item.status == "complete" and item.begin and item.end)
    second = replace(
        interval,
        observation=replace(interval.observation, entity=replace(interval.observation.entity, components=(999,))),
    )
    view = replace(view, intervals=(interval, second))
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    results = query_bundle(output, processor, sql="SELECT depth, dur FROM slice WHERE name = 'xpool.interval'")
    rows = [row for item in results for row in item.rows]
    assert rows
    for depth, duration in rows:
        assert isinstance(duration, int)
        assert depth == 0 and duration >= 0


def test_unavailable_cross_domain_duration_keeps_identity(tmp_path: Path, processor: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    results = query_bundle(output, processor, names=(QueryName.TRANSPORT,))
    relations = [row[2] for item in results for row in item.rows if row[1] == "protocol"]
    assert relations
    for payload in relations:
        assert isinstance(payload, dict)
        duration, edge = payload["duration"], payload["edge"]
        assert isinstance(duration, dict) and duration["status"] == "unavailable"
        assert isinstance(edge, dict) and edge["sources"]


def test_query_does_not_use_upstream_download_path(
    tmp_path: Path, processor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_download(*args: object, **kwargs: object) -> None:
        pytest.fail("query attempted an implicit tool download")

    monkeypatch.setattr("perfetto.trace_processor.platform.get_perfetto_prebuilt", forbidden_download)
    view = reconstruct(events_for(scenario("normal_dense")))
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    assert query_bundle(output, processor, sql="SELECT 1 AS n")


def test_timestamp_proximity_never_creates_flows(tmp_path: Path, processor: Path) -> None:
    events = tuple(
        replace(event, record=event.record.model_copy(update={"raw_time_ns": 100}))
        for event in events_for(scenario("normal_dense"))
    )
    view = replace(reconstruct(events), edges=())
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    results = query_bundle(output, processor, sql="SELECT count(*) AS n FROM flow")
    assert all(item.rows == ((0,),) for item in results)


def test_same_timestamp_protocol_flow_keeps_causal_packet_order(tmp_path: Path, processor: Path) -> None:
    events = tuple(
        replace(event, record=event.record.model_copy(update={"raw_time_ns": 100}))
        for event in events_for(scenario("normal_dense"))
    )
    view = reconstruct(events)
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    results = query_bundle(output, processor, sql="SELECT count(*) AS n FROM flow")
    assert any(item.rows[0][0] != 0 for item in results)


def test_slice_crossing_calibration_gap_remains_flagged(tmp_path: Path, processor: Path) -> None:
    data = fixture("calibration")["cases"][0]["input"]
    samples = data["samples"]
    data["samples"] = samples + [
        sample
        | {
            "attempt_sequence": sample["attempt_sequence"] + 10,
            "G": sample["G"] + 200,
            "H_before": sample["H_before"] + 200,
            "H_after": sample["H_after"] + 200,
        }
        for sample in samples
    ]
    original = tuple(event for event in events_for(scenario("normal_dense")) if event.record.clock_domain_id == 2)[:2]
    device = tuple(
        replace(event, record=event.record.model_copy(update={"raw_time_ns": timestamp}))
        for event, timestamp in zip(original, (55, 255), strict=True)
    )
    view = reconstruct((*calibration_events(data), *device))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 20),
        observation_interval=(0, 400),
    )
    index = build_calibration(view, (declaration,))
    assert index.map_raw(data["run_id"], 2, 55).estimate_ns is not None
    assert index.map_raw(data["run_id"], 2, 255).estimate_ns is not None
    output = tmp_path / "bundle"
    bundle = export_traces(view, index, output)
    unified = next(member for member in bundle.traces if member.kind == "unified")
    results = query_bundle(
        output, processor, sql=("SELECT dur, payload FROM xpool_evidence WHERE name = 'xpool.interval' AND dur > 0")
    )
    rows = next(item.rows for item in results if item.trace == unified.path)
    assert len(rows) == 1 and rows[0][0] == 200
    payload = json.loads(str(rows[0][1]))
    assert payload["calibration_coverage"] == "incomplete"
    assert payload["duration"]["status"] == "estimated"


def test_json_identity_queries_and_empty_results(tmp_path: Path, processor: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    output = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), output)
    results = query_bundle(
        output,
        processor,
        sql=(
            "SELECT json_extract(payload, '$.identity.run_id') AS run_id "
            "FROM xpool_evidence WHERE name = 'xpool.entity' AND category = 'invocation'"
        ),
    )
    assert any(item.rows == ((view.metadata[0].run_id,),) for item in results)
    empty = query_bundle(output, processor, sql="SELECT id FROM slice WHERE 0")
    assert all(item.columns == ("id",) and item.rows == () for item in empty)


@pytest.mark.parametrize("count", [100, 2**53 + 1, 2**63])
def test_checkpoint_counters_keep_exact_counts_and_host_time(tmp_path: Path, processor: Path, count: int) -> None:
    data = fixture("quality")["checkpoint"]
    data.update(collection_state="active", export_state="active", tail="pending", last_attempted_sequence=count)
    data["counters"] = {key: 0 for key in data["counters"]}
    data["counters"].update(attempted=count, dropped_full=count)
    checkpoint = QualityCheckpoint.model_validate(data)
    view = reconstruct(events_for(scenario("normal_dense")), checkpoints=(checkpoint,))
    output = tmp_path / "bundle"
    bundle = export_traces(view, build_calibration(view, ()), output)
    results = query_bundle(
        output,
        processor,
        sql=(
            "SELECT c.ts, c.value FROM counter c JOIN counter_track t ON c.track_id = t.id "
            "WHERE t.name = 'producer 3/counter/attempted'"
        ),
    )
    for item in results:
        member = next(member for member in bundle.traces if member.path == item.trace)
        for timestamp, value in item.rows:
            assert isinstance(timestamp, int)
            assert timestamp + member.origin_ns == checkpoint.host_time_ns
            assert value == count
    if count <= 2**53 - 1:
        assert any(item.rows for item in results)
    else:
        assert not any(item.rows for item in results)
        quality = query_bundle(output, processor, names=(QueryName.QUALITY,))
        assert any(
            isinstance(row[2], dict) and row[2].get("reason") == "outside_exact_counter_range"
            for item in quality
            for row in item.rows
        )


def test_published_routing_tensor_remains_a_verified_reference(tmp_path: Path, processor: Path) -> None:
    data = fixture("routing")["cases"][0]["input"]
    entry = RoutingAttachment.model_validate(data["attachments"][0])
    data["events"][0]["record"]["data2"] = 2
    root = tmp_path / "raw"
    manifest = publish_inputs(
        root,
        data,
        artifacts=(Artifact(id=2, kind="routing", status="published", path=entry.path, sha256=entry.sha256),),
    )
    tensor_bytes = (fixture_root / "routing.safetensors").read_bytes()
    (root / "routing.safetensors").write_bytes(tensor_bytes)
    declaration = TraceInputs.model_validate_json(manifest.read_bytes()).model_copy(
        update={
            "attachment_indices": (
                AttachmentInput(run_id=fixture("metadata")["run_id"], owner=5, entries=(entry,), closing_complete=True),
            ),
        }
    )
    manifest.write_text(declaration.model_dump_json())
    view, index, digests = load_inputs(root, manifest)
    output = tmp_path / "bundle"
    export_traces(view, index, output, protected=(root,), inputs=digests)
    results = query_bundle(output, processor, names=(QueryName.ROUTING,))
    assert any(
        isinstance(row[2], dict)
        and row[2].get("available") is True
        and row[2].get("path") == str(root / "routing.safetensors")
        for item in results
        for row in item.rows
    )
    assert (root / "routing.safetensors").read_bytes() == tensor_bytes
    assert not list(output.glob("*.safetensors"))


def test_documented_bundle_and_sql_examples(processor: Path) -> None:
    examples = Path(__file__).resolve().parents[4] / "docs/examples/observability"
    for name in ("identities", "durations"):
        results = query_bundle(examples / "bundle", processor, sql=(examples / f"{name}.sql").read_text())
        assert any(result.rows for result in results)


def test_query_cli_reports_missing_processor_without_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    bundle = tmp_path / "bundle"
    export_traces(view, build_calibration(view, ()), bundle)
    assert (
        main(
            [
                "trace",
                "query",
                "--input-dir",
                str(bundle),
                "--output-dir",
                str(tmp_path / "results"),
                "--processor",
                str(tmp_path / "missing"),
            ]
        )
        == 2
    )
    assert "dependency_missing" in capsys.readouterr().err
    assert not (tmp_path / "results/query.json").exists()
