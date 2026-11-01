import json
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent
from tests.support.observability import calibration_events, events_for, fixture, scenario
from tests.support.trace import publish_inputs

from xpool.observability import Sampler, SamplerDeclaration, build_calibration, reconstruct
from xpool.observability.trace import TraceError, TraceErrorCode, export_traces, load_inputs, verify_processor, versions
from xpool.observability.trace.export import Timeline
from xpool.observability.trace.inputs import confined, prepare_output
from xpool.observability.trace.models import TraceBundle
from xpool.observability.trace.query import read_bundle
from xpool.observability.trace.tools import setup_processor


def test_qualified_perfetto_version_matches_dependency_pin() -> None:
    project = Path(__file__).resolve().parents[4] / "pyproject.toml"
    with project.open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    assert f"perfetto=={versions.perfetto_version}" in dependencies


@pytest.mark.parametrize("field", ["perfetto_version", "processor_version"])
def test_bundle_rejects_unqualified_tool_version(tmp_path: Path, field: str) -> None:
    bundle = TraceBundle(traces=()).model_copy(update={field: "unqualified"})
    (tmp_path / "bundle.json").write_text(bundle.model_dump_json())
    with pytest.raises(TraceError) as error:
        read_bundle(tmp_path)
    assert error.value.code == TraceErrorCode.TOOL_VERSION


@pytest.mark.parametrize("name", ["../outside", "/etc/passwd", "nested/../../outside"])
def test_inventory_confinement(tmp_path: Path, name: str) -> None:
    with pytest.raises(TraceError) as error:
        confined(tmp_path, name)
    assert error.value.code == TraceErrorCode.INPUT


def test_inventory_rejects_symlink_escape(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(TraceError):
        confined(tmp_path, "link/escape")


@pytest.mark.parametrize("relation", ["same", "inside", "ancestor", "existing"])
def test_output_never_overlaps_or_reuses_input(tmp_path: Path, relation: str) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    output = {"same": root, "inside": root / "derived", "ancestor": tmp_path, "existing": tmp_path / "existing"}[
        relation
    ]
    if relation == "existing":
        output.mkdir()
    with pytest.raises(TraceError) as error:
        prepare_output(output, (root,))
    assert error.value.code == TraceErrorCode.OUTPUT


def test_explicit_inventory_and_hashes_preserve_raw_files(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    manifest = publish_inputs(root, scenario("normal_dense"))
    before = {path: path.read_bytes() for path in root.iterdir()}
    view, index, digests = load_inputs(root, manifest)
    output = tmp_path / "derived"
    bundle = export_traces(view, index, output, protected=(root,), inputs=digests)
    assert read_bundle(output) == bundle
    assert before == {path: path.read_bytes() for path in root.iterdir()}
    assert {item.path for item in digests} == {str(path) for path in before}
    assert (output / "quality.json").is_file()
    assert (output / "calibration.json").is_file()


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.json"
    manifest.write_text('{"chunks":[],"metadata":[],"guess_complete":true}')
    with pytest.raises(TraceError) as error:
        load_inputs(tmp_path, manifest)
    assert error.value.__cause__ is not None


def test_missing_processor_is_explicit_without_downloading(tmp_path: Path) -> None:
    with pytest.raises(TraceError) as error:
        verify_processor(tmp_path / "absent")
    assert error.value.code == TraceErrorCode.DEPENDENCY_MISSING
    assert list(tmp_path.iterdir()) == []


def test_wrong_processor_build_is_rejected(tmp_path: Path) -> None:
    binary = tmp_path / "processor"
    binary.write_bytes(b"different build")
    binary.chmod(0o755)
    with pytest.raises(TraceError) as error:
        verify_processor(binary)
    assert error.value.code == TraceErrorCode.TOOL_VERSION


def test_setup_checks_download_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def bad_download(method: str, url: str, **kwargs: object) -> Iterator[httpx.Response]:
        yield httpx.Response(200, content=b"wrong processor", request=httpx.Request(method, url))

    monkeypatch.setattr("xpool.observability.trace.tools.httpx.stream", bad_download)
    with pytest.raises(TraceError) as error:
        setup_processor(tmp_path)
    assert error.value.code == TraceErrorCode.TOOL_VERSION
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "case", ["missing_opening", "missing_result", "execution_never_returns", "execution_returns_failure"]
)
def test_partial_intervals_do_not_become_successful_slices(tmp_path: Path, case: str) -> None:
    view = reconstruct(events_for(scenario(case)))
    bundle = export_traces(view, build_calibration(view, ()), tmp_path / "out")
    partial = set()
    for interval in view.intervals:
        if interval.status != "complete":
            partial.add((interval.observation.entity.kind, interval.name))
    assert partial
    for member in bundle.traces:
        trace = Trace.FromString((tmp_path / "out" / member.path).read_bytes())
        for packet in trace.packet:
            event = packet.track_event
            if event.name != "xpool.interval":
                continue
            payload = json.loads(next(item.string_value for item in event.debug_annotations if item.name == "payload"))
            if payload["status"] != "complete":
                assert event.type == TrackEvent.TYPE_INSTANT


def test_unmapped_device_domains_have_no_clock_snapshots(tmp_path: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    bundle = export_traces(view, build_calibration(view, ()), tmp_path / "out")
    for member in bundle.traces:
        trace = Trace.FromString((tmp_path / "out" / member.path).read_bytes())
        assert not any(packet.HasField("clock_snapshot") for packet in trace.packet)
    assert len([entry for entry in bundle.traces if entry.kind == "raw"]) > 1


def test_corrupt_bundle_fails_before_query(tmp_path: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    output = tmp_path / "out"
    bundle = export_traces(view, build_calibration(view, ()), output)
    (output / bundle.traces[0].path).write_bytes(b"truncated")
    with pytest.raises(TraceError) as error:
        read_bundle(output)
    assert error.value.code == TraceErrorCode.INPUT


def test_failed_export_has_no_completion_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    original = Path.write_bytes

    def fail_trace(path: Path, data: bytes) -> int:
        if path.parent == output:
            raise OSError("simulated storage failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_trace)
    view = reconstruct(events_for(scenario("normal_dense")))
    with pytest.raises(TraceError) as error:
        export_traces(view, build_calibration(view, ()), output)
    assert error.value.code == TraceErrorCode.EXPORT
    assert isinstance(error.value.__cause__, OSError)
    assert not (output / "bundle.json").exists()


def test_overlapping_intervals_use_reusable_tracks(tmp_path: Path) -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    interval = next(
        item for item in view.intervals if item.status == "complete" and item.begin is not None and item.end is not None
    )
    # Two identities overlap at the same semantic site, requiring independent
    # slots rather than a fabricated parent/child relationship.
    assert interval.begin is not None
    second_identity = replace(interval.observation.entity, components=(*interval.observation.entity.components, 99))
    second = replace(interval, observation=replace(interval.observation, entity=second_identity))
    view = replace(view, intervals=(interval, second))
    output = tmp_path / "out"
    bundle = export_traces(view, build_calibration(view, ()), output)
    raw = next(
        member
        for member in bundle.traces
        if member.kind == "raw" and member.reference == str(interval.begin.record.clock_domain_id)
    )
    trace = Trace.FromString((output / raw.path).read_bytes())
    starts = [packet.track_event for packet in trace.packet if packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN]
    assert len(starts) == 2
    assert len({event.track_uuid for event in starts}) == 2


def test_zero_length_slice_closes_before_reusing_track() -> None:
    timeline = Timeline("run", "raw", "domain", {}, 0)
    first = timeline.add(10, "slot", "first", None, evidence_id="first", category="graph")
    first.type = TrackEvent.TYPE_SLICE_BEGIN
    closing = TrackEvent(type=TrackEvent.TYPE_SLICE_END, track_uuid=first.track_uuid)
    timeline.packets.append((10, 1, len(timeline.packets), closing))
    second = timeline.add(10, "slot", "second", None, evidence_id="second", category="graph")
    second.type = TrackEvent.TYPE_SLICE_BEGIN
    trace = Trace.FromString(timeline.serialize())
    assert [packet.track_event.type for packet in trace.packet if packet.HasField("track_event")] == [
        TrackEvent.TYPE_SLICE_BEGIN,
        TrackEvent.TYPE_SLICE_END,
        TrackEvent.TYPE_SLICE_BEGIN,
    ]


def test_uint64_producer_identity_does_not_overflow_perfetto_sequence(tmp_path: Path) -> None:
    data = fixture("calibration")["cases"][0]["input"]
    original = next(event for event in events_for(scenario("normal_dense")) if event.source.identity.producer == 3)
    original = replace(original, record=original.record.model_copy(update={"raw_time_ns": 55}))
    producer_id = 2**40
    events = []
    for event in (*calibration_events(data), original):
        document = event.metadata
        document = document.model_copy(
            update={
                "producer_id": producer_id if document.producer_id == 3 else document.producer_id,
                "producers": [
                    producer.model_copy(update={"id": producer_id}) if producer.id == 3 else producer
                    for producer in document.producers
                ],
                "endpoints": [
                    endpoint.model_copy(
                        update={
                            "sites": [
                                site.model_copy(update={"producer": producer_id}) if site.producer == 3 else site
                                for site in endpoint.sites
                            ]
                        }
                    )
                    for endpoint in document.endpoints
                ],
            }
        )
        source = event.source
        if source.identity.producer == 3:
            source = replace(source, identity=replace(source.identity, producer=producer_id))
        events.append(replace(event, source=source, metadata=document))
    view = reconstruct(events)
    index = build_calibration(
        view,
        (
            SamplerDeclaration(
                sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
                declared_attempt_range=(1, 10),
            ),
        ),
    )
    output = tmp_path / "out"
    bundle = export_traces(view, index, output)
    member = next(member for member in bundle.traces if member.kind == "unified")
    trace = Trace.FromString((output / member.path).read_bytes())
    snapshots = [packet for packet in trace.packet if packet.HasField("clock_snapshot")]
    assert snapshots and all(packet.trusted_packet_sequence_id < 2**32 for packet in snapshots)
