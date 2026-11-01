"""Evidence-preserving standard TrackEvent export over raw reconstruction and calibrated maps."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import BuiltinClock, Trace, TrackEvent
from pydantic import JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

from xpool.observability.calibration import CalibrationStatus, MappingStatus, estimate_duration
from xpool.observability.calibration_index import CalibrationIndex, map_event
from xpool.observability.calibration_report import build_calibration_report
from xpool.observability.intervals import Interval
from xpool.observability.models import Event
from xpool.observability.provenance import EventIdentity
from xpool.observability.reconstruction import Reconstruction
from xpool.observability.trace.inputs import prepare_output
from xpool.observability.trace.models import InputDigest, TraceBundle, TraceError, TraceErrorCode, TraceFile
from xpool.observability.wire import EventKind, PresenceBit


def json_value(value: object) -> JsonValue:
    # Serialization is a dynamic boundary shared by dataclasses and Pydantic
    # values. JsonValue validation immediately constrains the serialized result.
    return TypeAdapter(JsonValue).validate_python(to_jsonable_python(value))


def encoded(value: object) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def family(event: Event) -> str:
    kind = event.record.kind
    if kind in (EventKind.CALIBRATION_SAMPLE_SUCCEEDED, EventKind.CALIBRATION_SAMPLE_FAILED):
        return "calibration"
    if kind == EventKind.ROUTING_PUBLISHED:
        return "routing"
    if 500 <= kind < 550:
        return "graph"
    if 100 <= kind < 200:
        return "transport"
    return "fabric"


@dataclass
class Timeline:
    run_id: str
    kind: str
    reference: str
    times: dict[EventIdentity, int]
    origin: int
    trace: Trace = field(default_factory=Trace)
    tracks: dict[str, int] = field(default_factory=dict)
    # Each entry is a timestamp and a stable ordering key. End packets precede
    # subsequent starts at the same time; zero-duration pairs stay together.
    packets: list[tuple[int, int, int, TrackEvent]] = field(default_factory=list)

    def track(self, name: str, *, counter: bool = False) -> int:
        if name not in self.tracks:
            identifier = len(self.tracks) + 1
            self.tracks[name] = identifier
            descriptor = self.trace.packet.add().track_descriptor
            descriptor.uuid = identifier
            descriptor.name = name
            if counter:
                descriptor.counter.unit_name = "count"
        return self.tracks[name]

    def add(
        self, timestamp: int, track: str, name: str, payload: JsonValue, *, evidence_id: str, category: str
    ) -> TrackEvent:
        event = TrackEvent(type=TrackEvent.TYPE_INSTANT, track_uuid=self.track(track), name=name)
        for key, value in (("payload", encoded(payload)), ("evidence_id", evidence_id), ("category", category)):
            annotation = event.debug_annotations.add()
            annotation.name = key
            annotation.string_value = value
        self.packets.append((timestamp, 1, len(self.packets), event))
        return event

    def serialize(self) -> bytes:
        for timestamp, priority, ordinal, event in sorted(self.packets, key=lambda row: row[:3]):
            packet = self.trace.packet.add()
            packet.timestamp = timestamp - self.origin
            packet.timestamp_clock_id = 11  # BUILTIN_CLOCK_TRACE_FILE, an explicit display axis.
            packet.trusted_packet_sequence_id = 1
            packet.track_event.CopyFrom(event)
        return self.trace.SerializeToString(deterministic=True)


def interval_payload(interval: Interval, index: CalibrationIndex) -> JsonValue:
    begin, end = interval.begin, interval.end
    duration = None
    coverage = "unavailable"
    if begin is not None and end is not None:
        start, finish = map_event(index, begin), map_event(index, end)
        duration = estimate_duration(start, finish)
        domain = index.domain(begin.source.identity.run_id, begin.record.clock_domain_id).domain
        if domain.kind == "host_monotonic":
            coverage = "complete"
        else:
            models = index.models_for(begin.source.identity.run_id, domain.id)
            covering = [
                model
                for model in models
                if model.status == CalibrationStatus.VALID
                and model.valid_G is not None
                and model.valid_G[0] <= begin.record.raw_time_ns <= end.record.raw_time_ns <= model.valid_G[1]
            ]
            overlapping = [
                model
                for model in models
                if model.status == CalibrationStatus.VALID
                and model.valid_G is not None
                and model.valid_G[0] <= end.record.raw_time_ns
                and model.valid_G[1] >= begin.record.raw_time_ns
            ]
            # A single unambiguous supported model proves continuous coverage;
            # other endpoint-mappable spans retain a conservative warning.
            coverage = (
                "complete"
                if len(covering) == 1 and len(overlapping) == 1 and start.model_id == finish.model_id
                else "incomplete"
            )
    return json_value(
        {
            "observation": interval.observation,
            "name": interval.name,
            "status": interval.status,
            "raw_duration_ns": interval.duration_ns,
            "reasons": interval.reasons,
            "begin": begin.source.identity if begin is not None else None,
            "end": end.source.identity if end is not None else None,
            "sources": interval.sources,
            "duration": duration,
            "calibration_coverage": coverage,
        }
    )


def export_traces(
    view: Reconstruction,
    index: CalibrationIndex,
    output: Path,
    *,
    protected: tuple[Path, ...] = (),
    inputs: tuple[InputDigest, ...] = (),
) -> TraceBundle:
    """Publish standard protobuf traces and a completion index into a new directory.

    Unified traces are partitioned by Run/common Host. Every raw domain also
    gets a trace for authoritative local intervals and calibration-gap drilldown.
    Built-in queries deduplicate these views by evidence identity. Raw paths,
    including attachment references, are protected even for direct API callers.
    Reconstructed and calibrated values retain raw integers and conditional uncertainty in arguments.
    """
    sources = {source.path.resolve() for locations in view.event_sources.values() for source in locations}
    sources.update(attachment.path.resolve() for attachment in view.attachments if attachment.path is not None)
    prepare_output(output, (*protected, *sources, *(Path(item.path) for item in inputs)))

    # Display partitions never connect unknown clock domains. Raw traces retain
    # all events, allowing an interval to straddle a mapped/unmapped boundary.
    groups: dict[tuple[str, str, str], dict[EventIdentity, int]] = defaultdict(dict)
    mappings = {identity: map_event(index, event) for identity, event in view.events.items()}
    for identity, event in sorted(view.events.items()):
        groups[identity.run_id, "raw", str(event.record.clock_domain_id)][identity] = event.record.raw_time_ns
        mapped = mappings[identity]
        if mapped.status == MappingStatus.MAPPED and mapped.estimate_ns is not None:
            groups[identity.run_id, "unified", str(mapped.host_creation_id)][identity] = mapped.estimate_ns
    timelines = [
        Timeline(run, kind, reference, times, min(times.values()))
        for (run, kind, reference), times in sorted(groups.items())
    ]
    if not timelines:
        # Metadata-only inventories remain queryable even with no retained event.
        for run in sorted({document.run_id for document in view.metadata}):
            timelines.append(Timeline(run, "raw", "metadata", {}, 0))
    report = build_calibration_report(index, view)
    files: list[TraceFile] = []
    for number, timeline in enumerate(timelines):
        populate(timeline, view, index)
        path = output / f"{number:04d}-{timeline.kind}.pftrace"
        try:
            data = timeline.serialize()
            path.write_bytes(data)
        except (OSError, ValueError) as error:
            raise TraceError(TraceErrorCode.EXPORT, "protobuf export failed", path=path) from error
        files.append(
            TraceFile(
                path=path.name,
                sha256=hashlib.sha256(data).hexdigest(),
                run_id=timeline.run_id,
                kind="unified" if timeline.kind == "unified" else "raw",
                reference=timeline.reference,
                origin_ns=timeline.origin,
            )
        )
    (output / "calibration.json").write_text(report.to_json())
    (output / "quality.json").write_text(
        encoded(
            {
                "findings": view.quality.findings,
                "checkpoints": tuple(view.quality.checkpoints.values()),
            }
        )
        + "\n"
    )
    bundle = TraceBundle(traces=tuple(files), inputs=inputs)
    # This is the only completion marker. Partial exports cannot be queried as
    # complete bundles, and output directories are never reused implicitly.
    completion = output / "bundle.json.tmp"
    completion.write_text(bundle.model_dump_json(indent=2) + "\n")
    completion.replace(output / "bundle.json")
    return bundle


def populate(timeline: Timeline, view: Reconstruction, index: CalibrationIndex) -> None:
    checkpoint_times = [
        checkpoint.host_time_ns
        for key, checkpoint in view.quality.checkpoints.items()
        if checkpoint_on_timeline(timeline, view, index, key)
    ]
    timeline.origin = min((timeline.origin, *checkpoint_times))
    # Raw boundaries are retained as instants independently of reconstructed
    # slices. A source identity links every derived item back to the raw input.
    event_packets: dict[EventIdentity, TrackEvent] = {}
    # Raw Producer identities are uint64; Perfetto packet sequence IDs are
    # uint32 and are allocated independently of those protocol identities.
    snapshot_sequences = {
        # TrackEvents use sequence 1; start snapshot sequences at 2 to isolate each
        # Producer's sequence-scoped custom clock 64 from the event writer.
        producer: position + 2
        for position, producer in enumerate(sorted({identity.producer for identity in view.events}))
    }
    predecessors: dict[EventIdentity, set[EventIdentity]] = {identity: set() for identity in sorted(timeline.times)}
    for edge in view.edges:
        if (
            isinstance(edge.origin, EventIdentity)
            and isinstance(edge.destination, EventIdentity)
            and edge.origin in timeline.times
            and edge.destination in timeline.times
            and timeline.times[edge.origin] == timeline.times[edge.destination]
        ):
            predecessors[edge.destination].add(edge.origin)
    try:
        event_order = tuple(TopologicalSorter(predecessors).static_order())
    except CycleError as error:
        raise TraceError(TraceErrorCode.EXPORT, "conflicting same-timestamp protocol order") from error
    # Equal display timestamps still need evidenced publication-before-observation
    # packet order for Perfetto flows. This tie-break never changes timestamps.
    for identity in event_order:
        timestamp = timeline.times[identity]
        event = view.events[identity]
        mapped = map_event(index, event)
        category = family(event)
        payload = json_value(
            {
                "identity": identity,
                "record": event.record,
                "sources": view.event_sources[identity],
                "mapping": mapped,
                "unit": "ns",
                "category": category,
            }
        )
        event_packets[identity] = timeline.add(
            timestamp,
            f"producer {identity.producer}/{category}/events",
            EventKind(event.record.kind).name,
            payload,
            evidence_id=encoded(("event", identity)),
            category=category,
        )
        if mapped.status != MappingStatus.MAPPED:
            timeline.add(
                timestamp,
                f"producer {identity.producer}/calibration gaps",
                "xpool.calibration_gap",
                json_value(mapped),
                evidence_id=encoded(("gap", identity)),
                category="calibration",
            )
        if timeline.kind == "unified" and mapped.model_id is not None:
            # Events are already mapped by the calibration consumer. Snapshots record the evidenced
            # correspondence at each retained timestamp; Perfetto never fits or
            # extrapolates a slope on behalf of the calibration consumer.
            packet = timeline.trace.packet.add()
            packet.trusted_packet_sequence_id = snapshot_sequences[identity.producer]
            snapshot = packet.clock_snapshot
            snapshot.primary_trace_clock = BuiltinClock.BUILTIN_CLOCK_TRACE_FILE
            snapshot.clocks.add(clock_id=11, timestamp=timestamp - timeline.origin)
            snapshot.clocks.add(clock_id=64, timestamp=event.record.raw_time_ns)

    # Interval packing reuses tracks by semantic site and phase, with separate
    # slots only for simultaneous activities. Invocation IDs remain arguments.
    lanes: dict[str, list[int]] = defaultdict(list)
    ordered = sorted(
        view.intervals,
        key=lambda item: (
            timeline.times.get(item.begin.source.identity, timeline.origin) if item.begin else timeline.origin,
            encoded((item.observation, item.name)),
        ),
    )
    for interval in ordered:
        if interval.observation.entity.run_id != timeline.run_id:
            continue
        begin, end = interval.begin, interval.end
        available = [event for event in (begin, end) if event is not None and event.source.identity in timeline.times]
        if not available:
            continue
        payload = interval_payload(interval, index)
        evidence_id = encoded(("interval", interval.observation, interval.name))
        category = family(available[0])
        start = timeline.times[available[0].source.identity]
        base = f"producer {interval.observation.producer}/{interval.observation.role.name.lower()}"
        record = available[0].record
        if record.has(PresenceBit.LEASE):
            base += f"/lane {record.lane_index}"
        base += f"/{category}/{interval.name}"
        complete = interval.status == "complete" and begin is not None and end is not None
        finish = timeline.times.get(end.source.identity) if end is not None else None
        if not complete or finish is None or begin.source.identity not in timeline.times or finish < start:
            timeline.add(
                start, base + "/boundaries", "xpool.interval", payload, evidence_id=evidence_id, category=category
            )
            continue
        slots = lanes[base]
        slot = next((i for i, last in enumerate(slots) if last <= start), len(slots))
        if slot == len(slots):
            slots.append(finish)
        else:
            slots[slot] = finish
        track = f"{base}/slot {slot}"
        opening = timeline.add(start, track, "xpool.interval", payload, evidence_id=evidence_id, category=category)
        opening.type = TrackEvent.TYPE_SLICE_BEGIN
        closing = TrackEvent(type=TrackEvent.TYPE_SLICE_END, track_uuid=opening.track_uuid)
        timeline.packets.append((finish, 1 if finish == start else 0, len(timeline.packets), closing))

    # Entity relations carry evidence rather than invented execution times.
    # Only event-to-event protocol relations can supply concrete flow endpoints.
    for position, edge in enumerate(view.edges, 1):
        if edge.origin.run_id != timeline.run_id:
            continue
        endpoints = [source.identity for source in edge.sources if source.identity in timeline.times]
        timestamp = timeline.times[endpoints[0]] if endpoints else timeline.origin
        duration = None
        flow_status = "identity_relation"
        if isinstance(edge.origin, EventIdentity) and isinstance(edge.destination, EventIdentity):
            flow_status = "separate_domain"
            duration = estimate_duration(
                map_event(index, view.events[edge.origin]), map_event(index, view.events[edge.destination])
            )
            if (
                edge.origin in event_packets
                and edge.destination in event_packets
                and timeline.times[edge.origin] <= timeline.times[edge.destination]
            ):
                event_packets[edge.origin].flow_ids.append(position)
                event_packets[edge.destination].terminating_flow_ids.append(position)
                flow_status = "emitted"
            elif edge.origin in event_packets and edge.destination in event_packets:
                flow_status = "display_time_reversed"
        timeline.add(
            timestamp,
            "evidence/relations",
            "xpool.relation",
            json_value({"edge": edge, "duration": duration, "flow_status": flow_status}),
            evidence_id=encoded(("relation", edge.relation, edge.origin, edge.destination)),
            category=edge.relation,
        )
    for identity, entity in sorted(view.entities.items()):
        if identity.run_id != timeline.run_id:
            continue
        candidates = [source.identity for source in entity.sources if source.identity in timeline.times]
        timestamp = timeline.times[candidates[0]] if candidates else timeline.origin
        timeline.add(
            timestamp,
            "evidence/entities",
            "xpool.entity",
            json_value(
                {
                    "identity": identity,
                    "facts": entity.facts,
                    "sources": entity.sources,
                }
            ),
            evidence_id=encoded(("entity", identity)),
            category=identity.kind,
        )
    add_reports(timeline, view, index)


def checkpoint_on_timeline(
    timeline: Timeline, view: Reconstruction, index: CalibrationIndex, key: tuple[str, int]
) -> bool:
    if key[0] != timeline.run_id:
        return False
    producer = next(
        producer
        for document in view.metadata
        if document.run_id == key[0]
        for producer in document.producers
        if producer.id == key[1]
    )
    reference = index.domain(key[0], producer.clock_domain_id)
    if timeline.kind == "unified":
        return reference.host_creation_id == timeline.reference
    domain = next(
        (
            domain
            for document in view.metadata
            if document.run_id == key[0]
            for domain in document.clock_domains
            if str(domain.id) == timeline.reference
        ),
        None,
    )
    return (
        domain is not None and domain.kind == "host_monotonic" and domain.host_creation_id == reference.host_creation_id
    )


def add_reports(timeline: Timeline, view: Reconstruction, index: CalibrationIndex) -> None:
    for position, finding in enumerate(view.quality.findings):
        if finding.run_id is not None and finding.run_id != timeline.run_id:
            continue
        timeline.add(
            timeline.origin,
            "evidence/quality",
            "xpool.quality",
            json_value(finding),
            evidence_id=encoded(("quality", timeline.run_id, position)),
            category="quality",
        )
    for document in view.metadata:
        if document.run_id == timeline.run_id:
            timeline.add(
                timeline.origin,
                "evidence/metadata",
                "xpool.quality",
                json_value(document),
                evidence_id=encoded(("metadata", document.run_id, document.producer_id, document.revision)),
                category="quality",
            )
    for key, checkpoint in sorted(view.quality.checkpoints.items()):
        if key[0] != timeline.run_id:
            continue
        timeline.add(
            timeline.origin,
            "evidence/quality",
            "xpool.quality",
            json_value(checkpoint),
            evidence_id=encoded(("checkpoint", key)),
            category="quality",
        )
        # Counter values are observed cumulative checkpoints, never inferred
        # queue occupancy or a fabricated final total. The payload marks bounds.
        if not checkpoint_on_timeline(timeline, view, index, key):
            continue
        for name, count in checkpoint.counters.model_dump().items():
            # Perfetto stores counter.value as double. Larger integers retain
            # their exact checkpoint JSON instead of a rounded display value.
            if count > 2**53 - 1:
                timeline.add(
                    checkpoint.host_time_ns,
                    "evidence/quality",
                    "xpool.counter_display_unavailable",
                    json_value({"checkpoint": checkpoint, "counter": name, "reason": "outside_exact_counter_range"}),
                    evidence_id=encoded(("counter_limit", key, name)),
                    category="quality",
                )
                continue
            event = timeline.add(
                checkpoint.host_time_ns,
                f"producer {key[1]}/checkpoint/{name}",
                name,
                json_value(checkpoint),
                evidence_id=encoded(("counter", key, name)),
                category="counter",
            )
            event.track_uuid = timeline.track(f"producer {key[1]}/counter/{name}", counter=True)
            event.type = TrackEvent.TYPE_COUNTER
            event.counter_value = count
    for attachment in view.attachments:
        if attachment.run_id == timeline.run_id:
            timeline.add(
                timeline.origin,
                "evidence/attachments",
                "xpool.attachment",
                json_value(attachment),
                evidence_id=encoded(("attachment", attachment.run_id, attachment.owner, attachment.sequence)),
                category="routing",
            )
    # Coverage spans, models and uncalibrated domains remain queryable even
    # when no event falls at their boundaries. The report retains full sources.
    report = build_calibration_report(index, view)
    timeline.add(
        timeline.origin,
        "evidence/calibration",
        "xpool.calibration",
        json_value(report),
        evidence_id=encoded(("calibration", timeline.run_id)),
        category="calibration",
    )
