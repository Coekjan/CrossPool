"""Read immutable contract fixtures; expansion is test-only, never an oracle generator."""

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from xpool.observability.models import Event, EventRecord, Metadata
from xpool.observability.provenance import EventIdentity, Source
from xpool.observability.wire import PresenceBit

fixture_root = Path(__file__).resolve().parents[2] / "docs/plans/unified-timeline-observability/fixtures"

# Fixed fixtures use a heterogeneous test-only DSL. Production payloads cross
# their strict Pydantic model boundary in events_for() and the public readers.
type FixtureData = dict[str, Any]


def fixture(name: str) -> FixtureData:
    return TypeAdapter(FixtureData).validate_json((fixture_root / f"{name}.json").read_bytes())


def scenario(case_id: str) -> FixtureData:
    cases = fixture("causality")["cases"]
    case = next(case for case in cases if case["id"] == case_id)
    return expand(case["input"])


def expand(data: FixtureData) -> FixtureData:
    data = copy.deepcopy(data)
    if "base_case" not in data:
        return data
    base = scenario(data.pop("base_case"))
    for producer, sequence in data.pop("remove_events", []):
        matches = [
            event
            for event in base["events"]
            if (event["producer"], event["record"]["event_sequence"]) == (producer, sequence)
        ]
        assert len(matches) == 1
        base["events"].remove(matches[0])
    for replacement in data.pop("replace_events", []):
        matches = [
            index
            for index, event in enumerate(base["events"])
            if (event["producer"], event["record"]["event_sequence"])
            == (replacement["producer"], replacement["record"]["event_sequence"])
        ]
        assert len(matches) == 1
        base["events"][matches[0]] = replacement
    base["events"].extend(data.pop("append_events", []))
    base.update(data)
    return base


def metadata_for(data: FixtureData) -> FixtureData:
    metadata = copy.deepcopy(fixture("metadata"))
    overrides = data.get("metadata_overrides", {})
    for name in ("producers", "clock_domains", "endpoints", "graph_templates", "generations"):
        for replacement in overrides.get(f"{name}_replace", []):
            matches = [index for index, entry in enumerate(metadata[name]) if entry["id"] == replacement["id"]]
            assert len(matches) == 1
            metadata[name][matches[0]] = replacement
        additions = overrides.get(f"{name}_add", [])
        if additions and name in ("producers", "clock_domains", "generations"):
            assert not {entry["id"] for entry in metadata[name]} & {entry["id"] for entry in additions}
        metadata[name].extend(additions)
    if "revision" in overrides:
        metadata["revision"] = overrides["revision"]
    if "model_topologies" in overrides:
        metadata["generations"][0]["model_topologies"] = overrides["model_topologies"]
    metadata["capabilities"].update(data.get("capabilities", {}))
    return metadata


def events_for(data: FixtureData) -> tuple[Event, ...]:
    metadata = metadata_for(data)
    events: list[Event] = []
    for position, item in enumerate(data["events"]):
        producer = next(p for p in metadata["producers"] if p["id"] == item["producer"])
        document = Metadata.model_validate_json(
            json.dumps(
                metadata
                | {
                    "producer_id": producer["id"],
                    "process": producer["process"],
                    "clock_domain_id": producer["clock_domain_id"],
                }
            )
        )
        record = EventRecord.model_validate(item["record"])
        source = Source(
            EventIdentity(document.run_id, producer["id"], record.event_sequence),
            fixture_root / "causality.json",
            position * 256,
            document.process.creation_id,
            record.clock_domain_id,
            (record.endpoint_owner, record.endpoint_sequence) if record.has(PresenceBit.ENDPOINT) else None,
            record.instance_rank if record.has(PresenceBit.INSTANCE_RANK) else None,
            record.pe if record.has(PresenceBit.PE) else None,
        )
        events.append(Event(record, source, document))
    return tuple(events)
