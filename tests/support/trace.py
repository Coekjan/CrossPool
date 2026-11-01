"""Published fixture inventories used by offline trace acceptance tests."""

from collections import defaultdict
from pathlib import Path

from tests.support.observability import FixtureData, events_for

from xpool.observability import ChunkHeader, Event, encode_chunk
from xpool.observability.models import Artifact
from xpool.observability.trace import TraceInputs


def publish_inputs(root: Path, data: FixtureData, *, artifacts: tuple[Artifact, ...] | None = None) -> Path:
    root.mkdir()
    grouped: dict[int, list[Event]] = defaultdict(list)
    for event in events_for(data):
        grouped[event.source.identity.producer].append(event)
    chunks: list[str] = []
    metadata: list[str] = []
    for producer, events in sorted(grouped.items()):
        document = events[0].metadata
        if artifacts is not None:
            document = document.model_copy(update={"artifacts": list(artifacts)})
        metadata_name = f"metadata-{producer}.json"
        metadata_bytes = document.model_dump_json().encode()
        (root / metadata_name).write_bytes(metadata_bytes)
        chunk_name = f"chunk-{producer}.xpool"
        header = ChunkHeader(document.run_id, producer, 1, document.revision, 0, 10000)
        (root / chunk_name).write_bytes(
            encode_chunk(
                header,
                tuple(event.record for event in sorted(events, key=lambda item: item.record.event_sequence)),
                metadata_bytes,
            )
        )
        chunks.append(chunk_name)
        metadata.append(metadata_name)
    manifest = root / "inputs.json"
    manifest.write_text(TraceInputs(chunks=tuple(chunks), metadata=tuple(metadata)).model_dump_json(indent=2))
    return manifest
