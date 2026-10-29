"""Published raw files remain authoritative through calibration and reporting."""

import hashlib
from pathlib import Path

from tests.support.observability import calibration_events, fixture

from xpool.observability import (
    ArtifactInputs,
    ChunkHeader,
    Sampler,
    SamplerDeclaration,
    build_calibration,
    build_calibration_report,
    encode_chunk,
    read_artifacts,
    reconstruct,
)


def test_published_calibration_inventory_to_report_keeps_raw_bytes(tmp_path: Path) -> None:
    data = fixture("calibration")["cases"][2]["input"]
    events = calibration_events(data)
    metadata_bytes = events[0].metadata.model_dump_json().encode()
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(metadata_bytes)
    chunk = tmp_path / "samples.xpool"
    chunk.write_bytes(
        encode_chunk(
            ChunkHeader(data["run_id"], 1, 1, 1, 0, 10000), tuple(event.record for event in events), metadata_bytes
        )
    )
    inputs = ArtifactInputs(chunks=(chunk,), metadata=(metadata,))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (metadata, chunk)}
    view = reconstruct(read_artifacts(inputs))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
        observation_interval=(0, 200),
    )
    index = build_calibration(view, (declaration,))
    result = index.map_raw(data["run_id"], 2, 55)
    assert (result.estimate_ns, result.lower_ns, result.upper_ns) == (160, 155, 165)
    report = build_calibration_report(index, view)
    assert report.samplers[0].models[0].block.samples[0].source.path == chunk
    assert report.samplers[0].models[0].block.samples[0].source.offset >= 128
    assert report.to_json().endswith("\n")
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (metadata, chunk)}
