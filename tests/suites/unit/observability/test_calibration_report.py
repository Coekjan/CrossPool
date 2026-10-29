import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.support.observability import calibration_events, events_for, fixture, scenario

from xpool.observability import (
    CalibrationReport,
    LoadPhase,
    LoadPhaseRange,
    MappingStatus,
    Sampler,
    SamplerDeclaration,
    SamplerDiagnostics,
    build_calibration,
    build_calibration_report,
    calibrate,
    reconstruct,
)
from xpool.observability.calibration_report import DomainReport


def test_report_denominators_phase_measurements_and_deterministic_json() -> None:
    data = fixture("calibration")["cases"][0]["input"]
    samples = calibration_events(data)
    device = tuple(e for e in events_for(scenario("normal_dense")) if e.record.clock_domain_id == 2)[:2]
    device = tuple(
        replace(event, record=event.record.model_copy(update={"raw_time_ns": time}))
        for event, time in zip(device, (55, 105), strict=True)
    )
    view = reconstruct((*samples, *device))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
        observation_interval=(0, 200),
        load_phases=(
            LoadPhaseRange(first=1, last=5, phase=LoadPhase.IDLE),
            LoadPhaseRange(first=6, last=10, phase=LoadPhase.SERVING),
        ),
        diagnostics=SamplerDiagnostics(
            attempted=10, failed=0, skipped_busy=3, sampling_overhead_ns=(10, 30), provenance=("measured externally",)
        ),
    )
    index = build_calibration(view, (declaration,))
    report = build_calibration_report(index, view)
    domain = next(domain for domain in report.domains if domain.domain.id == 2)
    assert (domain.covered_time_ns, domain.observation_time_ns, domain.time_coverage) == (90, 200, Fraction(9, 20))
    assert (domain.mapped_events, domain.retained_events, domain.retained_event_coverage) == (1, 2, Fraction(1, 2))
    assert domain.uncertainty.mean_ns == 0
    sampler = report.samplers[0]
    assert sampler.window_widths.values_ns == (0,) * 10
    assert sampler.host_before_intervals.values_ns == (10,) * 9
    assert [(phase.phase, phase.widths.count) for phase in sampler.phase_windows] == [
        (LoadPhase.IDLE, 5),
        (LoadPhase.SERVING, 5),
    ]
    assert sampler.sampling_overhead is not None and sampler.sampling_overhead.mean_ns == 20
    assert sampler.declaration.diagnostics.skipped_busy == 3
    text = report.to_json()
    assert json.loads(text)["domains"][0]["time_coverage"] == {"numerator": "9", "denominator": "20"}
    reversed_view = reconstruct(reversed((*samples, *device)))
    assert build_calibration_report(build_calibration(reversed_view, (declaration,)), reversed_view).to_json() == text
    assert report.samplers[0].models[0].block.source_locations


@pytest.mark.parametrize("interval", [None, (50, 50)])
def test_missing_or_zero_time_denominator_and_no_events_remain_null(interval: tuple[int, int] | None) -> None:
    data = fixture("calibration")["cases"][0]["input"]
    view = reconstruct(calibration_events(data))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
        observation_interval=interval,
    )
    report = build_calibration_report(build_calibration(view, (declaration,)), view)
    assert report.domains[0].time_coverage is None
    assert report.domains[0].retained_event_coverage is None
    assert report.samplers[0].sampling_overhead is None
    assert report.samplers[0].declaration.diagnostics.skipped_busy is None


def test_overlapping_models_preserve_ambiguity_and_remove_overlap_from_coverage() -> None:
    data = fixture("calibration")["cases"][0]["input"]
    view = reconstruct(calibration_events(data))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
        observation_interval=(0, 220),
    )
    original = build_calibration(view, (declaration,))
    block = original.models[0].block
    times = (90, 100, 110, 120, 130, 140, 150, 160, 170, 200)
    samples = tuple(
        replace(
            s,
            attempt_sequence=s.attempt_sequence + 10,
            device_time_ns=time,
            host_before_ns=time + 100,
            host_after_ns=time + 100,
        )
        for s, time in zip(block.samples, times, strict=True)
    )
    second = calibrate(
        block.model_copy(update={"block_index": 1, "samples": samples, "declared_attempt_range": (1, 20)})
    )
    index = replace(original, models=(*original.models, second))
    assert index.map_raw(data["run_id"], 2, 95).status == MappingStatus.AMBIGUOUS_MODEL
    domain = build_calibration_report(index, view).domains[0]
    assert domain.covered_time_ns == 180
    assert domain.time_coverage == Fraction(9, 11)
    assert any(span.model_count == 2 for span in domain.spans)


def test_duplicate_chunk_locations_are_retained_without_inflating_attempt_statistics() -> None:
    data = fixture("calibration")["cases"][0]["input"]
    events = calibration_events(data)
    duplicate = replace(events[0], source=replace(events[0].source, path=Path("other.xpool")))
    view = reconstruct((*events, duplicate))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
    )
    report = build_calibration_report(build_calibration(view, (declaration,)), view)
    sampler = report.samplers[0]
    assert sampler.retained_attempts == 10
    assert len(sampler.models[0].block.source_locations) == 11


@pytest.mark.parametrize("field", ["contract_version", "algorithm_version", "assumption"])
def test_report_rejects_unsupported_versions_and_assumptions(field: str) -> None:
    data = fixture("calibration")["cases"][0]["input"]
    view = reconstruct(calibration_events(data))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
    )
    report = build_calibration_report(build_calibration(view, (declaration,)), view)
    assert json.loads(report.to_json())["algorithm_version"] == 1
    with pytest.raises(ValidationError):
        CalibrationReport.model_validate({**report.model_dump(), field: "unknown" if field == "assumption" else 99})


@pytest.mark.parametrize("value", [-1, 2**64, True])
def test_report_raw_intervals_and_counts_reject_out_of_range_integers(value: int) -> None:
    data = fixture("calibration")["cases"][0]["input"]
    view = reconstruct(calibration_events(data))
    declaration = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, 10),
    )
    domain = build_calibration_report(build_calibration(view, (declaration,)), view).domains[0]
    for changes in (
        {"observation_interval": (0, value)},
        {"support_intervals": ((0, value),)},
        {"retained_events": value},
    ):
        with pytest.raises(ValidationError):
            DomainReport.model_validate({**domain.model_dump(), **changes})
