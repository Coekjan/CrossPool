from dataclasses import replace
from fractions import Fraction

import pytest
from pydantic import ValidationError
from tests.support.observability import calibration_events, events_for, fixture, scenario

from xpool.observability import ObservationError, reconstruct
from xpool.observability.calibration import (
    CalibrationError,
    CalibrationStatus,
    LoadPhase,
    LoadPhaseRange,
    MappingStatus,
    Sampler,
    calibrate,
    estimate_drift,
    estimate_duration,
)
from xpool.observability.calibration_index import SamplerDeclaration, SamplerDiagnostics, build_calibration, map_event
from xpool.observability.errors import ErrorCode


def declaration(last: int = 10) -> SamplerDeclaration:
    data = fixture("calibration")["cases"][0]["input"]
    return SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=1, device_clock_domain_id=2, raw_epoch=1),
        declared_attempt_range=(1, last),
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"declared_attempt_range": (10, 1)},
        {"observation_interval": (100, 1)},
        {"load_phases": (LoadPhaseRange(first=5, last=1, phase=LoadPhase.IDLE),)},
        {
            "load_phases": (
                LoadPhaseRange(first=1, last=5, phase=LoadPhase.IDLE),
                LoadPhaseRange(first=5, last=10, phase=LoadPhase.SERVING),
            )
        },
        {"load_phases": (LoadPhaseRange(first=1, last=11, phase=LoadPhase.IDLE),)},
        {"diagnostics": SamplerDiagnostics(attempted=1, failed=2)},
    ],
    ids=("attempt-range", "observation-range", "phase-range", "phase-overlap", "phase-outside", "counts"),
)
@pytest.mark.parametrize("retained_samples", [False, True])
def test_invalid_declaration_semantics_preserve_sampler_and_available_sources(
    updates: dict[str, object], retained_samples: bool
) -> None:
    value = SamplerDeclaration.model_validate({**declaration().model_dump(), **updates})
    events = calibration_events(fixture("calibration")["cases"][0]["input"]) if retained_samples else ()
    view = reconstruct(events)
    with pytest.raises(CalibrationError) as caught:
        build_calibration(view, (value,))
    assert caught.value.code == ErrorCode.INVALID_SAMPLE
    assert caught.value.sampler == value.sampler
    assert caught.value.producer == value.sampler.host_producer
    assert set(caught.value.sources) == {event.source for event in events}


@pytest.mark.parametrize("value", [-1, 2**64, True])
def test_declaration_fields_still_reject_invalid_integer_inputs_at_construction(value: int) -> None:
    with pytest.raises(ValidationError) as caught:
        SamplerDiagnostics(attempted=value)
    assert caught.value.errors()[0]["loc"] == ("attempted",)
    with pytest.raises(ValidationError) as caught:
        SamplerDeclaration(sampler=declaration().sampler, observation_interval=(0, value))
    assert caught.value.errors()[0]["loc"] == ("observation_interval", 1)


def test_phase_partition_and_equal_counts_are_valid_at_the_build_boundary() -> None:
    value = SamplerDeclaration(
        sampler=declaration().sampler,
        declared_attempt_range=(1, 10),
        load_phases=(
            LoadPhaseRange(first=6, last=10, phase=LoadPhase.SERVING),
            LoadPhaseRange(first=1, last=5, phase=LoadPhase.IDLE),
        ),
        diagnostics=SamplerDiagnostics(attempted=10, failed=10),
    )
    view = reconstruct(calibration_events(fixture("calibration")["cases"][0]["input"]))
    assert build_calibration(view, (value,)).models[0].status == CalibrationStatus.VALID


def test_direct_calibration_still_rejects_reversed_attempt_declarations() -> None:
    view = reconstruct(calibration_events(fixture("calibration")["cases"][0]["input"]))
    block = build_calibration(view, (declaration(),)).models[0].block
    with pytest.raises(CalibrationError) as caught:
        calibrate(block.model_copy(update={"declared_attempt_range": (10, 1)}))
    assert caught.value.code == ErrorCode.INVALID_SAMPLE
    assert caught.value.sampler == block.sampler


def test_common_host_identity_and_process_device_domains_remain_independent() -> None:
    view = reconstruct(events_for(scenario("normal_dense")))
    index = build_calibration(view, ())
    host = next(event for event in view.events.values() if event.record.clock_domain_id == 1)
    assert map_event(index, host).estimate_ns == host.record.raw_time_ns
    for domain in (2, 3, 4):
        assert index.map_raw(host.source.identity.run_id, domain, 55).status == MappingStatus.UNCALIBRATED


def test_known_mapping_and_source_check() -> None:
    view = reconstruct(calibration_events(fixture("calibration")["cases"][0]["input"]))
    index = build_calibration(view, (declaration(),))
    assert index.map_raw(declaration().sampler.run_id, 2, 55).estimate_ns == 155
    event = next(iter(view.events.values()))
    result = map_event(index, event)
    assert result.sources == (event.source,)
    with pytest.raises(ObservationError):
        map_event(index, replace(event, source=replace(event.source, clock_domain_id=2)))
    with pytest.raises(CalibrationError):
        build_calibration(view, ())
    with pytest.raises(CalibrationError):
        build_calibration(view, (declaration(), declaration(9)))


def test_fixed_drift_and_between_block_gap() -> None:
    case = fixture("calibration")["series_cases"][0]
    data = {"samples": [s for block in case["input"]["blocks"] for s in block]}
    view = reconstruct(calibration_events(data))
    index = build_calibration(view, (declaration(20),))
    a, b = index.models
    assert [a.a, b.a] == [Fraction(1), Fraction(101, 100)]
    drift = estimate_drift(a, b)
    assert drift.estimate_ppm == Fraction(10000) and drift.resolved
    assert index.map_raw(declaration().sampler.run_id, 2, 1050).status == MappingStatus.OUTSIDE_SUPPORT


def test_interblock_jump_requires_new_epoch_and_all_later_blocks_stay_unavailable() -> None:
    original = fixture("calibration")["cases"][0]["input"]["samples"]
    data = {
        "samples": [
            dict(
                s,
                attempt_sequence=s["attempt_sequence"] + block * 10,
                H_before=s["H_before"] + block * 1000,
                H_after=s["H_after"] + block * 1000,
            )
            for block in range(3)
            for s in original
        ]
    }
    index = build_calibration(reconstruct(calibration_events(data)), (declaration(30),))
    assert [model.status for model in index.models] == [
        CalibrationStatus.VALID,
        CalibrationStatus.CLOCK_JUMP,
        CalibrationStatus.CLOCK_JUMP,
    ]


def test_missing_declared_blocks_unknown_range_and_zero_retained_samples() -> None:
    view = reconstruct(calibration_events(fixture("calibration")["cases"][0]["input"]))
    index = build_calibration(view, (declaration(20),))
    assert index.models[1].status == CalibrationStatus.SAMPLING_GAP
    empty = replace(view, calibration_samples=())
    assert build_calibration(empty, (declaration(),)).models[0].status == CalibrationStatus.SAMPLING_GAP
    unknown = declaration().model_copy(update={"declared_attempt_range": None})
    assert build_calibration(view, (unknown,)).models[0].status == CalibrationStatus.ATTEMPT_RANGE_UNKNOWN


def test_unresolved_rate_difference_keeps_interval_containing_zero() -> None:
    data = fixture("calibration")["cases"][2]["input"]
    index = build_calibration(reconstruct(calibration_events(data)), (declaration(),))
    model = index.models[0]
    drift = estimate_drift(model, model)
    assert drift.lower_ppm == -Fraction(2, 9) * 10**6
    assert drift.upper_ppm == Fraction(2, 9) * 10**6
    assert drift.resolved is False


def test_two_gpu_uncertainty_does_not_change_observed_causality() -> None:
    data = fixture("calibration")["cases"][2]["input"]
    original = events_for(scenario("normal_dense"))
    edges = reconstruct(original).edges
    view = reconstruct(
        (
            *original,
            *calibration_events(data, event_sequence_base=1000),
            *calibration_events(data, host_producer=4, device_clock_domain_id=4, event_sequence_base=1000),
        )
    )
    second = SamplerDeclaration(
        sampler=Sampler(run_id=data["run_id"], host_producer=4, device_clock_domain_id=4, raw_epoch=1),
        declared_attempt_range=(1, 10),
    )
    index = build_calibration(view, (declaration(), second))
    duration = estimate_duration(index.map_raw(data["run_id"], 2, 55), index.map_raw(data["run_id"], 4, 60))
    assert (duration.estimate_ns, duration.lower_ns, duration.upper_ns, duration.ordering) == (
        5,
        -5,
        15,
        "indeterminate",
    )
    assert view.edges == edges and edges
    assert index.map_raw(data["run_id"], 3, 55).status == MappingStatus.UNCALIBRATED


def test_new_registered_epoch_reuses_attempt_numbers_without_cross_epoch_fit() -> None:
    epochs = fixture("calibration")["series_cases"][1]["input"]["epochs"]
    first = calibration_events(epochs[0])
    metadata = fixture("metadata")
    domain = dict(metadata["clock_domains"][1], id=6, raw_epoch=2)
    producer = dict(next(p for p in metadata["producers"] if p["id"] == 3), id=10, clock_domain_id=6)
    second_data = dict(
        epochs[1], metadata_overrides={"revision": 2, "clock_domains_add": [domain], "producers_add": [producer]}
    )
    second = calibration_events(second_data, device_clock_domain_id=6, event_sequence_base=100)
    second_declaration = SamplerDeclaration(
        sampler=Sampler(run_id=declaration().sampler.run_id, host_producer=1, device_clock_domain_id=6, raw_epoch=2),
        declared_attempt_range=(1, 10),
    )
    view = reconstruct((*first, *second))
    index = build_calibration(view, (declaration(), second_declaration))
    assert [model.b for model in index.models] == [Fraction(100), Fraction(1000)]
    assert [model.block.sampler.raw_epoch for model in index.models] == [1, 2]
    assert index.map_raw(declaration().sampler.run_id, 6, 55).estimate_ns == 1055


def test_cross_run_and_cross_host_duration_or_drift_refuse_subtraction() -> None:
    view = reconstruct(calibration_events(fixture("calibration")["cases"][0]["input"]))
    index = build_calibration(view, (declaration(),))
    mapped = index.map_raw(declaration().sampler.run_id, 2, 55)
    for other in (
        mapped.model_copy(update={"run_id": "other-run"}),
        mapped.model_copy(update={"host_creation_id": "other-host"}),
    ):
        with pytest.raises(ObservationError):
            estimate_duration(mapped, other)
    model = index.models[0]
    other_model = model.model_copy(update={"block": model.block.model_copy(update={"host_creation_id": "other-host"})})
    with pytest.raises(ObservationError):
        estimate_drift(model, other_model)
