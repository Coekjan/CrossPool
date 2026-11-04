"""Fixed numerical vectors are independent of the implementation under test."""

from dataclasses import replace
from fractions import Fraction
from math import ceil, floor

import pytest
from pydantic import ValidationError
from tests.support.observability import FixtureData, events_for, fixture

from xpool.observability.calibration import (
    CalibrationError,
    CalibrationModel,
    CalibrationStatus,
    ClockAssumption,
    DurationEstimate,
    DurationReason,
    MappedTime,
    MappingReason,
    MappingStatus,
    ModelId,
    SampleBlock,
    Sampler,
    calibrate,
    estimate_duration,
    map_time,
)
from xpool.observability.errors import ErrorCode
from xpool.observability.models import PairedClockSample
from xpool.observability.wire import SampleFailureCode


def block_for(case: FixtureData) -> SampleBlock:
    data = case["input"]
    source = events_for({"events": [fixture("calibration")["wire_sample_cases"][0]["input"]["event"]]})[0].source
    samples = tuple(
        PairedClockSample(
            replace(source, identity=replace(source.identity, event_sequence=item["attempt_sequence"])),
            item["attempt_sequence"],
            data["device_clock_domain_id"],
            data["raw_epoch"],
            item["H_before"],
            item.get("G"),
            item.get("H_after"),
            item["outcome"],
            SampleFailureCode(item["failure_code"]) if "failure_code" in item else None,
        )
        for item in data["samples"]
    )
    return SampleBlock(
        sampler=Sampler(
            run_id=data["run_id"],
            host_producer=data["host_producer"],
            device_clock_domain_id=data["device_clock_domain_id"],
            raw_epoch=data["raw_epoch"],
        ),
        host_creation_id="test-host",
        block_index=0,
        declared_attempt_range=tuple(data["declared_attempt_range"]),
        samples=samples,
    )


@pytest.mark.parametrize("case", fixture("calibration")["cases"], ids=lambda case: case["id"])
def test_fixed_block_and_mapping_vectors(case: FixtureData) -> None:
    expected = case["expected"]
    block = block_for(case)
    if "error" in expected:
        with pytest.raises(CalibrationError) as caught:
            calibrate(block)
        assert caught.value.code.value == expected["error"]
        assert caught.value.attempt_sequence == expected["attempt_sequence"]
        assert caught.value.sources
        return
    model = calibrate(block)
    actual = model.model_dump(mode="json")
    for key, value in expected.items():
        if key != "mappings":
            assert actual[key] == value
    for mapping in expected.get("mappings", []):
        result = map_time(model, mapping["G"]).model_dump(mode="json")
        for key, value in mapping.items():
            if key != "G":
                assert result[key] == value
    if model.status != CalibrationStatus.VALID:
        assert all(value is None for value in (model.a, model.b, model.valid_G, model.G0, model.h))
        unavailable = map_time(model, 50)
        assert unavailable.estimate_exact is unavailable.lower_exact is unavailable.upper_exact is None


def mapped_endpoint(data: FixtureData) -> MappedTime:
    if "estimate_exact" not in data:
        return MappedTime(
            run_id="test-run",
            host_creation_id="test-host",
            clock_domain_id=2,
            raw_time_ns=0,
            status=MappingStatus(data.get("status", "uncalibrated")),
        )
    exact = {
        key: Fraction(int(data[key]["numerator"]), int(data[key]["denominator"]))
        for key in ("estimate_exact", "lower_exact", "upper_exact")
    }
    return MappedTime(
        run_id="test-run",
        host_creation_id="test-host",
        clock_domain_id=2,
        raw_time_ns=0,
        status=MappingStatus.MAPPED,
        estimate_exact=exact["estimate_exact"],
        lower_exact=exact["lower_exact"],
        upper_exact=exact["upper_exact"],
        estimate_ns=round(exact["estimate_exact"]),
        lower_ns=floor(exact["lower_exact"]),
        upper_ns=ceil(exact["upper_exact"]),
    )


@pytest.mark.parametrize("case", fixture("calibration")["duration_cases"], ids=lambda case: case["id"])
def test_fixed_duration_vectors(case: FixtureData) -> None:
    data = case["input"]
    end = mapped_endpoint(data["end"] if "estimate_exact" in data["end"] else {})
    result = estimate_duration(mapped_endpoint(data["start"]), end).model_dump(mode="json")
    for key, value in case["expected"].items():
        if key != "protocol_edge_retained":
            assert result[key] == value


def test_validation_never_changes_fit_or_repairs_failed_block() -> None:
    block = block_for(fixture("calibration")["cases"][2])
    original = calibrate(block)
    widened = tuple(
        replace(s, host_before_ns=0, host_after_ns=10000) if s.attempt_sequence in (4, 9) else s for s in block.samples
    )
    result = calibrate(block.model_copy(update={"samples": widened}))
    assert (result.vertices, result.a, result.b) == (original.vertices, original.a, original.b)
    changed = tuple(
        replace(s, host_before_ns=9999, host_after_ns=10000) if s.attempt_sequence == 4 else s for s in block.samples
    )
    failed = calibrate(block.model_copy(update={"samples": changed}))
    assert failed.status == CalibrationStatus.VALIDATION_FAILED
    assert failed.validation_pass == (False, True)


def test_unknown_range_does_not_infer_completeness_from_samples() -> None:
    block = block_for(fixture("calibration")["cases"][0])
    assert calibrate(block.model_copy(update={"declared_attempt_range": None})).status == (
        CalibrationStatus.ATTEMPT_RANGE_UNKNOWN
    )


def test_unordered_samples_and_multiple_failure_reasons() -> None:
    block = block_for(fixture("calibration")["cases"][0])
    assert calibrate(block.model_copy(update={"samples": tuple(reversed(block.samples))})).a == Fraction(1)
    samples = tuple(
        replace(s, device_time_ns=None, host_after_ns=None, outcome="failed", failure_code=SampleFailureCode.TIMED_OUT)
        if s.attempt_sequence == 2
        else s
        for s in block.samples[1:]
    )
    result = calibrate(block.model_copy(update={"samples": samples, "clock_jump": True}))
    assert result.reasons[:3] == (
        CalibrationStatus.SAMPLING_GAP,
        CalibrationStatus.SAMPLE_FAILED,
        CalibrationStatus.CLOCK_JUMP,
    )


def test_validation_outside_support_is_not_repaired_by_expanding_support() -> None:
    block = block_for(fixture("calibration")["cases"][0])
    samples = tuple(
        replace(s, device_time_ns=101, host_before_ns=201, host_after_ns=201) if s.attempt_sequence == 9 else s
        for s in block.samples
    )
    # Position 10 follows it in raw time, so isolate support behavior from a jump.
    samples = tuple(
        replace(s, device_time_ns=102, host_before_ns=202, host_after_ns=202) if s.attempt_sequence == 10 else s
        for s in samples
    )
    # Put the first held-out value below fitting support while preserving raw order
    # checks as an independent diagnostic; support failure must still be recorded.
    samples = tuple(
        replace(s, device_time_ns=0, host_before_ns=100, host_after_ns=100) if s.attempt_sequence == 4 else s
        for s in samples
    )
    result = calibrate(block.model_copy(update={"samples": samples}))
    assert CalibrationStatus.VALIDATION_OUTSIDE_SUPPORT in result.reasons
    assert result.valid_G is None


def test_outward_rounding_happens_after_exact_subtraction() -> None:
    model = calibrate(block_for(fixture("calibration")["cases"][3]))
    duration = estimate_duration(map_time(model, 3), map_time(model, 5))
    assert (duration.estimate_ns, duration.lower_ns, duration.upper_ns) == (1, 1, 1)


@pytest.mark.parametrize("value", [-1, 2**64, True])
def test_raw_sample_integer_bounds_fail_with_source_context(value: int) -> None:
    block = block_for(fixture("calibration")["cases"][0])
    sample = replace(block.samples[0], host_before_ns=value)
    with pytest.raises(CalibrationError) as caught:
        calibrate(block.model_copy(update={"samples": (sample, *block.samples[1:])}))
    assert caught.value.code == ErrorCode.INVALID_SAMPLE
    assert caught.value.sampler == block.sampler
    assert caught.value.sources == (sample.source,)


def test_integer_subclasses_and_uint64_endpoints_preserve_exact_calibration() -> None:
    class Integer(int):
        pass

    block = block_for(fixture("calibration")["cases"][0])
    offset = 2**64 - 1 - max(sample.host_before_ns for sample in block.samples)
    samples = tuple(
        replace(
            sample,
            host_before_ns=Integer(sample.host_before_ns + offset),
            host_after_ns=Integer(sample.host_before_ns + offset),
        )
        for sample in block.samples
    )
    model = calibrate(block.model_copy(update={"samples": samples}))
    assert model.status == CalibrationStatus.VALID
    assert map_time(model, 100).estimate_ns == 2**64 - 1
    assert model.b == 100 + offset


@pytest.mark.parametrize("value", [-1, 2**64, True])
def test_block_model_and_mapping_reject_invalid_fixed_width_fields(value: int) -> None:
    block = block_for(fixture("calibration")["cases"][0])
    with pytest.raises(ValidationError):
        SampleBlock.model_validate({**block.model_dump(), "block_index": value})
    with pytest.raises(ValidationError):
        ModelId(sampler=block.sampler.identity, block_index=value)
    mapped = map_time(calibrate(block), 55)
    model = calibrate(block)
    for field in ("G0", "h", "duplicate_sample_count"):
        with pytest.raises(ValidationError):
            CalibrationModel.model_validate({**model.model_dump(), field: value})
    for field in ("clock_domain_id", "raw_epoch", "raw_time_ns"):
        with pytest.raises(ValidationError):
            MappedTime.model_validate({**mapped.model_dump(), field: value})


def test_model_versions_and_closed_mapping_values_validate_json() -> None:
    model = calibrate(block_for(fixture("calibration")["cases"][0]))
    with pytest.raises(ValidationError):
        CalibrationModel.model_validate({**model.model_dump(), "algorithm_version": 99})
    mapped = map_time(model, 55)
    assert mapped.assumption == ClockAssumption.SEGMENT_AFFINE_BRACKETING
    assert MappedTime.model_validate(mapped.model_dump()) == mapped
    for field, value in (("assumption", "unknown"), ("reasons", ("unknown",))):
        with pytest.raises(ValidationError):
            MappedTime.model_validate({**mapped.model_dump(), field: value})
    outside = map_time(model, 0)
    assert outside.reasons == (MappingReason.OUTSIDE_SUPPORT,)
    duration = estimate_duration(mapped, outside)
    assert duration.reasons == (DurationReason.UNAVAILABLE_ENDPOINT,)
    with pytest.raises(ValidationError):
        DurationEstimate.model_validate({**duration.model_dump(), "reasons": ("unknown",)})


def test_exact_mapping_serialization_retains_arbitrary_rational_precision() -> None:
    exact = Fraction(2**160 + 1, 3)
    mapped = MappedTime(
        run_id="test-run",
        host_creation_id="test-host",
        clock_domain_id=1,
        raw_time_ns=0,
        status=MappingStatus.MAPPED,
        estimate_exact=exact,
        lower_exact=exact,
        upper_exact=exact,
        estimate_ns=round(exact),
        lower_ns=floor(exact),
        upper_ns=ceil(exact),
    )
    assert mapped.model_dump(mode="json")["estimate_exact"] == {
        "numerator": str(exact.numerator),
        "denominator": "3",
    }
