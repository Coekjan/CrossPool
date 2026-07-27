"""Native Fabric canonical failure and activation-rejection behavior."""

from __future__ import annotations

import math

import pytest

from tests.harness.native.fabric import fabric_bootstrap, run_fabric_topology
from xpool.abi import FfnResultCode, TensorDType, XPoolForwardMode

pytestmark = [pytest.mark.requires_mps, pytest.mark.timeout(180)]


@pytest.mark.requires_cuda(min_devices=3)
def test_fabric_failure_uses_one_canonical_first_writer() -> None:
    """Converge two failing FfnAgents on one canonical failure payload."""

    with fabric_bootstrap() as uid:
        report = run_fabric_topology(
            uid,
            atnagent_count=1,
            ffnagent_count=2,
            executor_count=1,
            forward_modes=(XPoolForwardMode.DECODE,),
            dtype=TensorDType.FP32,
            repetition_count=1,
            loopback_enabled=False,
        )

    assert len(report.instances) == 1
    result = report.instances[0]
    assert result.repetitions_completed == 1
    assert result.result_code is FfnResultCode.NOT_IMPLEMENTED
    assert all(math.isnan(value) for value in result.actual)

    failures = tuple(participant.failure for participant in report.participants)
    assert all(failure is not None for failure in failures)
    payloads = {
        (
            failure.result_code,
            failure.origin_pe,
            failure.model_index,
            failure.invocation_sequence,
            failure.layer_ordinal,
        )
        for failure in failures
        if failure is not None
    }
    assert len(payloads) == 1
    result_code, origin_pe, model_index, invocation_sequence, layer_ordinal = payloads.pop()
    assert result_code is FfnResultCode.NOT_IMPLEMENTED
    assert origin_pe in {1, 2}
    assert (model_index, invocation_sequence, layer_ordinal) == (0, 1, 0)
    assert tuple(failure.publication for failure in failures if failure is not None) == (1, 1, 1)
    assert tuple(failure.claim for failure in failures if failure is not None) == (0, 1, 0)
    for participant in report.participants:
        assert participant.dropped == 0
        assert participant.sequence == len(participant.records)


@pytest.mark.requires_cuda(min_devices=2)
def test_fabric_rejects_over_capacity_before_resident_launch() -> None:
    """Reject an impossible cooperative grid and retain legal cleanup."""

    with fabric_bootstrap() as uid:
        report = run_fabric_topology(
            uid,
            atnagent_count=1,
            ffnagent_count=1,
            executor_count=4096,
            forward_modes=(XPoolForwardMode.DECODE,),
            dtype=TensorDType.FP32,
            expect_activation_rejection=True,
        )

    assert not report.instances
    assert len(report.activation_rejections) == 1
    rejection = report.activation_rejections[0]
    assert rejection.pe == 1
    assert "concurrently resident blocks" in rejection.message
    assert len(report.participants) == 2
    for participant in report.participants:
        assert participant.failure is None
        assert participant.sequence == 0
        assert participant.dropped == 0
        assert not participant.records
