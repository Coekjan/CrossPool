"""Cross-process native FFN Fabric payload loopback behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.native.fabric.bootstrap import fabric_bootstrap
from tests.harness.native.fabric.topology import run_fabric_topology
from tests.harness.support.native.fabric import assert_fabric_report, coordinator_records
from xpool.abi import TensorDType, XPoolForwardMode

pytestmark = [
    pytest.mark.requires_cuda(min_devices=2),
    pytest.mark.requires_mps,
    pytest.mark.timeout(180),
]


@pytest.mark.parametrize(
    "forward_mode",
    [
        pytest.param(XPoolForwardMode.DECODE, id="decode"),
        pytest.param(XPoolForwardMode.EXTEND, id="prefill"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(TensorDType.FP32, id="fp32"),
        pytest.param(TensorDType.FP16, id="fp16"),
        pytest.param(TensorDType.BF16, id="bf16"),
    ],
)
def test_fabric_loopback_round_trips_device_payload_between_agents(
    forward_mode: XPoolForwardMode,
    dtype: TensorDType,
    tmp_path: Path,
) -> None:
    """Round-trip hidden states through distinct AtnAgent and FfnAgent PEs."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_count=1,
            forward_modes=(forward_mode,),
            dtype=dtype,
        )
    assert_fabric_report(report, expected_repetitions=3)


@pytest.mark.parametrize(
    "forward_mode",
    [
        pytest.param(XPoolForwardMode.DECODE, id="decode"),
        pytest.param(XPoolForwardMode.EXTEND, id="prefill"),
    ],
)
def test_fabric_reuses_released_publications(forward_mode: XPoolForwardMode, tmp_path: Path) -> None:
    """Reuse model and Executor publications without stale sequences."""

    with fabric_bootstrap(workdir=tmp_path / "bootstrap") as uid:
        report = run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_count=1,
            forward_modes=(forward_mode,),
            dtype=TensorDType.FP32,
            repetition_count=2,
        )
    assert_fabric_report(report, expected_repetitions=2)
    sequences = [record.invocation_sequence for record in coordinator_records(report)]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 2
