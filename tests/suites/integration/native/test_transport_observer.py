"""Cross-process native Transport trace observation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.debug import native_debug_options
from tests.harness.native.process import run_native_case
from tests.harness.native.transport import controlled_atnagent_arena_process
from xpool.abi import TensorDType
from xpool.config import LoopbackSite
from xpool.runtime import RuntimeRole

pytestmark = [pytest.mark.requires_cuda(), pytest.mark.requires_mps, pytest.mark.timeout(180)]


def isolated_transport_observer_records_cross_process_phases(output_path: str) -> None:
    xpool.native.initialize(
        RuntimeRole.INSTANCE,
        cuda_device=torch.cuda.current_device(),
        debug_options=native_debug_options(loopback_site=LoopbackSite.ATNAGENT),
    )
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    with controlled_atnagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=1,
        activate_resident=True,
        observer_output_path=output_path,
    ) as controller:
        xpool.native.transport.attach_arena(0, 0, controller.handle)
        output = torch.ops.xpool.ffn_shim(hidden_states, None, 0, 2, 1, 0)
        torch.cuda.synchronize(hidden_states.device)
        controller.drain()
        xpool.native.transport.detach_arena()
        assert output.shape == hidden_states.shape

    snapshot = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert snapshot["sequence"] == 1
    assert snapshot["dropped"] == 0
    record = snapshot["records"][0]
    assert record["trace_id"] == 1
    assert record["payload_rows"] == hidden_states.shape[0]
    assert record["layer_ordinal"] == 0
    assert record["forward_mode"] == 2
    assert record["result_handoff"] == 1
    assert record["dp_padding_mode"] == 0
    assert record["result_code"] == 0
    timestamps = [
        record[name]
        for name in (
            "staging_started",
            "staging_completed",
            "published",
            "published_observed",
            "execution_started",
            "execution_admitted",
            "execution_completed",
            "evaluated",
            "evaluated_observed",
            "output_copied",
            "acknowledged",
        )
    ]
    assert all(timestamp > 0 for timestamp in timestamps)
    assert timestamps == sorted(timestamps)
    assert record["closed"] == 0


def test_transport_observer_records_cross_process_phases(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_observer_records_cross_process_phases,
        str(tmp_path / "observer.json"),
    )
