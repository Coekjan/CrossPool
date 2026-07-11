from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tests.harness.native.loopback import expected_loopback_rotation, run_isolated_native_case
from tests.harness.native.transport import devagent_arena_process
from xpool.abi import DebugOption, FfnResultErrorCode, RuntimeRole

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.timeout(180),
]

INSTANCE_TRANSPORT_REQUEST_SCRIPT = r"""
import json
import sys

import torch

from xpool.abi import RuntimeRole
from xpool.cext import ensure_xpool_ops_loaded

cuda_device = int(sys.argv[1])
arena = str(json.loads(sys.argv[2]))
debug_options = int(sys.argv[3])

ensure_xpool_ops_loaded()
torch.cuda.set_device(cuda_device)
torch.ops.xpool.init(cuda_device, int(RuntimeRole.INSTANCE), debug_options)
hidden_states = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
    device="cuda",
    dtype=torch.float32,
)
torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
try:
    torch.ops.xpool.instance.ffn_shim(
        hidden_states,
        None,
        0,
        0,
        7,
        2,
        1,
        0,
        hidden_states.shape[0],
        0,
        1,
        0,
        1,
    )
finally:
    torch.ops.xpool.instance.detach_transport_arena(0, 0)
"""


def isolated_case_ffn_shim_transport_loopback_runtime_rotates_hidden_pairs() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.TRANSPORT_LOOPBACK))
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    with devagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        try:
            output = torch.ops.xpool.instance.ffn_shim(
                hidden_states,
                None,
                0,
                0,
                7,
                2,
                1,
                0,
                hidden_states.shape[0],
                0,
                1,
                0,
                1,
            )
        finally:
            torch.ops.xpool.instance.detach_transport_arena(0, 0)

    assert torch.allclose(output, expected_loopback_rotation(hidden_states), atol=1e-5, rtol=1e-5)


def isolated_case_transport_observer_records_cross_process_phases(output_path: str) -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.TRANSPORT_LOOPBACK))
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    with devagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
        observer_output_path=output_path,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        output = torch.ops.xpool.instance.ffn_shim(
            hidden_states,
            None,
            0,
            0,
            7,
            2,
            1,
            0,
            hidden_states.shape[0],
            0,
            1,
            0,
            1,
        )
        torch.ops.xpool.instance.detach_transport_arena(0, 0)
        assert output.shape == hidden_states.shape

    snapshot = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert snapshot["sequence"] == 1
    assert snapshot["dropped"] == 0
    record = snapshot["records"][0]
    assert record[0] == 1
    assert record[2] == hidden_states.shape[0]
    timestamps = record[3:15]
    assert all(timestamp > 0 for timestamp in timestamps)
    assert timestamps == sorted(timestamps)


def isolated_case_ffn_shim_transport_dp_padding_requires_token_count_metadata() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.TRANSPORT_LOOPBACK))
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    with devagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=False,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        try:
            with pytest.raises(RuntimeError, match="DP padding requires global_num_tokens_gpu"):
                torch.ops.xpool.instance.ffn_shim(
                    hidden_states,
                    None,
                    0,
                    0,
                    7,
                    2,
                    1,
                    1,
                    hidden_states.shape[0],
                    0,
                    1,
                    0,
                    1,
                )
        finally:
            torch.ops.xpool.instance.detach_transport_arena(0, 0)


def isolated_case_ffn_shim_transport_rejects_dp_token_count_size_mismatch() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.TRANSPORT_LOOPBACK))
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    global_num_tokens_gpu = torch.tensor([2], device="cuda", dtype=torch.int32)
    with devagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=2,
        launch_kernel=False,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        try:
            with pytest.raises(RuntimeError, match="one entry per attention DP rank"):
                torch.ops.xpool.instance.ffn_shim(
                    hidden_states,
                    global_num_tokens_gpu,
                    0,
                    0,
                    7,
                    2,
                    1,
                    1,
                    hidden_states.shape[0],
                    0,
                    1,
                    0,
                    2,
                )
        finally:
            torch.ops.xpool.instance.detach_transport_arena(0, 0)


def isolated_case_ffn_shim_transport_accepts_int64_dp_token_counts() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.TRANSPORT_LOOPBACK))
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    global_num_tokens_gpu = torch.tensor([hidden_states.shape[0]], device="cuda", dtype=torch.int64)
    with devagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        try:
            output = torch.ops.xpool.instance.ffn_shim(
                hidden_states,
                global_num_tokens_gpu,
                0,
                0,
                7,
                2,
                1,
                1,
                hidden_states.shape[0],
                0,
                1,
                0,
                1,
            )
        finally:
            torch.ops.xpool.instance.detach_transport_arena(0, 0)

    assert torch.allclose(output, expected_loopback_rotation(hidden_states), atol=1e-5, rtol=1e-5)


def isolated_case_ffn_shim_transport_disabled_reports_not_implemented() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption(0)))
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    with devagent_arena_process(
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
        transport_loopback_enabled=False,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = torch.ops.xpool.instance.ffn_shim(
                    hidden_states,
                    None,
                    0,
                    0,
                    7,
                    2,
                    1,
                    0,
                    hidden_states.shape[0],
                    0,
                    1,
                    0,
                    1,
                )
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isnan(output).all()
            assert torch.ops.xpool.instance.transport_error_snapshot(0, 0) == int(FfnResultErrorCode.NOT_IMPLEMENTED)
            smoke = torch.arange(32, device="cuda", dtype=torch.float32)
            assert smoke.sum().item() == 496
        finally:
            torch.ops.xpool.instance.detach_transport_arena(0, 0)


def isolated_case_ffn_shim_transport_reuses_contended_slot_for_concurrent_graphs() -> None:
    torch.ops.xpool.init(torch.cuda.current_device(), int(RuntimeRole.INSTANCE), int(DebugOption.TRANSPORT_LOOPBACK))
    first = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    second = torch.tensor(
        [[10.0, 4.0, -2.0, 6.0], [5.0, -1.0, 7.0, 9.0]],
        device="cuda",
        dtype=torch.float32,
    )
    with devagent_arena_process(
        cuda_device=first.device.index,
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
    ) as arena:
        torch.ops.xpool.instance.attach_transport_arena(0, 0, arena)
        try:
            stream_a = torch.cuda.Stream()
            stream_b = torch.cuda.Stream()

            torch.ops.xpool.instance.ffn_shim(first, None, 0, 0, 7, 2, 1, 0, first.shape[0], 0, 1, 0, 1)
            torch.cuda.synchronize()

            graph_a = torch.cuda.CUDAGraph()
            graph_b = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_a, stream=stream_a):
                first_output = torch.ops.xpool.instance.ffn_shim(
                    first, None, 0, 0, 7, 2, 1, 0, first.shape[0], 0, 1, 0, 1
                )
            with torch.cuda.graph(graph_b, stream=stream_b):
                second_output = torch.ops.xpool.instance.ffn_shim(
                    second, None, 0, 0, 7, 2, 1, 0, second.shape[0], 0, 1, 0, 1
                )
            torch.cuda.synchronize()

            graph_a.replay()
            graph_b.replay()
            torch.cuda.synchronize()
        finally:
            torch.ops.xpool.instance.detach_transport_arena(0, 0)

    assert torch.allclose(first_output, expected_loopback_rotation(first), atol=1e-5, rtol=1e-5)
    assert torch.allclose(second_output, expected_loopback_rotation(second), atol=1e-5, rtol=1e-5)


def test_ffn_shim_transport_loopback_runtime_rotates_hidden_pairs() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_ffn_shim_transport_loopback_runtime_rotates_hidden_pairs",
    )


def test_transport_observer_records_cross_process_phases(tmp_path: Path) -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_transport_observer_records_cross_process_phases",
        str(tmp_path / "observer.json"),
    )


def test_ffn_shim_transport_dp_padding_requires_token_count_metadata() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_ffn_shim_transport_dp_padding_requires_token_count_metadata",
    )


def test_ffn_shim_transport_rejects_dp_token_count_size_mismatch() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_ffn_shim_transport_rejects_dp_token_count_size_mismatch",
    )


def test_ffn_shim_transport_accepts_int64_dp_token_counts() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_ffn_shim_transport_accepts_int64_dp_token_counts",
    )


def test_ffn_shim_transport_disabled_fails_closed() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_ffn_shim_transport_disabled_reports_not_implemented",
    )


def test_transport_persistent_arena_launches_and_destroys() -> None:
    with devagent_arena_process(
        cuda_device=torch.cuda.current_device(),
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
    ) as arena:
        assert len(arena) == 128


def test_launch_transport_kernel_accepts_non_loopback_until_request() -> None:
    with devagent_arena_process(
        cuda_device=torch.cuda.current_device(),
        max_tokens=8,
        hidden_size=4,
        element_size=4,
        atn_dp_size=1,
        launch_kernel=True,
        transport_loopback_enabled=False,
    ):
        pass


def test_ffn_shim_transport_reuses_contended_slot_for_concurrent_graphs() -> None:
    run_isolated_native_case(
        Path(__file__),
        "isolated_case_ffn_shim_transport_reuses_contended_slot_for_concurrent_graphs",
    )
