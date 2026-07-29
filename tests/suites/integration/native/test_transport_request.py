"""Native Transport request validation and mailbox-reuse contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import xpool.native
from tests.harness.native.case import run_native_case
from tests.harness.native.transport.owner import (
    atnagent_arena_process,
)
from tests.harness.support.config import reset_global_config
from tests.harness.support.native.loopback import expected_loopback_rotation
from tests.harness.support.native.transport import initialize_instance_transport, instance_transport_runtime
from xpool.abi import TensorDType

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.timeout(180),
    pytest.mark.usefixtures(reset_global_config.__name__),
]


@pytest.mark.usefixtures(instance_transport_runtime.__name__)
def test_transport_request_metadata_is_validated_before_submit() -> None:
    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="invalid request metadata"):
        torch.ops.xpool.ffn_shim(hidden_states, None, 0, 99, 1, 0)


@pytest.mark.usefixtures(instance_transport_runtime.__name__)
def test_transport_request_tensors_are_validated_before_submit() -> None:
    cpu_hidden_states = torch.empty((2, 4), dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="only available for these backends"):
        torch.ops.xpool.ffn_shim(cpu_hidden_states, None, 0, 2, 1, 0)

    noncontiguous_hidden_states = torch.empty((4, 2), device="cuda", dtype=torch.float32).transpose(0, 1)
    with pytest.raises(RuntimeError, match="expects a contiguous tensor"):
        torch.ops.xpool.ffn_shim(noncontiguous_hidden_states, None, 0, 2, 1, 0)

    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.float32)
    cpu_token_counts = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="token counts must be a CUDA tensor"):
        torch.ops.xpool.ffn_shim(hidden_states, cpu_token_counts, 0, 2, 1, 1)


@pytest.mark.requires_cuda(min_devices=2)
@pytest.mark.usefixtures(instance_transport_runtime.__name__)
def test_transport_request_rejects_cross_device_token_counts() -> None:
    current_device = torch.cuda.current_device()
    other_device = next(device for device in range(torch.cuda.device_count()) if device != current_device)
    hidden_states = torch.empty((2, 4), device=torch.device("cuda", current_device), dtype=torch.float32)
    token_counts = torch.tensor([2], device=torch.device("cuda", other_device), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="token counts must be on the hidden-state device"):
        torch.ops.xpool.ffn_shim(hidden_states, token_counts, 0, 2, 1, 1)


def isolated_transport_dp_padding_requires_token_count_metadata(workdir: str) -> None:
    initialize_instance_transport(atnagent_loopback=True)
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    with atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=1,
        activate_resident=True,
    ) as arena:
        xpool.native.transport.attach_arena(0, 0, arena)
        try:
            with pytest.raises(RuntimeError, match="DP padding requires global_num_tokens_gpu"):
                torch.ops.xpool.ffn_shim(hidden_states, None, 0, 2, 1, 1)
        finally:
            xpool.native.transport.detach_arena()


def isolated_transport_rejects_dp_token_count_size_mismatch(workdir: str) -> None:
    initialize_instance_transport(atnagent_loopback=True)
    hidden_states = torch.ones((2, 4), device="cuda", dtype=torch.float32)
    global_num_tokens_gpu = torch.tensor([2], device="cuda", dtype=torch.int32)
    with atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=2,
        activate_resident=True,
    ) as arena:
        xpool.native.transport.attach_arena(0, 0, arena)
        try:
            with pytest.raises(RuntimeError, match="one entry per attention DP rank"):
                torch.ops.xpool.ffn_shim(hidden_states, global_num_tokens_gpu, 0, 2, 1, 1)
        finally:
            xpool.native.transport.detach_arena()


def isolated_transport_accepts_int64_dp_token_counts(workdir: str) -> None:
    initialize_instance_transport(atnagent_loopback=True)
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    global_num_tokens_gpu = torch.tensor([hidden_states.shape[0], 0], device="cuda", dtype=torch.int64)
    with atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=hidden_states.device.index,
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=2,
        activate_resident=True,
    ) as arena:
        xpool.native.transport.attach_arena(0, 0, arena)
        try:
            output = torch.ops.xpool.ffn_shim(hidden_states, global_num_tokens_gpu, 0, 2, 1, 1)
        finally:
            xpool.native.transport.detach_arena()

    assert torch.allclose(output, expected_loopback_rotation(hidden_states), atol=1e-5, rtol=1e-5)


def isolated_transport_reuses_mailbox_across_serial_graphs(workdir: str) -> None:
    initialize_instance_transport(atnagent_loopback=True)
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
    with atnagent_arena_process(
        workdir=Path(workdir),
        cuda_device=first.device.index,
        max_tokens=8,
        hidden_size=4,
        dtype=TensorDType.FP32,
        atn_dp_size=1,
        activate_resident=True,
    ) as arena:
        xpool.native.transport.attach_arena(0, 0, arena)
        try:
            stream_a = torch.cuda.Stream()
            stream_b = torch.cuda.Stream()
            torch.ops.xpool.ffn_shim(first, None, 0, 2, 1, 0)
            torch.cuda.synchronize()

            graph_a = torch.cuda.CUDAGraph()
            graph_b = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_a, stream=stream_a):
                first_output = torch.ops.xpool.ffn_shim(first, None, 0, 2, 1, 0)
            with torch.cuda.graph(graph_b, stream=stream_b):
                second_output = torch.ops.xpool.ffn_shim(second, None, 0, 2, 1, 0)
            torch.cuda.synchronize()

            graph_a.replay()
            torch.cuda.synchronize()
            graph_b.replay()
            torch.cuda.synchronize()
        finally:
            xpool.native.transport.detach_arena()

    assert torch.allclose(first_output, expected_loopback_rotation(first), atol=1e-5, rtol=1e-5)
    assert torch.allclose(second_output, expected_loopback_rotation(second), atol=1e-5, rtol=1e-5)


@pytest.mark.requires_mps
def test_transport_dp_padding_requires_token_count_metadata(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_dp_padding_requires_token_count_metadata,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )


@pytest.mark.requires_mps
def test_transport_rejects_dp_token_count_size_mismatch(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_rejects_dp_token_count_size_mismatch,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )


@pytest.mark.requires_mps
def test_transport_accepts_int64_dp_token_counts(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_accepts_int64_dp_token_counts,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )


@pytest.mark.requires_mps
def test_transport_reuses_mailbox_across_serial_graphs(tmp_path: Path) -> None:
    run_native_case(
        isolated_transport_reuses_mailbox_across_serial_graphs,
        str(tmp_path / "owner"),
        workdir=tmp_path / "case",
    )
