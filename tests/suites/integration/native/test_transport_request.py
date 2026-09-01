from __future__ import annotations

import pytest
import torch

from tests.harness.support.config import reset_global_config
from tests.harness.support.native.transport import instance_transport_runtime

pytestmark = [
    pytest.mark.requires_cuda(),
    pytest.mark.timeout(180),
    pytest.mark.usefixtures(reset_global_config.__name__),
    pytest.mark.usefixtures(instance_transport_runtime.__name__),
]


def test_transport_request_enum_is_validated_before_submit() -> None:
    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    with pytest.raises(RuntimeError, match="invalid forward mode"):
        torch.ops.xpool.ffn_shim(hidden_states, None, output, 0, 99, 1, 0)


def test_transport_request_tensors_are_validated_before_submit() -> None:
    cpu_hidden_states = torch.empty((2, 4), dtype=torch.bfloat16)
    cpu_output = torch.empty_like(cpu_hidden_states)
    with pytest.raises(NotImplementedError, match="only available for these backends"):
        torch.ops.xpool.ffn_shim(cpu_hidden_states, None, cpu_output, 0, 2, 1, 0)

    noncontiguous_hidden_states = torch.empty((4, 2), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    contiguous_output = torch.empty(noncontiguous_hidden_states.shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="expects a contiguous tensor"):
        torch.ops.xpool.ffn_shim(noncontiguous_hidden_states, None, contiguous_output, 0, 2, 1, 0)

    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    cpu_dp_rank_payload_rows = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="DP-rank payload rows must be a CUDA tensor"):
        torch.ops.xpool.ffn_shim(hidden_states, cpu_dp_rank_payload_rows, output, 0, 2, 1, 1)

    invalid_outputs = (
        (torch.empty(8, device="cuda"), "output must be a 2D tensor"),
        (torch.empty((3, 4), device="cuda"), "output shape must match hidden states"),
        (torch.empty_like(hidden_states, dtype=torch.float16), "output dtype must match hidden states"),
        (torch.empty((4, 2), device="cuda").transpose(0, 1), "output must be contiguous"),
        (torch.empty_like(hidden_states, device="cpu"), "output must be a CUDA tensor"),
    )
    for invalid_output, message in invalid_outputs:
        with pytest.raises(RuntimeError, match=message):
            torch.ops.xpool.ffn_shim(hidden_states, None, invalid_output, 0, 2, 1, 0)


@pytest.mark.requires_cuda(min_devices=2)
def test_transport_request_rejects_cross_device_tensors() -> None:
    current_device = torch.cuda.current_device()
    other_device = next(device for device in range(torch.cuda.device_count()) if device != current_device)
    hidden_states = torch.empty((2, 4), device=torch.device("cuda", current_device), dtype=torch.bfloat16)
    dp_rank_payload_rows = torch.tensor([2], device=torch.device("cuda", other_device), dtype=torch.int32)
    output = torch.empty_like(hidden_states)

    with pytest.raises(RuntimeError, match="DP-rank payload rows must be on the hidden-state device"):
        torch.ops.xpool.ffn_shim(hidden_states, dp_rank_payload_rows, output, 0, 2, 1, 1)

    other_output = torch.empty_like(hidden_states, device=torch.device("cuda", other_device))
    with pytest.raises(RuntimeError, match="output device must match hidden states"):
        torch.ops.xpool.ffn_shim(hidden_states, None, other_output, 0, 2, 1, 0)
