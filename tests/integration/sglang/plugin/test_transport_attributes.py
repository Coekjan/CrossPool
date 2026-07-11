from __future__ import annotations

import torch

from tests.harness.sglang.fakes import FakeModelConfig, FakeModelRunner, server_args
from tests.harness.sglang.plugin import (
    binding,
    pytest,
    sglang_plugin,
)


@pytest.mark.parametrize(
    ("torch_dtype", "element_size"),
    [
        (torch.bfloat16, 2),
        (torch.float16, 2),
        (torch.float32, 4),
    ],
)
def test_transport_attributes_use_sglang_resolved_element_size(
    torch_dtype: torch.dtype,
    element_size: int,
) -> None:
    runner = FakeModelRunner(model_config=FakeModelConfig(dtype=torch_dtype))

    attributes = sglang_plugin.derive_transport_attributes(
        runner.as_model_runner(),
        binding(),
        server_args(max_prefill_tokens=8),
    )

    assert attributes.element_size == element_size


def test_transport_attributes_include_eager_decode_request_capacity() -> None:
    runner = FakeModelRunner(max_running_requests=32)

    attributes = sglang_plugin.derive_transport_attributes(
        runner.as_model_runner(),
        binding(),
        server_args(
            max_prefill_tokens=8,
            cuda_graph_max_bs=16,
            piecewise_cuda_graph_max_tokens=24,
        ),
    )

    assert attributes.max_tokens == 32


def test_transport_attributes_reject_non_positive_eager_decode_request_capacity() -> None:
    runner = FakeModelRunner(max_running_requests=0)

    with pytest.raises(RuntimeError, match="eager decode request capacity"):
        sglang_plugin.derive_transport_attributes(
            runner.as_model_runner(),
            binding(),
            server_args(max_prefill_tokens=8),
        )


def test_transport_attributes_reject_missing_torch_dtype() -> None:
    runner = FakeModelRunner()
    setattr(runner.model_config, "dtype", "float16")

    with pytest.raises(RuntimeError, match="transport element size"):
        sglang_plugin.derive_transport_attributes(
            runner.as_model_runner(),
            binding(),
            server_args(max_prefill_tokens=8),
        )


def test_transport_attributes_reject_missing_hidden_size() -> None:
    runner = FakeModelRunner(model_config=FakeModelConfig(hf_config=None))

    with pytest.raises(RuntimeError, match="transport hidden size"):
        sglang_plugin.derive_transport_attributes(
            runner.as_model_runner(),
            binding(),
            server_args(max_prefill_tokens=8),
        )
