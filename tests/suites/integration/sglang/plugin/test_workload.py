from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from sglang.srt.server_args import ServerArgs
from torch import nn

from tests.harness.support.sglang.fakes import FakeModelConfig, FakeModelRunner, server_args
from tests.harness.support.sglang.plugin import binding
from xpool.abi import TensorDType
from xpool.fabric import FfnLayerKind, FfnLayerSpec
from xpool.integrations.sglang.adapter import XpoolModelBinding
from xpool.integrations.sglang.plugin import derive_workload
from xpool.integrations.sglang.shim import FfnShimModule


def test_workload_includes_only_enabled_graph_capacities(tmp_path: Path) -> None:
    args = server_args(
        max_prefill_tokens=48,
        disable_cuda_graph=False,
        cuda_graph_bs=[1, 64],
        cuda_graph_max_bs=32,
        disable_piecewise_cuda_graph=False,
        piecewise_cuda_graph_tokens=[64, 256],
        piecewise_cuda_graph_max_tokens=128,
    )
    runner, model_binding = workload_inputs(
        tmp_path,
        args,
        max_running_requests=16,
    )
    config_bytes = (model_binding.model_path / "config.json").read_bytes()

    workload = derive_workload(runner.as_model_runner(), model_binding, args)

    assert workload.model_config_digest == hashlib.sha256(config_bytes).hexdigest()
    assert workload.dtype is TensorDType.FP16
    assert workload.hidden_size == 2048
    assert workload.layers == (
        FfnLayerSpec(layer_id=1, kind=FfnLayerKind.SPARSE),
        FfnLayerSpec(layer_id=3, kind=FfnLayerKind.DENSE),
    )
    assert workload.max_decode_rows == 64
    assert workload.max_prefill_rows == 256


def test_workload_ignores_retained_buckets_for_disabled_graph_paths(tmp_path: Path) -> None:
    args = server_args(
        max_prefill_tokens=33,
        disable_cuda_graph=True,
        cuda_graph_bs=[1024],
        cuda_graph_max_bs=2048,
        disable_piecewise_cuda_graph=True,
        piecewise_cuda_graph_tokens=[4096],
        piecewise_cuda_graph_max_tokens=8192,
    )
    runner, model_binding = workload_inputs(
        tmp_path,
        args,
        max_running_requests=17,
    )

    workload = derive_workload(runner.as_model_runner(), model_binding, args)

    assert workload.max_decode_rows == 17
    assert workload.max_prefill_rows == 33


@pytest.mark.parametrize(
    "args",
    [
        server_args(disable_cuda_graph=False, cuda_graph_bs=[1, 0]),
        server_args(
            disable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_max_tokens=0,
        ),
    ],
)
def test_workload_rejects_invalid_enabled_graph_geometry(
    tmp_path: Path,
    args: ServerArgs,
) -> None:
    runner, model_binding = workload_inputs(tmp_path, args)

    with pytest.raises(RuntimeError, match="positive"):
        derive_workload(runner.as_model_runner(), model_binding, args)


@pytest.mark.parametrize(
    ("max_running_requests", "max_prefill_tokens", "message"),
    [
        (True, 16, "eager decode rows"),
        (16, True, "eager prefill rows"),
    ],
)
def test_workload_rejects_boolean_eager_capacity(
    tmp_path: Path,
    max_running_requests: int,
    max_prefill_tokens: int,
    message: str,
) -> None:
    args = server_args(max_prefill_tokens=max_prefill_tokens)
    runner, model_binding = workload_inputs(
        tmp_path,
        args,
        max_running_requests=max_running_requests,
    )

    with pytest.raises(RuntimeError, match=message):
        derive_workload(runner.as_model_runner(), model_binding, args)


def workload_inputs(
    tmp_path: Path,
    args: ServerArgs,
    *,
    max_running_requests: int = 8,
) -> tuple[FakeModelRunner, XpoolModelBinding]:
    model_path = tmp_path / "synthetic-model"
    model_path.mkdir(exist_ok=True)
    config_bytes = b'{"model_type":"synthetic"}'
    (model_path / "config.json").write_bytes(config_bytes)

    model = nn.Module()
    model.shims = nn.ModuleList(
        [
            FfnShimModule(layer_id=3, hidden_size=2048, layer_kind=FfnLayerKind.DENSE),
            FfnShimModule(layer_id=1, hidden_size=2048, layer_kind=FfnLayerKind.SPARSE),
        ]
    )
    runner = FakeModelRunner(
        model_config=FakeModelConfig(model_path=str(model_path), dtype=torch.float16),
        server_args=args,
        max_running_requests=max_running_requests,
        model=model,
    )
    model_binding = replace(
        binding(),
        instance_id=model_path.name,
        model_path=model_path,
    )
    return runner, model_binding
