from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.server_args import ServerArgs
from torch import nn

from tests.harness.support.sglang.fakes import FakeModelConfig, FakeModelRunner, server_args
from tests.harness.support.sglang.plugin import binding
from xpool.fabric import InstanceFfnLayerProfile
from xpool.integrations.sglang.adapter import SglangInstanceRankBinding
from xpool.integrations.sglang.plugin import derive_instance_ffn_profile
from xpool.integrations.sglang.shim import FfnShimModule
from xpool.native.ffn import LayerKind


def test_ffn_profile_includes_only_enabled_graph_capacities(tmp_path: Path) -> None:
    args = server_args(
        max_prefill_tokens=48,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="full", bs=[1, 64], max_bs=32),
            prefill=PhaseConfig(backend="breakable", bs=[64, 256], max_bs=128),
        ),
    )
    runner, model_binding = workload_inputs(
        tmp_path,
        args,
        max_running_requests=16,
    )
    config_bytes = (model_binding.model_path / "config.json").read_bytes()

    profile = derive_instance_ffn_profile(runner.as_model_runner(), model_binding, args)

    assert profile.model_config_digest == hashlib.sha256(config_bytes).hexdigest()
    assert profile.payload_dtype is torch.float16
    assert profile.hidden_size == 2048
    assert profile.layers == (
        InstanceFfnLayerProfile(layer_id=1, kind=LayerKind.MOE),
        InstanceFfnLayerProfile(layer_id=3, kind=LayerKind.DENSE),
    )
    assert profile.decode_payload_row_capacity == 64
    assert profile.prefill_payload_row_capacity == 256
    assert profile.group_sum_complete_admitted is False


def test_ffn_profile_ignores_retained_buckets_for_disabled_graph_paths(tmp_path: Path) -> None:
    args = server_args(
        max_prefill_tokens=33,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled", bs=[1024], max_bs=2048),
            prefill=PhaseConfig(backend="disabled", bs=[4096], max_bs=8192),
        ),
    )
    runner, model_binding = workload_inputs(
        tmp_path,
        args,
        max_running_requests=17,
    )

    profile = derive_instance_ffn_profile(runner.as_model_runner(), model_binding, args)

    assert profile.decode_payload_row_capacity == 17
    assert profile.prefill_payload_row_capacity == 33


@pytest.mark.parametrize(
    "args",
    [
        server_args(
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="full", bs=[1, 0]),
                prefill=PhaseConfig(backend="disabled"),
            )
        ),
        server_args(
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend="disabled"),
                prefill=PhaseConfig(backend="breakable", max_bs=0),
            )
        ),
    ],
)
def test_ffn_profile_rejects_invalid_enabled_graph_geometry(
    tmp_path: Path,
    args: ServerArgs,
) -> None:
    runner, model_binding = workload_inputs(tmp_path, args)

    with pytest.raises(RuntimeError, match="positive"):
        derive_instance_ffn_profile(runner.as_model_runner(), model_binding, args)


@pytest.mark.parametrize(
    ("max_running_requests", "max_prefill_tokens", "message"),
    [
        (True, 16, "eager decode rows"),
        (16, True, "eager prefill rows"),
    ],
)
def test_ffn_profile_rejects_boolean_eager_capacity(
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
        derive_instance_ffn_profile(runner.as_model_runner(), model_binding, args)


def workload_inputs(
    tmp_path: Path,
    args: ServerArgs,
    *,
    max_running_requests: int = 8,
) -> tuple[FakeModelRunner, SglangInstanceRankBinding]:
    model_path = tmp_path / "synthetic-model"
    model_path.mkdir(exist_ok=True)
    config_bytes = b'{"model_type":"synthetic"}'
    (model_path / "config.json").write_bytes(config_bytes)

    model = nn.Module()
    model.shims = nn.ModuleList(
        [
            FfnShimModule(layer_id=3, hidden_size=2048, layer_kind=LayerKind.DENSE),
            FfnShimModule(layer_id=1, hidden_size=2048, layer_kind=LayerKind.MOE),
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
