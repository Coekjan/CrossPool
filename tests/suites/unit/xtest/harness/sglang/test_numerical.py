"""Numerical qualification topology construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xpool.config import XpoolConfig
from xpool.fabric import InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.model import ModelId
from xpool.native.ffn import DpRowLayout, ForwardMode, LayerKind, OutputRequirement
from xtest.harness.native.ffn.protocol import FfnInvocationSpec
from xtest.harness.sglang import numerical

MODEL_ID = ModelId("test/numerical")


def test_numerical_instance_specs_cover_every_attention_tp_rank(tmp_path: Path) -> None:
    config = numerical_config(tmp_path, atn_tp_size=2, atn_dp_size=1)
    profile = numerical_profile()
    invocations = tuple((numerical_invocation(tmp_path, rank),) for rank in range(2))

    specs = numerical.materialize_numerical_instance_specs(
        config=config,
        model_id=MODEL_ID,
        environment={"QUALIFICATION": "1"},
        ffn_tp_size=4,
        ffn_profile=profile,
        payload_row_capacity=32,
        invocations_by_rank=invocations,
    )

    assert tuple(spec.rank for spec in specs) == (0, 1)
    assert tuple(spec.transport.atn_tp_rank for spec in specs) == (0, 1)
    assert all(spec.transport.atn_tp_size == 2 for spec in specs)
    assert all(spec.transport.atn_dp_rank == 0 and spec.transport.atn_dp_size == 1 for spec in specs)
    assert tuple(spec.invocations for spec in specs) == invocations
    assert specs[0].environment == specs[1].environment == {"QUALIFICATION": "1"}
    assert specs[0].environment is not specs[1].environment


def test_numerical_instance_specs_require_every_attention_rank(tmp_path: Path) -> None:
    config = numerical_config(tmp_path, atn_tp_size=2, atn_dp_size=1)

    with pytest.raises(ValueError, match="cover every attention rank"):
        numerical.materialize_numerical_instance_specs(
            config=config,
            model_id=MODEL_ID,
            environment={},
            ffn_tp_size=4,
            ffn_profile=numerical_profile(),
            payload_row_capacity=32,
            invocations_by_rank=((numerical_invocation(tmp_path, 0),),),
        )


def test_numerical_instance_specs_reject_attention_dp(tmp_path: Path) -> None:
    config = numerical_config(tmp_path, atn_tp_size=1, atn_dp_size=2)

    with pytest.raises(ValueError, match="requires attention DP 1"):
        numerical.materialize_numerical_instance_specs(
            config=config,
            model_id=MODEL_ID,
            environment={},
            ffn_tp_size=4,
            ffn_profile=numerical_profile(),
            payload_row_capacity=32,
            invocations_by_rank=tuple((numerical_invocation(tmp_path, rank),) for rank in range(2)),
        )


def numerical_config(tmp_path: Path, *, atn_tp_size: int, atn_dp_size: int) -> XpoolConfig:
    """Build a CPU-only numerical deployment config."""

    return XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": list(range(atn_tp_size * atn_dp_size))},
            "ffn": {"devices": [2, 3, 4, 5]},
            "models": [
                {
                    "id": str(MODEL_ID),
                    "path": str(tmp_path / "model"),
                    "atn_tp_size": atn_tp_size,
                    "atn_dp_size": atn_dp_size,
                    "ffn_tp_size": 4,
                }
            ],
        }
    )


def numerical_profile() -> InstanceFfnProfile:
    """Build the immutable profile used by topology-only unit checks."""

    return InstanceFfnProfile(
        payload_dtype=torch.float16,
        hidden_size=16,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=32,
        group_sum_complete_admitted=False,
    )


def numerical_invocation(tmp_path: Path, rank: int) -> FfnInvocationSpec:
    """Build one rank-local invocation without executing CUDA."""

    return FfnInvocationSpec(
        case_id="layer-0-rows-1",
        input_path=tmp_path / "input.safetensors",
        output_path=tmp_path / f"output-rank-{rank}.safetensors",
        layer_ordinal=0,
        forward_mode=ForwardMode.DECODE,
        output_requirement=OutputRequirement.PER_RANK_COMPLETE,
        dp_row_layout=DpRowLayout.NONE,
        dp_rank_payload_rows=None,
    )
