"""Numerical qualification topology construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import safetensors.torch
import torch

from xkit.network import TcpEndpointReservation
from xkit.serving.cluster import XpoolClusterLaunch
from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationId, InstanceFfnLayerProfile, InstanceFfnProfile
from xpool.model import ModelId
from xpool.native.ffn import DpRowLayout, ForwardMode, LayerKind, OutputRequirement
from xpool.runtime.ffnagent.architecture import FfnSourceConfig
from xpool.runtime.ffnagent.models.llama import LlamaAdapter
from xtest.harness.native.ffn.protocol import FfnInstanceReady, FfnInstanceSpec, FfnInvocationSpec
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import numerical
from xtest.harness.sglang.catalog import E2eFfnInputMatrix, E2eFfnNumericalCase
from xtest.harness.sglang.reference.ffn import (
    SglangFfnReferenceCase,
    SglangFfnReferenceResult,
    SglangFfnReferenceRunner,
)
from xtest.harness.support.ffn import FFN_NUMERICAL_ARTIFACT_FILENAME, FfnNumericalEvidence

MODEL_ID = ModelId("test/numerical")


@pytest.mark.parametrize(
    ("atn_tp_size", "bad_ranks", "dtype"),
    (
        pytest.param(1, (), torch.bfloat16, id="tp1-pass"),
        pytest.param(1, (0,), torch.float16, id="tp1-fail"),
        pytest.param(2, (), torch.float16, id="tp2-pass"),
        pytest.param(2, (0,), torch.bfloat16, id="tp2-rank0-fail"),
        pytest.param(2, (1,), torch.float16, id="tp2-rank1-fail"),
        pytest.param(2, (0, 1), torch.float16, id="tp2-both-fail"),
    ),
)
def test_numerical_case_retains_every_rank_before_rejecting_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atn_tp_size: int,
    bad_ranks: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Exercise real comparison and evidence I/O with CPU-only execution substitutes."""

    config = numerical_config(tmp_path, atn_tp_size=atn_tp_size, atn_dp_size=1)
    model_spec = LlamaAdapter.compile(
        model_id=MODEL_ID,
        model_config=FfnSourceConfig(
            {
                "model_type": "llama",
                "hidden_act": "silu",
                "torch_dtype": "bfloat16" if dtype == torch.bfloat16 else "float16",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
            }
        ),
    )
    case = E2eFfnNumericalCase(
        description="CPU evidence persistence across attention ranks.",
        deployment=tmp_path / "deployment.toml",
        model_id=MODEL_ID,
        layer_ids=(0, 1),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 3), payload_dtype=dtype),
        estimated_duration_seconds=1,
        timeout_seconds=180,
    )
    generation = FabricGenerationId(high=1, low=1)
    endpoint = TcpEndpointReservation(address=("127.0.0.1", 32199), listener=None)

    def run_reference(
        self: SglangFfnReferenceRunner,
        *,
        model_path: Path,
        tensor_parallel_size: int,
        cases: Sequence[SglangFfnReferenceCase],
    ) -> tuple[SglangFfnReferenceResult, ...]:
        return tuple(
            SglangFfnReferenceResult(
                case_id=case.case_id, layer_id=case.layer_id, output=case.hidden_states.clone(), routing=None
            )
            for case in cases
        )

    def materialize_launch(*, config: XpoolConfig, daemon_port: int, workdir: Path) -> XpoolClusterLaunch:
        workdir.mkdir()
        return XpoolClusterLaunch(config=config, config_path=workdir / "xpool.toml", environment={}, cwd=workdir)

    def run_topology(
        *,
        launch: XpoolClusterLaunch,
        endpoint: TcpEndpointReservation,
        instance_specs: tuple[FfnInstanceSpec, ...],
        workdir: Path,
        timeout_seconds: float,
    ) -> tuple[FfnInstanceReady, ...]:
        for instance in instance_specs:
            for invocation in instance.invocations:
                output = safetensors.torch.load_file(invocation.input_path)["hidden_states"]
                if instance.rank in bad_ranks and invocation.case_id == "layer-0-rows-1":
                    output = output * 2
                safetensors.torch.save_file({"hidden_states": output}, invocation.output_path)
        return tuple(
            FfnInstanceReady(generation=generation, instance_index=0, layer_execution_groups=((0, 1, 2, 3),) * 2)
            for _ in instance_specs
        )

    monkeypatch.setattr(numerical, "assemble_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(numerical.architecture, "load", lambda **kwargs: model_spec)
    monkeypatch.setattr(SglangFfnReferenceRunner, "run", run_reference)
    monkeypatch.setattr(numerical.TcpEndpointReservation, "reserve", lambda *args, **kwargs: endpoint)
    monkeypatch.setattr(numerical, "materialize_ffn_cluster_launch", materialize_launch)
    monkeypatch.setattr(numerical, "run_ffn_topology", run_topology)
    monkeypatch.setattr(numerical, "read_ffn_routing_records", lambda *args: {})
    base_config = ResolvedConfig(path=tmp_path / "runtime.toml", config=config)
    task_artifact_dir = tmp_path / "artifacts" if atn_tp_size == 2 else None
    if task_artifact_dir is not None:
        task_artifact_dir.mkdir()

    if bad_ranks:
        with pytest.raises(AssertionError, match="failed numerical comparison") as failure:
            numerical.run_numerical_case(case, base_config, tmp_path, task_artifact_dir)
        for rank in bad_ranks:
            assert f"rank {rank} layer-0-rows-1" in str(failure.value)
    else:
        numerical.run_numerical_case(case, base_config, tmp_path, task_artifact_dir)

    for rank in range(atn_tp_size):
        filename = FFN_NUMERICAL_ARTIFACT_FILENAME if rank == 0 else f"ffn-numerical-rank-{rank}.json"
        evidence = FfnNumericalEvidence.model_validate_json(((task_artifact_dir or tmp_path) / filename).read_bytes())
        assert evidence.model_id == MODEL_ID
        assert evidence.model_spec_digest == model_spec.digest()
        assert evidence.generation == generation
        assert evidence.payload_dtype == dtype
        assert tuple((sample.layer_id, sample.row_count) for sample in evidence.samples) == (
            (0, 1),
            (0, 3),
            (1, 1),
            (1, 3),
        )
        assert tuple(sample.output.passed for sample in evidence.samples) == (rank not in bad_ranks, True, True, True)
        assert evidence.samples[0].output.normalized_rmse == pytest.approx(1 if rank in bad_ranks else 0)


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

    atn_world_size = atn_tp_size * atn_dp_size
    return XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": list(range(atn_world_size))},
            "ffn": {"devices": list(range(atn_world_size, atn_world_size + 4))},
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
