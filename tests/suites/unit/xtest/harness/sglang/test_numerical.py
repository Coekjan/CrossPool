"""Per-rank numerical comparison and retained evidence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import safetensors.torch
import torch

from xkit.network import TcpEndpointReservation
from xkit.serving.cluster import XpoolClusterLaunch
from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationId
from xpool.runtime.ffnagent.architecture import FfnSourceConfig
from xpool.runtime.ffnagent.models.llama import LlamaAdapter
from xpool.runtime.ffnagent.models.qwen3_moe import Qwen3MoeAdapter
from xtest.harness.native.ffn.protocol import FfnInstanceReady, FfnInstanceSpec
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import numerical
from xtest.harness.sglang.catalog import E2eFfnInputMatrix, E2eFfnNumericalCase
from xtest.harness.sglang.reference.ffn import (
    SglangFfnReferenceCase,
    SglangFfnReferenceResult,
    SglangFfnReferenceRunner,
    SglangFfnRoutingEvidence,
)
from xtest.harness.support.config import TEST_MODEL_ID, synthetic_config
from xtest.harness.support.ffn import FfnNumericalEvidence


@pytest.mark.parametrize(
    ("atn_tp_size", "bad_rank", "moe"),
    ((1, None, False), (2, None, False), (2, 1, False), (2, None, True)),
    ids=("tp1", "tp2", "tp2-rank1-mismatch", "tp2-shared-routing"),
)
def test_numerical_case_compares_and_retains_every_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atn_tp_size: int,
    bad_rank: int | None,
    moe: bool,
) -> None:
    dtype = torch.bfloat16 if atn_tp_size == 1 else torch.float16
    config = synthetic_config(
        atn_devices=tuple(range(atn_tp_size)),
        ffn_devices=tuple(range(atn_tp_size, atn_tp_size + 4)),
    )
    source = FfnSourceConfig(
        {
            "model_type": "qwen3_moe" if moe else "llama",
            "hidden_act": "silu",
            "torch_dtype": "bfloat16" if dtype == torch.bfloat16 else "float16",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "moe_intermediate_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
        }
    )
    adapter = Qwen3MoeAdapter if moe else LlamaAdapter
    model_spec = adapter.compile(model_id=TEST_MODEL_ID, model_config=source)
    case = E2eFfnNumericalCase(
        description="CPU comparison and evidence across attention TP ranks.",
        deployment=tmp_path / "deployment.toml",
        model_id=TEST_MODEL_ID,
        layer_ids=(0, 1),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 3), payload_dtype=dtype),
        estimated_duration_seconds=1,
        timeout_seconds=180,
    )
    generation = FabricGenerationId(high=1, low=1)
    endpoint = TcpEndpointReservation(address=("127.0.0.1", 32199), listener=None)
    routing: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def run_reference(
        self: SglangFfnReferenceRunner,
        *,
        model_path: Path,
        tensor_parallel_size: int,
        cases: Sequence[SglangFfnReferenceCase],
    ) -> tuple[SglangFfnReferenceResult, ...]:
        results = []
        for sequence, sample in enumerate(cases, start=1):
            route = None
            if moe:
                rows = sample.hidden_states.shape[0]
                ids = torch.tensor([[0, 1]], dtype=torch.int32).repeat(rows, 1)
                weights = torch.full((rows, 2), 0.5)
                routing[0, sequence, sample.layer_id] = (ids, weights)
                route = SglangFfnRoutingEvidence(ids, weights)
            results.append(
                SglangFfnReferenceResult(sample.case_id, sample.layer_id, sample.hidden_states.clone(), route)
            )
        return tuple(results)

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
        assert tuple(spec.rank for spec in instance_specs) == tuple(range(atn_tp_size))
        for spec in instance_specs:
            assert spec.transport.atn_tp_rank == spec.rank
            assert spec.transport.atn_tp_size == atn_tp_size
            for invocation in spec.invocations:
                output = safetensors.torch.load_file(invocation.input_path)["hidden_states"]
                if spec.rank == bad_rank:
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
    monkeypatch.setattr(numerical, "read_ffn_routing_records", lambda *args: routing)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    base = ResolvedConfig(path=tmp_path / "runtime.toml", config=config)
    if bad_rank is None:
        numerical.run_numerical_case(case, base, tmp_path, artifacts)
    else:
        with pytest.raises(AssertionError, match=f"attention rank {bad_rank}"):
            numerical.run_numerical_case(case, base, tmp_path, artifacts)

    for rank in range(atn_tp_size):
        name = "ffn-numerical.json" if rank == 0 else f"ffn-numerical-rank-{rank}.json"
        evidence = FfnNumericalEvidence.model_validate_json((artifacts / name).read_bytes())
        assert len(evidence.samples) == 4
        assert evidence.payload_dtype == dtype
        assert all(sample.output.passed == (rank != bad_rank) for sample in evidence.samples)
        assert all((sample.routing is not None) == moe for sample in evidence.samples)
        assert all(sample.routing is None or sample.routing.passed for sample in evidence.samples)
