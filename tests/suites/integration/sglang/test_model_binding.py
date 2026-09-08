from __future__ import annotations

from dataclasses import replace

import pytest
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import get_context

from tests.harness.support.sglang.plugin import binding
from tests.harness.support.sglang.runtime import published_sglang_config


def test_model_binding_accepts_complete_tp_fastest_result_group() -> None:
    model_binding = replace(
        binding(),
        worker_rank=1,
        worker_world_size=2,
    )

    model_binding.validate_result_group(result_group(ranks=[0, 1], rank=1, rank_in_group=1))


@pytest.mark.parametrize(
    "ranks",
    [
        [1, 0],
        [0],
    ],
)
def test_model_binding_rejects_reordered_or_incomplete_result_group(ranks: list[int]) -> None:
    model_binding = replace(binding(), worker_world_size=2)

    with pytest.raises(RuntimeError, match="result-group ranks"):
        model_binding.validate_result_group(result_group(ranks=ranks, rank=0, rank_in_group=0))


@pytest.mark.usefixtures(published_sglang_config.__name__)
def test_model_binding_accepts_breakable_graphs_for_attention_dp() -> None:
    model_binding = replace(
        binding(),
        worker_world_size=2,
        atn_tp_size=1,
        atn_dp_size=2,
    )
    get_context().override(
        "test",
        tp_size=2,
        dp_size=2,
        enable_dp_attention=True,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend="disabled"),
            prefill=PhaseConfig(backend="breakable"),
        ),
    )

    model_binding.validate_server_args()


def result_group(
    *,
    ranks: list[int],
    rank: int,
    rank_in_group: int,
) -> GroupCoordinator:
    """Build a concrete coordinator value without initializing process groups."""

    group = GroupCoordinator.__new__(GroupCoordinator)
    group.ranks = ranks
    group.world_size = len(ranks)
    group.rank = rank
    group.rank_in_group = rank_in_group
    return group
