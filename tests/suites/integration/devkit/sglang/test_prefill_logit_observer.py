from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput

import xpool.integrations.sglang.devkit.prefill_logit_observer
from tests.harness.support.config import install_test_config, reset_global_config
from xpool.config import XpoolConfig


@pytest.fixture
def reset_prefill_logit_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(xpool.integrations.sglang.devkit.prefill_logit_observer, "installed", False)
    monkeypatch.setattr(xpool.integrations.sglang.devkit.prefill_logit_observer, "captured", False)
    yield


pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__,
    reset_prefill_logit_observer.__name__,
)


def test_prefill_logit_observer_records_first_rank_zero_extend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    def forward(model_runner: ModelRunner, forward_batch: ForwardBatch) -> ModelRunnerOutput:
        return ModelRunnerOutput(LogitsProcessorOutput(logits), can_run_graph=False)

    monkeypatch.setattr(ModelRunner, "forward", forward)
    install_test_config(config=observer_config(tmp_path))
    xpool.integrations.sglang.devkit.prefill_logit_observer.install()
    runner = cast(ModelRunner, SimpleNamespace(ps=ParallelState.trivial()))

    ModelRunner.forward(runner, forward_batch(ForwardMode.DECODE, ["decode-rid"]))
    assert not tuple(tmp_path.glob("xpool.prefill-logits.*.safetensors"))

    ModelRunner.forward(runner, forward_batch(ForwardMode.EXTEND, ["warmup-rid"]))
    assert not tuple(tmp_path.glob("xpool.prefill-logits.*.safetensors"))

    ModelRunner.forward(runner, forward_batch(ForwardMode.EXTEND, ["xpool-serving-graph-request-rid"]))
    ModelRunner.forward(runner, forward_batch(ForwardMode.EXTEND, ["xpool-serving-graph-later-rid"]))

    (path,) = tuple(tmp_path.glob("xpool.prefill-logits.*.safetensors"))
    saved = load_file(path)
    assert tuple(saved) == ("next_token_logits",)
    torch.testing.assert_close(saved["next_token_logits"], logits)
    with safe_open(path, framework="pt", device="cpu") as file:
        assert file.metadata() == {"rid": "xpool-serving-graph-request-rid"}


def test_prefill_logit_observer_skips_nonzero_tp_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forward(model_runner: ModelRunner, forward_batch: ForwardBatch) -> ModelRunnerOutput:
        return ModelRunnerOutput(LogitsProcessorOutput(torch.ones((1, 3))), can_run_graph=False)

    monkeypatch.setattr(ModelRunner, "forward", forward)
    install_test_config(config=observer_config(tmp_path))
    xpool.integrations.sglang.devkit.prefill_logit_observer.install()

    ModelRunner.forward(
        cast(ModelRunner, SimpleNamespace(ps=ParallelState.trivial(tp_rank=1))),
        forward_batch(ForwardMode.EXTEND, ["xpool-serving-graph-request-rid"]),
    )

    assert not tuple(tmp_path.glob("xpool.prefill-logits.*.safetensors"))


def observer_config(outdir: Path) -> XpoolConfig:
    return XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_OUTDIR": str(outdir),
        },
    )


def forward_batch(mode: ForwardMode, rids: list[str]) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=mode,
        batch_size=len(rids),
        input_ids=torch.zeros(1, dtype=torch.int64),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int64),
        out_cache_loc=torch.zeros(1, dtype=torch.int64),
        seq_lens_sum=1,
        rids=rids,
    )
