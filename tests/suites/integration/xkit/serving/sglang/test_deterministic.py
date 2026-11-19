import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import OutputLogprobProcessor
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import APIServerReqTimeStats, ReqState, ServerStatus, TokenizerManager
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

import xkit.serving.sglang.deterministic
from xkit.envs import DETERMINISTIC_MODE, DETERMINISTIC_PATH
from xkit.serving.sglang.deterministic import DeterministicConfig, SamplingRecord
from xpool.config import XpoolConfig
from xtest.harness.support.config import TEST_MODEL_ID


def append_record(arguments: tuple[Path, int]) -> None:
    path, index = arguments
    SamplingRecord(
        TEST_MODEL_ID,
        f"request-{index}",
        {"text": "prompt", "sampling_params": {"max_new_tokens": 2}},
        1,
        (index, index + 1),
    ).append(path)


def test_concurrent_model_writers_preserve_complete_request_records(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    DeterministicConfig("record", path).prepare()
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
        tuple(executor.map(append_record, ((path, index) for index in range(12))))
    records = {record.request_id: record for record in SamplingRecord.read(path)}
    assert len(records) == 12
    for index in range(12):
        assert records[f"request-{index}"].output_ids == (index, index + 1)


@pytest.mark.parametrize("return_logprob", [False, True])
def test_real_logprob_processor_preserves_distributions_when_replay_changes_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, return_logprob: bool
) -> None:
    path = tmp_path / "reference.jsonl"
    DeterministicConfig("record", path).prepare()
    for request_id in ("active-a", "active-b", "lookahead"):
        SamplingRecord(
            TEST_MODEL_ID,
            request_id,
            {"input_ids": [1, 2], "sampling_params": {}},
            2,
            (3, 2),
        ).append(path)
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": str(tmp_path)}],
        },
        env={},
    )
    monkeypatch.setattr(xkit.serving.sglang.deterministic, "get_global_config", lambda: config)
    monkeypatch.setenv(DETERMINISTIC_MODE.name, "replay")
    monkeypatch.setenv(DETERMINISTIC_PATH.name, str(path))
    # HookRegistry is a dynamic engine boundary; exercise its registered handlers with real CPU values.
    hooks: dict[str, Callable[..., object]] = {}

    def register(cls: type[HookRegistry], target: str, handler: Callable[..., object], kind: HookType) -> None:
        del cls, kind
        hooks[target.rsplit(".", 1)[-1]] = handler

    monkeypatch.setattr(HookRegistry, "register", classmethod(register))
    xkit.serving.sglang.deterministic.install()
    original_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]] * 4)
    logits = LogitsProcessorOutput(next_token_logits=original_logits.clone())
    info = SamplingBatchInfo(
        torch.ones((4, 1)),
        torch.ones(4),
        torch.ones(4, dtype=torch.int32),
        torch.zeros(4),
        True,
        True,
        False,
        False,
        False,
        4,
        device="cpu",
    )
    batch = ForwardBatch(
        ForwardMode.DECODE,
        4,
        torch.ones(4, dtype=torch.int64),
        torch.arange(4),
        torch.tensor([2, 2, 3, 4]),
        torch.arange(4),
        11,
        rids=["unrelated", "active-a", "active-b", "lookahead"],
        positions=torch.tensor([1, 1, 2, 3]),
        sampling_info=info,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model_path=str(tmp_path), vocab_size=4), ps=SimpleNamespace(tp_rank=0)
    )
    observed: list[torch.Tensor] = []

    def original(runner: ModelRunner, output: LogitsProcessorOutput, batch: ForwardBatch) -> torch.Tensor:
        del runner, batch
        ids = original_logits.argmax(-1)
        if return_logprob:
            logprobs = torch.log_softmax(original_logits, -1)
            before = logprobs.clone()
            processor = OutputLogprobProcessor()
            result = processor.compute_logprobs(logprobs, [1] * 4, [[0]] * 4, ids)
            hooks["compute_logprobs"](result, processor, logprobs, [1] * 4, [[0]] * 4, ids)
            result.write_output_to(output)
            torch.testing.assert_close(logprobs, before, rtol=0, atol=0)
            observed.append(cast(torch.Tensor, result.token_logprobs).clone())
            assert result.token_ids_logprobs_val is not None
            torch.testing.assert_close(result.token_ids_logprobs_val[1], before[1, :1])
        hooks["forward"](
            ids, cast(Sampler, object()), output, info, return_logprob, [1] * 4, [[0]] * 4, torch.tensor([1, 1, 2, 3])
        )
        return ids

    actual = hooks["sample"](original, cast(ModelRunner, runner), logits, batch)
    torch.testing.assert_close(cast(torch.Tensor, actual), torch.tensor([0, 3, 2, 0]))
    torch.testing.assert_close(logits.next_token_logits, original_logits, rtol=0, atol=0)
    if return_logprob:
        expected = torch.log_softmax(original_logits, -1)[torch.arange(4), torch.tensor([0, 3, 2, 0])]
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize(("mode", "extra_output"), [("record", False), ("replay", False), ("replay", True)])
def test_native_response_recording_and_replay_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Literal["record", "replay"], extra_output: bool
) -> None:
    path = tmp_path / "reference.jsonl"
    DeterministicConfig("record", path).prepare()
    record = SamplingRecord(
        TEST_MODEL_ID, "request", {"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 2}}, 2, (3, 4)
    )
    if mode == "replay":
        record.append(path)
    before = path.read_bytes()
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": str(tmp_path)}],
        },
        env={},
    )
    monkeypatch.setattr(xkit.serving.sglang.deterministic, "get_global_config", lambda: config)
    monkeypatch.setenv(DETERMINISTIC_MODE.name, mode)
    monkeypatch.setenv(DETERMINISTIC_PATH.name, str(path))
    hooks: dict[str, Callable[..., object]] = {}

    def register(cls: type[HookRegistry], target: str, handler: Callable[..., object], kind: HookType) -> None:
        del cls, kind
        hooks[target.rsplit(".", 1)[-1]] = handler

    monkeypatch.setattr(HookRegistry, "register", classmethod(register))
    xkit.serving.sglang.deterministic.install()
    obj = GenerateReqInput(input_ids=[1, 2], rid="request", sampling_params={"max_new_tokens": 2}, return_logprob=True)
    native: dict[str, object] = {
        "output_ids": [3, 4, 5] if extra_output else [3, 4],
        "meta_info": {
            "prompt_tokens": 2,
            "finish_reason": {"type": "length", "length": 2},
            "output_token_logprobs": [(-0.5, 3, None), (-1.0, 4, None)],
        },
    }

    async def source(*args: object) -> AsyncIterator[dict[str, object]]:
        del args
        yield native

    async def consume() -> None:
        manager = SimpleNamespace(server_status=ServerStatus.Up, model_config=SimpleNamespace(model_path=str(tmp_path)))
        state = ReqState([], True, asyncio.Event(), obj, APIServerReqTimeStats())
        response = cast(
            AsyncIterator[dict[str, object]],
            hooks["_stream_one_response"](
                source,
                cast(TokenizerManager, manager),
                obj,
                state,
            ),
        )
        if extra_output:
            with pytest.raises(ValueError, match="mismatched sampler replay"):
                await anext(response)
        else:
            assert await anext(response) == native
        assert tuple(SamplingRecord.read(path)) == (record,)

    asyncio.run(consume())
    if mode == "replay":
        assert path.read_bytes() == before
