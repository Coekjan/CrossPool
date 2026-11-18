import json
from collections.abc import Callable
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Literal, cast

import httpx
import pytest
import tomli_w
import torch
from pydantic import JsonValue, TypeAdapter
from safetensors.torch import save_file

import xtest.harness.sglang.serving.qualification
from xkit.process import OwnedProcessGroup
from xkit.serving.sglang.deterministic import SamplerReplayEvidence
from xkit.serving.sglang.endpoints import SglangEndpointFamily
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xkit.serving.sglang.launch import SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xpool.config import XpoolConfig
from xpool.integrations.sglang.devkit.requests import LocalModelMetadata, SglangRequestLimits
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving.attempt import ProbeRun
from xtest.harness.sglang.serving.qualification import model_rows, run_model_case
from xtest.harness.sglang.serving.server import SglangProbeServer, SglangServerResult
from xtest.harness.support.config import TEST_MODEL_ID


@pytest.mark.parametrize(
    ("failure", "prefill_difference"),
    [(None, False), (None, True), ("prefill", False), ("decode", True), ("control", False)],
)
def test_model_case_replays_same_phase_and_enforces_kl_without_runner_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal["prefill", "decode", "control"] | None,
    prefill_difference: bool,
) -> None:
    modes = tuple(SglangGraphMode)
    case = E2eServingCase(
        description="One case owns cross-mode numerical verdicts.",
        deployment=tmp_path / "deployment.toml",
        models=(TEST_MODEL_ID,),
        graph_modes=modes,
        estimated_duration_seconds=1,
        timeout_seconds=10,
    )
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": str(tmp_path)}],
        },
        env={},
    )
    metadata = LocalModelMetadata(
        vocab_size=2, limits=SglangRequestLimits.from_context(128), admissible_ids=(0, 1), text_lengths={}
    )
    monkeypatch.setattr(xtest.harness.sglang.serving.qualification, "resolve_model_weights", lambda *args: tmp_path)
    monkeypatch.setattr(LocalModelMetadata, "from_checkpoint", classmethod(lambda cls, *args, **kwargs: metadata))
    monkeypatch.setattr(xtest.harness.sglang.serving.qualification, "assert_serving_evidence", lambda *args: None)
    case.deployment.write_text(
        tomli_w.dumps(
            {
                "scheduler": {"ffn_concurrency": 1, "slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "atn_tp_size": 1, "atn_dp_size": 1, "ffn_tp_size": 1}],
            }
        )
    )
    client_type = httpx.Client
    monkeypatch.setattr(
        xtest.harness.sglang.serving.qualification.httpx,
        "Client",
        lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    executed: list[SglangGraphSettings] = []
    requests: list[dict[str, JsonValue]] = []

    def generate(
        server: SglangProbeServer,
        request: dict[str, JsonValue],
        *,
        record_path: Path,
        request_barrier: Barrier | None = None,
    ) -> SglangServerResult:
        del request_barrier
        requests.append(request)
        input_ids = (1, 2)
        output_ids = (0, 1) * 4
        values = torch.zeros((8, 2))
        settings = server.process.model.graph_settings
        if prefill_difference and settings.prefill_backend == "breakable":
            values[:] = torch.tensor([0.1, -0.1])
        if failure == "prefill" and settings.prefill_backend == "breakable":
            values[0] = torch.tensor([8.0, -8.0])
        if failure == "decode" and settings == SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings():
            values[1] = torch.tensor([8.0, -8.0])
        record_path.parent.mkdir(parents=True, exist_ok=True)
        path = record_path.with_suffix(".safetensors")
        save_file({"logprobs": values}, path)
        return SglangServerResult(
            TEST_MODEL_ID,
            settings,
            str(request["rid"]),
            input_ids,
            output_ids,
            0,
            record_path,
            path,
        )

    monkeypatch.setattr(SglangProbeServer, "generate", generate)

    def run_probe(
        case: E2eServingCase,
        *,
        base_config: ResolvedConfig,
        graph_settings: SglangGraphSettings,
        workdir: Path,
        workload: Callable[[list[SglangProbeServer]], tuple[SglangServerResult, ...]],
        deterministic_mode: Literal["none", "record", "replay"] = "none",
        replay_path: Path | None = None,
    ) -> ProbeRun:
        del case, base_config
        executed.append(graph_settings)
        is_eager = graph_settings == SglangGraphMode.EAGER.settings()
        assert deterministic_mode == ("record" if is_eager else "replay")
        if not is_eager:
            assert replay_path == tmp_path / SglangGraphMode.EAGER.settings().id() / "sampling-record.jsonl"
        process = SglangServerProcess(
            SglangLaunchModel(TEST_MODEL_ID, graph_settings),
            cast(OwnedProcessGroup, SimpleNamespace()),
            SglangEndpointFamily("127.0.0.1", 19000, 19001, 19002, 1),
        )
        results = workload([SglangProbeServer(process)])
        observer = workdir / "observers"
        observer.mkdir(parents=True)
        if not is_eager and failure != "control":
            result = results[0]
            (observer / "xkit.sampler-replay.1.jsonl").write_bytes(
                b"".join(
                    TypeAdapter(SamplerReplayEvidence).dump_json(
                        SamplerReplayEvidence(
                            TEST_MODEL_ID,
                            result.request_id,
                            0,
                            prefix,
                            "extend" if index == 0 else "decode",
                            token,
                            token,
                        )
                    )
                    + b"\n"
                    for index, (prefix, token) in enumerate(zip(result.prefix_lengths, result.output_ids, strict=True))
                )
            )
        return ProbeRun(
            (TEST_MODEL_ID,),
            observer,
            graph_settings,
            results,
            [],
            0.1,
            1.0,
            workdir / "sampling-record.jsonl" if is_eager else replay_path,
        )

    monkeypatch.setattr(xtest.harness.sglang.serving.qualification, "run_probe", run_probe)
    if failure is not None:
        message = {"prefill": "prefix 2", "decode": "prefix 3", "control": "routed TP group"}[failure]
        with pytest.raises(AssertionError, match=message):
            run_model_case(case, ResolvedConfig(tmp_path / "config.toml", config), tmp_path, None)
    else:
        run_model_case(case, ResolvedConfig(tmp_path / "config.toml", config), tmp_path, None)

    expected_settings = [
        SglangGraphMode.EAGER.settings(),
        SglangGraphMode.DECODE_FULL.settings(),
        SglangGraphSettings("disabled", "breakable"),
        SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(),
    ]
    assert executed == (expected_settings[:2] if failure == "control" else expected_settings)
    assert len({str(request["rid"]) for request in requests}) == 1
    assert all(cast(dict, request["sampling_params"])["max_new_tokens"] == 8 for request in requests)
    if failure != "control":
        report = json.loads((tmp_path / "numerical/numerical.json").read_text())
        decode = next(
            entry
            for entry in report["stage_comparisons"]
            if entry["phase"] == "decode" and entry["candidate"]["prefill_backend"] == "breakable"
        )
        assert decode["reference"] == {"decode_backend": "disabled", "prefill_backend": "breakable"}
        assert decode["prefix_lengths"] == list(range(3, 10))
        assert decode["passed"] is (failure != "decode")
        if prefill_difference:
            combined = report["end_to_end_comparisons"][-1]
            assert combined["decode_passed"] is False
        if failure is None:
            assert all(entry["passed"] for entry in report["stage_comparisons"])


@pytest.mark.parametrize("modes", [tuple(SglangGraphMode)[:2], tuple(SglangGraphMode)])
def test_model_rows_budget_includes_missing_stage_references(
    tmp_path: Path, modes: tuple[SglangGraphMode, ...]
) -> None:
    case = E2eServingCase(
        description="Deployment time includes each required numerical reference.",
        deployment=tmp_path / "deployment.toml",
        models=(TEST_MODEL_ID,),
        graph_modes=modes,
        estimated_duration_seconds=2,
        timeout_seconds=10,
    )
    (row,) = model_rows(case)
    deployments = 2 if len(modes) == 2 else 4
    assert next(mark for mark in row.marks if mark.name == "timeout").args == (10 * deployments,)
    assert next(mark for mark in row.marks if mark.name == "estimated_duration").kwargs == {"seconds": 2 * deployments}
