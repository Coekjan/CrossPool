"""Explicit SGLang numerical and graph qualification for Qwen/Qwen3.5-9B."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import load_file
from tests import TEST_CATALOG_PATH

import xtest
from xkit.deployment import resolve_deployment_path
from xkit.results import write_json
from xkit.serving.sglang.graph import SglangGraphMode
from xpool.model import ModelId
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang import numerical
from xtest.harness.sglang.catalog import (
    E2eElasticKvWorkload,
    E2eFfnInputMatrix,
    E2eFfnNumericalCase,
    E2eServingCase,
    E2eServingInput,
)
from xtest.harness.sglang.reference.serving import run_reference
from xtest.harness.sglang.serving import qualification
from xtest.harness.sglang.serving.alignment import assert_prefill_alignment
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import SglangProbeServer, SglangServerResult
from xtest.harness.support.native.observer import (
    assert_fabric_observer_snapshots,
    assert_transport_observer_snapshots,
    assert_two_model_executor_overlap,
)
from xtest.harness.support.sglang.graph import assert_run_graph_evidence

pytest_plugins = ("xtest.harness.support.config",)

MODEL = ModelId("Qwen/Qwen3.5-9B")
NUMERICAL_CASES: tuple[E2eFfnNumericalCase, ...] = (
    E2eFfnNumericalCase(
        description="Representative real-checkpoint FFN numerical parity against the original SGLang implementation.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn1-lanes1"),
        model_id=MODEL,
        layer_ids=(0, 31),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 32, 129, 4096)),
        estimated_duration_seconds=600,
        timeout_seconds=1800,
    ),
)
MEDIA = Path(__file__).with_name("media")
MEDIA_PREFIX = "Observe the supplied media carefully. " * 120
SERVER_ARGUMENTS = (
    "--context-length",
    "4096",
    "--chunked-prefill-size",
    "256",
    "--max-running-requests",
    "2",
    "--max-mamba-cache-size",
    "16",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mm-process-config",
    '{"image":{"min_pixels":16384,"max_pixels":16384},"video":{"min_pixels":16384,"max_pixels":16384,"max_frames":4,"fps":2}}',
)
GRAPH_MODES = (
    SglangGraphMode.EAGER,
    SglangGraphMode.DECODE_FULL,
    SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE,
)


def media_prompt(placeholders: str, question: str) -> str:
    # The original tokenizer's chat template with enable_thinking=False.
    return (
        "<|im_start|>user\n"
        + placeholders
        + MEDIA_PREFIX
        + question
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


NATIVE_INPUTS = (
    E2eServingInput(
        text=media_prompt(
            "<|vision_start|><|image_pad|><|vision_end|>", "What color is the shape? Answer only red or blue."
        ),
        images=(MEDIA / "red-square.png",),
    ),
    E2eServingInput(
        text=media_prompt(
            "<|vision_start|><|image_pad|><|vision_end|>", "What color is the shape? Answer only red or blue."
        ),
        images=(MEDIA / "blue-circle.png",),
    ),
    E2eServingInput(
        text=media_prompt(
            "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>",
            "What color is the first shape? Answer only red or blue.",
        ),
        images=(MEDIA / "red-square.png", MEDIA / "blue-circle.png"),
    ),
    E2eServingInput(
        text=media_prompt(
            "<|vision_start|><|video_pad|><|vision_end|>",
            "What color appears first in this video? Answer only red or blue.",
        ),
        videos=(MEDIA / "red-blue.mp4",),
    ),
    E2eServingInput(
        text=media_prompt(
            "<|vision_start|><|video_pad|><|vision_end|>",
            "What color appears first in this video? Answer only red or blue.",
        ),
        videos=(MEDIA / "blue-red.mp4",),
    ),
)

SERVING_CASES: tuple[E2eServingCase, ...] = (
    E2eServingCase(
        description="Installed serving graph qualification with attention TP 1, DP 1 and FFN TP 1.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn1-lanes1"),
        models=(MODEL,),
        graph_modes=(
            SglangGraphMode.EAGER,
            SglangGraphMode.DECODE_FULL,
            SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE,
        ),
        estimated_duration_seconds=600,
        timeout_seconds=3600,
    ),
    E2eServingCase(
        description="Installed serving graph qualification with attention TP 2, DP 1 and FFN TP 1.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn2-ffn1-lanes1"),
        models=(MODEL,),
        graph_modes=(SglangGraphMode.DECODE_FULL,),
        estimated_duration_seconds=700,
        timeout_seconds=3600,
    ),
    *(
        E2eServingCase(
            description="Native encoded media, chunked Prefill and recurrent checkpoint graph parity.",
            deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn1-lanes1"),
            models=(MODEL,),
            graph_modes=GRAPH_MODES,
            native_input=native_input,
            server_arguments=SERVER_ARGUMENTS,
            estimated_duration_seconds=600,
            timeout_seconds=3600,
        )
        for native_input in NATIVE_INPUTS
    ),
    E2eServingCase(
        description="Attention DP replicas retain isolated linear state and exercise IDLE FFN rows.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn2-dp2-ffn1-lanes1"),
        models=(MODEL,),
        graph_modes=(SglangGraphMode.DECODE_FULL,),
        server_arguments=SERVER_ARGUMENTS,
        estimated_duration_seconds=600,
        timeout_seconds=3600,
    ),
)


@xtest.parameterize(
    "case",
    tuple(numerical.case_parameter(case) for case in NUMERICAL_CASES),
)
@xtest.requirements(numerical.requirements_of)
def test_ffn_numerical(
    case: E2eFfnNumericalCase,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Compare installed FFN with an independent SGLang reference."""

    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir)


@xtest.parameterize(
    ("case", "graph_mode"),
    SERVING_CASES,
    rows=partial(qualification.graph_rows, compare_modes=True),
)
@xtest.requirements(qualification.requirements_of)
def test_serving_graph(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove installed graph modes and their cross-mode alignment."""

    qualification.run_serving_case(
        case,
        graph_mode,
        e2e_base_config,
        tmp_path,
        task_artifact_dir,
        compare_modes=len(case.graph_modes) > 1,
    )


@xtest.requirements(device_count=2, requires_config=True, model_ids=(MODEL,))
@pytest.mark.timeout(1800)
def test_native_media_identity_and_original_sglang_parity(
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    # Media appears before the first recurrent checkpoint so content/order
    # changes exercise retained-state identity, including a changed text suffix.
    inputs = (
        *NATIVE_INPUTS,
        NATIVE_INPUTS[0].model_copy(
            update={
                "text": media_prompt(
                    "<|vision_start|><|image_pad|><|vision_end|>", "What is this shape? Answer only square or circle."
                )
            }
        ),
    )
    # Each fixture asks for one color/shape word. Compare that complete answer
    # to the original model; graph qualification separately checks eight Decode
    # tokens. Continuing past the answer would compare unsolicited formatting
    # after an intentionally suppressed EOS, rather than the requested answer.
    inputs = tuple(native.model_copy(update={"new_tokens": 1}) for native in inputs)
    expected = run_reference(
        model_id=MODEL,
        model_path=e2e_base_config.config.model_path_of(MODEL),
        inputs=inputs,
        server_arguments=SERVER_ARGUMENTS,
        workdir=tmp_path / "original",
        timeout_seconds=600,
    )
    case = E2eServingCase(
        description="Original native media reference, repeated checkpoint reuse and media namespace isolation.",
        deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL,), "atn1-ffn1-lanes1"),
        models=(MODEL,),
        graph_modes=(SglangGraphMode.EAGER,),
        server_arguments=SERVER_ARGUMENTS,
        estimated_duration_seconds=300,
        timeout_seconds=1800,
    )

    def requests(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
        server = servers[0]
        results = []
        for index in (3, 0, 1, 2, 4, 5):
            native, baseline = inputs[index], expected[index]
            for repeat in range(2):
                probe = replace(
                    server,
                    inference_path=server.inference_path.with_name(f"input-{index}-{repeat}.json"),
                    native_input=native,
                )
                result = probe.result()
                assert result.output_ids == baseline, (
                    f"native media {index}, repeat {repeat} differs from original SGLang"
                )
                evidence = json.loads(probe.inference_path.read_text())
                cached = evidence["response"]["meta_info"]["cached_tokens"]
                if repeat:
                    assert cached >= 512
                elif index < 5:
                    assert cached == 0
                results.append(result)
        # Releasing every checkpoint must preserve the cold numerical result.
        response = httpx.get(f"{server.url()}/flush_cache", timeout=30)
        response.raise_for_status()
        probe = replace(
            server, inference_path=server.inference_path.with_name("after-flush.json"), native_input=inputs[0]
        )
        assert probe.result().output_ids == expected[0]
        return (results[0],)

    run_probe(
        case,
        base_config=e2e_base_config,
        graph_settings=SglangGraphMode.EAGER.settings(),
        workdir=tmp_path / "production",
        workload=requests,
    )
    if task_artifact_dir is not None:
        shutil.copyfile(tmp_path / "original" / "reference.json", task_artifact_dir / "native-reference.json")


PRESSURE_MODEL = ModelId("Qwen/Qwen3-14B")
PRESSURE_CASE = E2eServingCase(
    description="Native visual input survives a full ordinary-model grant and recurrent checkpoint reclamation.",
    deployment=resolve_deployment_path(TEST_CATALOG_PATH, (MODEL, PRESSURE_MODEL), "atn1-ffn2-lanes2"),
    models=(MODEL, PRESSURE_MODEL),
    elastic_kv=E2eElasticKvWorkload(
        prefix_model_id=MODEL,
        prefix_tokens=1024,
        pressure_model_id=PRESSURE_MODEL,
        atn_device_memory_budget_bytes=28 << 30,
    ),
    server_arguments=(
        # Resolve the strategy per model: forcing the linear cache option
        # onto the ordinary peer would enable tracking without a state pool.
        *(arg for arg in SERVER_ARGUMENTS if arg not in ("--mamba-radix-cache-strategy", "extra_buffer")),
        "--context-length",
        "32768",
        "--chunked-prefill-size",
        "1024",
        "--cuda-graph-max-bs-prefill",
        "1024",
        "--mem-fraction-static",
        "0.96",
        "--max-mamba-cache-size",
        "32",
    ),
    estimated_duration_seconds=900,
    timeout_seconds=3600,
)


@xtest.requirements(device_count=3, requires_config=True, model_ids=(MODEL, PRESSURE_MODEL))
@pytest.mark.timeout(3600)
def test_native_media_under_peer_state_pressure(
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    observations = []

    def requests(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
        by_model = {server.process.model.model_id: server for server in servers}
        primary, peer = by_model[MODEL], by_model[PRESSURE_MODEL]
        results = (primary.result(), peer.result())
        info = httpx.get(f"{peer.url()}/server_info", timeout=30).json()
        token_count = info["max_req_input_len"] - 16
        native = NATIVE_INPUTS[2].model_copy(update={"new_tokens": 1})

        def media(index: int, value: E2eServingInput = native) -> tuple[int, torch.Tensor]:
            probe = replace(
                primary,
                inference_path=primary.inference_path.with_name(f"pressure-media-{index}.json"),
                native_input=value,
            )
            result = probe.result()
            record = json.loads(probe.inference_path.read_text())
            if result.prefill_logits_path is None:
                raise AssertionError(f"missing completed-Prefill logits for media request {index}")
            # The shared request ID is overwritten by later probes. Preserve
            # each completed request's evidence before starting its successor.
            logits_path = probe.inference_path.with_suffix(".prefill.safetensors")
            shutil.copyfile(result.prefill_logits_path, logits_path)
            logits = load_file(logits_path)["next_token_logits"]
            return record["response"]["meta_info"]["cached_tokens"], logits

        def pressure(index: int) -> None:
            response = httpx.post(
                f"{peer.url()}/generate",
                json={
                    "rid": f"state-pressure-{index}",
                    "input_ids": [index + 1] * token_count,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8, "min_new_tokens": 8, "ignore_eos": True},
                },
                timeout=300,
            )
            response.raise_for_status()
            assert len(response.json()["output_ids"]) == 8

        # Confirm a set of independent checkpoints. The donor may preserve
        # its newest prefix while evicting older checkpoints under pressure.
        confirmed = []
        for index in range(12):
            value = native.model_copy(update={"text": native.text.replace("Observe", f"Example {index}. Observe", 1)})
            _, expected = media(2 * index, value)
            warm, repeated = media(2 * index + 1, value)
            assert warm >= 512
            assert_prefill_alignment(expected, repeated)
            confirmed.append((value, warm, expected))
        pressure(0)
        victim = None
        for index, (value, warm, expected) in enumerate(confirmed):
            after, output = media(100 + index, value)
            assert_prefill_alignment(expected, output)
            if after < warm:
                victim = value, warm, after, expected
                break
        assert victim is not None, "peer pressure did not reclaim a confirmed checkpoint"
        value, warm, after, expected = victim
        restored, output = media(200, value)
        assert restored >= warm
        assert_prefill_alignment(expected, output)
        # Visual cache misses and both request streams run while the borrower
        # repeatedly consumes its ceiling. Source-owned bounds are the complete
        # image/video envelope exercised here, rather than arbitrary media size.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(pressure, 1)
            media(300, NATIVE_INPUTS[3])
            future.result()
        observations.append(
            dict(warm_tokens=warm, after_pressure_tokens=after, restored_tokens=restored, peer_input_tokens=token_count)
        )
        return results

    run = run_probe(
        PRESSURE_CASE,
        base_config=e2e_base_config,
        graph_settings=SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(),
        workdir=tmp_path,
        workload=requests,
    )
    assert_transport_observer_snapshots(
        run.observer_outdir,
        expected_count=PRESSURE_CASE.atnagent_count * len(PRESSURE_CASE.models),
        site="atnagent",
    )
    assert_fabric_observer_snapshots(
        run.observer_outdir,
        atnagent_count=PRESSURE_CASE.atnagent_count,
        ffnagent_count=PRESSURE_CASE.ffnagent_count,
        expected_topologies=tuple(
            (PRESSURE_CASE.atnagent_count // model.atn_dp_size, model.atn_dp_size)
            for model_id in PRESSURE_CASE.models
            for model in (PRESSURE_CASE.deployment_config.model_by_id[model_id],)
        ),
    )
    assert_two_model_executor_overlap(run.observer_outdir)
    assert_run_graph_evidence(run)
    if task_artifact_dir is not None:
        write_json(task_artifact_dir / "native-state-pressure.json", observations)
