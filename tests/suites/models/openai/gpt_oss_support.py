"""GPT-OSS serving checks shared by the two concrete model qualifications."""

from __future__ import annotations

from pathlib import Path

import httpx
from pydantic import JsonValue

from xkit.serving.sglang.graph import SglangGraphMode
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving.probe import run_probe
from xtest.harness.sglang.serving.server import SglangInferenceRecord, SglangProbeServer, SglangServerResult
from xtest.harness.support.sglang.graph import assert_run_graph_evidence


def sliding_window_workload(servers: list[SglangProbeServer]) -> tuple[SglangServerResult, ...]:
    """Cross the 4096-token window and observe reuse through the public API."""

    for server in servers:
        with httpx.Client(base_url=server.url(), timeout=600) as client:
            for repetition in range(2):
                request: dict[str, JsonValue] = {
                    "rid": f"gpt-oss-window-{repetition}",
                    "input_ids": [44] * 4097,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 2, "min_new_tokens": 2, "ignore_eos": True},
                    "stream": False,
                }
                response = client.post("/generate", json=request)
                payload = response.json()
                SglangInferenceRecord(
                    model_id=server.process.model.model_id,
                    url=server.url() + "/generate",
                    request=request,
                    response_status_code=response.status_code,
                    response_content_type=response.headers.get("content-type"),
                    response=payload,
                ).write(server.inference_path.with_name(f"sliding-window-{repetition}.json"))
                response.raise_for_status()
                assert len(payload["output_ids"]) == 2
                assert payload["meta_info"]["prompt_tokens"] == 4097
                if repetition:
                    assert payload["meta_info"]["cached_tokens"] > 0
    return tuple(server.result() for server in servers)


def qualify_sliding_window(case: E2eServingCase, base_config: ResolvedConfig, workdir: Path) -> None:
    """Run full Decode and Breakable Prefill with a long repeated prefix."""

    run = run_probe(
        case,
        base_config=base_config,
        graph_settings=SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE.settings(),
        workdir=workdir,
        workload=sliding_window_workload,
    )
    assert_run_graph_evidence(run)
