from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tests.harness.sglang.graph import read_graph_events
from tests.harness.sglang.offline_probe import (
    ProbeResult,
    SglangGraphSettings,
)
from tests.harness.sglang.process import collect_process_output_after_timeout, tail, terminate_process_group
from xpool.config import LoopbackSite

REPO_ROOT = Path(__file__).resolve().parents[3]

type JsonValue = str | int | float | bool | list[int] | None

type GraphEvent = dict[str, JsonValue]

type GraphSettings = tuple[bool, bool]

PROBE_TIMEOUT_SECONDS = 30 * 60

PROCESS_TERMINATE_TIMEOUT_SECONDS = 30

PROCESS_KILL_TIMEOUT_SECONDS = 30

PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ProbeRun:
    graph_settings: SglangGraphSettings
    result: ProbeResult
    events: list[GraphEvent]
    base_gpu_id: int
    duration_s: float


def run_probe_worker(
    *,
    graph_settings_list: list[SglangGraphSettings],
    base_gpu_id: int,
    config_path: Path,
    tmp_path: Path,
    loopback_site: LoopbackSite,
) -> list[ProbeRun]:
    runs: list[ProbeRun] = []
    for graph_settings in graph_settings_list:
        event_outdir = tmp_path / f"{int(graph_settings.cuda_graph)}-{int(graph_settings.piecewise_cuda_graph)}"
        started_at = time.perf_counter()
        result = run_probe(
            graph_settings=graph_settings,
            base_gpu_id=base_gpu_id,
            config_path=config_path,
            event_outdir=event_outdir,
            loopback_site=loopback_site,
        )
        events = read_graph_events(event_outdir)
        runs.append(
            ProbeRun(
                graph_settings=graph_settings,
                result=result,
                events=events,
                base_gpu_id=base_gpu_id,
                duration_s=time.perf_counter() - started_at,
            )
        )
    return runs


def graph_settings_key(graph_settings: SglangGraphSettings) -> GraphSettings:
    return (graph_settings.cuda_graph, graph_settings.piecewise_cuda_graph)


def run_probe(
    *,
    graph_settings: SglangGraphSettings,
    base_gpu_id: int,
    config_path: Path,
    event_outdir: Path,
    loopback_site: LoopbackSite,
) -> ProbeResult:
    event_outdir.mkdir(parents=True, exist_ok=True)
    result_path = event_outdir / "result.json"
    env = probe_env(
        config_path=config_path,
        event_outdir=event_outdir,
        loopback_site=loopback_site,
    )
    command = [
        sys.executable,
        "-m",
        "tests.harness.sglang.offline_probe",
        "--config-path",
        str(config_path),
        "--base-gpu-id",
        str(base_gpu_id),
        "--result-path",
        str(result_path),
    ]
    if not graph_settings.cuda_graph:
        command.append("--disable-cuda-graph")
    if not graph_settings.piecewise_cuda_graph:
        command.append("--disable-piecewise-cuda-graph")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        cleanup_status = terminate_process_group(process)
        stdout, stderr, output_status = collect_process_output_after_timeout(process)
        raise AssertionError(
            "SGLang offline probe timed out for "
            f"cuda_graph={graph_settings.cuda_graph}, piecewise_cuda_graph={graph_settings.piecewise_cuda_graph}, "
            f"base_gpu_id={base_gpu_id}\n"
            f"cleanup: {cleanup_status}; output: {output_status}\n"
            f"stdout:\n{tail(stdout)}\nstderr:\n{tail(stderr)}"
        ) from None

    if process.returncode != 0:
        raise AssertionError(
            "SGLang offline probe failed for "
            f"cuda_graph={graph_settings.cuda_graph}, piecewise_cuda_graph={graph_settings.piecewise_cuda_graph} "
            f"base_gpu_id={base_gpu_id} with exit code {process.returncode}\n"
            f"stdout:\n{tail(stdout)}\nstderr:\n{tail(stderr)}"
        )
    return read_probe_result(
        result_path,
        graph_settings=graph_settings,
        base_gpu_id=base_gpu_id,
        stdout=stdout,
        stderr=stderr,
    )


def probe_env(*, config_path: Path, event_outdir: Path, loopback_site: LoopbackSite) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("XPOOL_DEBUG_LOOPBACK_ENABLE", None)
    env.pop("XPOOL_DEBUG_LOOPBACK_SITE", None)
    current_pythonpath = env.get("PYTHONPATH")
    pythonpath_entries = [str(REPO_ROOT / "tests"), str(REPO_ROOT)]
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SGLANG_PLUGINS": "xpool",
            "XPOOL_CONFIG": str(config_path),
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(event_outdir),
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        }
    )
    env["XPOOL_DEBUG_LOOPBACK_ENABLE"] = "1"
    env["XPOOL_DEBUG_LOOPBACK_SITE"] = loopback_site.value
    if loopback_site is LoopbackSite.ATNAGENT:
        env["XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE"] = "1"
        env["XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR"] = str(event_outdir)
    return env


def read_probe_result(
    result_path: Path,
    *,
    graph_settings: SglangGraphSettings,
    base_gpu_id: int,
    stdout: str,
    stderr: str,
) -> ProbeResult:
    try:
        result = cast(ProbeResult, json.loads(result_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(
            f"SGLang offline probe did not write a valid result JSON at {result_path}\n"
            f"stdout:\n{tail(stdout)}\nstderr:\n{tail(stderr)}"
        ) from exc
    if result["cuda_graph"] != graph_settings.cuda_graph:
        raise AssertionError(
            f"probe reported cuda_graph={result['cuda_graph']!r}, expected {graph_settings.cuda_graph!r}"
        )
    if result["piecewise_cuda_graph"] != graph_settings.piecewise_cuda_graph:
        raise AssertionError(
            "probe reported piecewise_cuda_graph="
            f"{result['piecewise_cuda_graph']!r}, expected {graph_settings.piecewise_cuda_graph!r}"
        )
    if result["base_gpu_id"] != base_gpu_id:
        raise AssertionError(f"probe reported base_gpu_id={result['base_gpu_id']!r}, expected {base_gpu_id!r}")
    return result
