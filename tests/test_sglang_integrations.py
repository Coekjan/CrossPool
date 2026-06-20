from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from helpers.sglang_offline_probe import GRAPH_SETTINGS, MODEL_ID, ProbeResult, SglangGraphSettings

from xpool.cext import NativeLoadError, ensure_xpool_ops_loaded
from xpool.config import MissingRequiredConfig, XpoolConfig
from xpool.integrations.sglang.topology import derive_parallel_policy, load_model_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
type JsonValue = str | int | float | bool | list[int] | None
type GraphEvent = dict[str, JsonValue]
type GraphSettings = tuple[bool, bool]

# SGLang model load and long decode can take several minutes; only cleanup paths
# use shorter bounded waits after the main probe has already timed out.
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


def test_sglang_offline_inference(tmp_path: Path) -> None:
    try:
        ensure_xpool_ops_loaded()
    except NativeLoadError as exc:
        pytest.skip(f"xpool native ops are not installed: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SGLang offline graph integration coverage")
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        pytest.skip("CUDA is required for SGLang offline graph integration coverage")

    config_path = resolve_config_path()
    config = XpoolConfig.from_file(config_path)
    try:
        model_path = config.model_path_of(MODEL_ID).resolve()
    except MissingRequiredConfig as exc:
        pytest.skip(str(exc))
    if not (model_path / "config.json").is_file():
        pytest.skip(f"{MODEL_ID} weights are unavailable at {model_path}")

    devices_per_probe = devices_per_sglang_probe(config=config, model_path=model_path)
    worker_count = probe_worker_count(
        case_count=len(GRAPH_SETTINGS),
        device_count=device_count,
        devices_per_probe=devices_per_probe,
    )
    assignments = assign_graph_settings(GRAPH_SETTINGS, worker_count=worker_count)
    runs: list[ProbeRun] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_probe_worker,
                graph_settings_list=graph_settings_list,
                base_gpu_id=worker_index * devices_per_probe,
                config_path=config_path,
                tmp_path=tmp_path,
            )
            for worker_index, graph_settings_list in enumerate(assignments)
        ]
        for future in as_completed(futures):
            runs.extend(future.result())

    results: dict[GraphSettings, ProbeResult] = {}
    runs_by_settings: dict[GraphSettings, ProbeRun] = {}
    for run in runs:
        graph_settings = run.graph_settings
        key = graph_settings_key(graph_settings)
        results[key] = run.result
        runs_by_settings[key] = run
        assert_graph_events(graph_settings, run.events)

    durations = [
        {
            "base_gpu_id": run.base_gpu_id,
            "cuda_graph": run.graph_settings.cuda_graph,
            "duration_s": run.duration_s,
            "piecewise_cuda_graph": run.graph_settings.piecewise_cuda_graph,
        }
        for run in (runs_by_settings[graph_settings_key(graph_settings)] for graph_settings in GRAPH_SETTINGS)
    ]
    print("XPOOL_SGLANG_INTEGRATION_DURATIONS=" + json.dumps(durations, sort_keys=True))
    eager_output_ids = results[(False, False)]["output_ids"]
    assert results[(False, True)]["output_ids"] == eager_output_ids
    assert results[(True, False)]["output_ids"] == eager_output_ids
    assert results[(True, True)]["output_ids"] == eager_output_ids


def devices_per_sglang_probe(*, config: XpoolConfig, model_path: Path) -> int:
    spec = load_model_spec(model_path, model_id=MODEL_ID)
    policy = derive_parallel_policy(
        spec,
        attention_device_count=len(config.devices.attention_cuda_devices),
        ffn_tp_size=len(config.devices.ffn_cuda_devices),
    )
    return policy.sglang_tp_size * policy.sglang_dp_size


def probe_worker_count(*, case_count: int, device_count: int, devices_per_probe: int) -> int:
    if devices_per_probe <= 0:
        raise AssertionError(f"devices_per_probe must be positive, got {devices_per_probe}")
    available_workers = device_count // devices_per_probe
    if available_workers <= 0:
        pytest.skip(
            f"SGLang offline graph integration requires {devices_per_probe} CUDA device(s) per probe, "
            f"but only {device_count} CUDA device(s) are available"
        )
    return min(case_count, available_workers)


def assign_graph_settings(
    graph_settings: tuple[SglangGraphSettings, ...],
    *,
    worker_count: int,
) -> list[list[SglangGraphSettings]]:
    assignments: list[list[SglangGraphSettings]] = [[] for _ in range(worker_count)]
    for index, settings in enumerate(graph_settings):
        assignments[index % worker_count].append(settings)
    return [assignment for assignment in assignments if assignment]


def run_probe_worker(
    *,
    graph_settings_list: list[SglangGraphSettings],
    base_gpu_id: int,
    config_path: Path,
    tmp_path: Path,
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


def resolve_config_path() -> Path:
    candidates: list[Path] = []
    if config_path := os.environ.get("XPOOL_CONFIG"):
        candidates.append(Path(config_path))
    candidates.append(REPO_ROOT / "configs" / "dev.local.toml")
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        if path.is_file():
            return path.resolve()
    pytest.skip("set XPOOL_CONFIG or create configs/dev.local.toml to run SGLang offline integration coverage")


def run_probe(
    *,
    graph_settings: SglangGraphSettings,
    base_gpu_id: int,
    config_path: Path,
    event_outdir: Path,
) -> ProbeResult:
    event_outdir.mkdir(parents=True)
    result_path = event_outdir / "result.json"
    env = probe_env(
        config_path=config_path,
        event_outdir=event_outdir,
    )
    command = [
        sys.executable,
        "-m",
        "helpers.sglang_offline_probe",
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


def probe_env(*, config_path: Path, event_outdir: Path) -> dict[str, str]:
    env = dict(os.environ)
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
            "XPOOL_DEBUG_SHIM_LOOPBACK_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(event_outdir),
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        }
    )
    return env


def test_terminate_process_group_reports_unreaped_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        pid = 12345

        def wait(self, timeout: float | None = None) -> None:
            assert timeout is not None
            raise subprocess.TimeoutExpired(cmd="probe", timeout=timeout)

    process = FakeProcess()
    signals: list[int] = []

    def fake_killpg(pid: int, signal_number: int) -> None:
        assert pid == process.pid
        signals.append(signal_number)

    monkeypatch.setattr(os, "killpg", fake_killpg)

    status = terminate_process_group(cast(subprocess.Popen[str], process))

    assert status == "process group still running after SIGKILL"
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_collect_process_output_after_timeout_closes_stuck_pipes() -> None:
    class FakePipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakePipe()
            self.stderr = FakePipe()

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            assert timeout is not None
            raise subprocess.TimeoutExpired(
                cmd="probe",
                timeout=timeout,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

    process = FakeProcess()

    stdout, stderr, status = collect_process_output_after_timeout(cast(subprocess.Popen[str], process))

    assert stdout == "partial stdout"
    assert stderr == "partial stderr"
    assert status == "stdout/stderr pipes did not close after cleanup"
    assert process.stdout.closed
    assert process.stderr.closed


def terminate_process_group(process: subprocess.Popen[str]) -> str:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "process group already exited before SIGTERM"
    try:
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
        return "process group exited after SIGTERM"
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "process group exited before SIGKILL"
    try:
        process.wait(timeout=PROCESS_KILL_TIMEOUT_SECONDS)
        return "process group exited after SIGKILL"
    except subprocess.TimeoutExpired:
        return "process group still running after SIGKILL"


def collect_process_output_after_timeout(process: subprocess.Popen[str]) -> tuple[str, str, str]:
    try:
        stdout, stderr = process.communicate(timeout=PROCESS_OUTPUT_DRAIN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        close_process_pipes(process)
        return (
            timeout_output_text(exc.output),
            timeout_output_text(exc.stderr),
            "stdout/stderr pipes did not close after cleanup",
        )
    return stdout, stderr, "stdout/stderr drained after cleanup"


def close_process_pipes(process: subprocess.Popen[str]) -> None:
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            with suppress(OSError, ValueError):
                pipe.close()


def timeout_output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


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


def read_graph_events(outdir: Path) -> list[GraphEvent]:
    events: list[GraphEvent] = []
    for path in sorted(outdir.glob("xpool.graph-observer.*.jsonl")):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    events.append(cast(GraphEvent, json.loads(line)))
    return events


def assert_graph_events(graph_settings: SglangGraphSettings, events: list[GraphEvent]) -> None:
    full_graph_phases = graph_phases(events, kind="full_cuda_graph")
    pcg_phases = graph_phases(events, kind="piecewise_cuda_graph")
    full_graph_phases_expected = {"capture_begin", "capture_end", "replay_begin", "replay_end"}
    piecewise_graph_phases_expected = {"capture_begin", "capture_end"}
    if graph_settings.cuda_graph:
        assert full_graph_phases_expected <= full_graph_phases
    else:
        assert full_graph_phases == set()
    if graph_settings.piecewise_cuda_graph:
        assert piecewise_graph_phases_expected <= pcg_phases
    else:
        assert pcg_phases == set()


def graph_phases(events: list[GraphEvent], *, kind: str) -> set[str]:
    phases: set[str] = set()
    for event in events:
        if event.get("kind") == kind and isinstance(event.get("phase"), str):
            phases.add(cast(str, event["phase"]))
    return phases


def tail(text: str, *, limit: int = 12000) -> str:
    return text[-limit:]
