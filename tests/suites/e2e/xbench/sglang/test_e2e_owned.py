from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import tomli_w

from xbench.harness.serving.case import BenchCatalog, JsonlPrompts, NativeSampling, OwnedBenchCase, TraceArrivals
from xbench.harness.serving.measure import BenchRunManifest
from xbench.harness.serving.report import load_series
from xbench.harness.serving.workload import ScheduledRequest
from xkit.deployment import resolve_deployment_path
from xkit.gpu import query_physical_gpus, visible_gpu_uuids
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.support.config import e2e_base_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CASE = BenchCatalog.load(REPOSITORY_ROOT / "benches/benches.toml").select(("serving-001",))[0]
assert isinstance(CASE, OwnedBenchCase)


@pytest.mark.usefixtures(e2e_base_config.__name__)
@pytest.mark.requires_config
@pytest.mark.requires_cuda(min_devices=2)
@pytest.mark.requires_mps
@pytest.mark.requires_model_weights("Qwen/Qwen2.5-0.5B")
@pytest.mark.requires_model_weights("Qwen/Qwen3-0.6B")
@pytest.mark.estimated_duration(seconds=90)
@pytest.mark.timeout(4200)
def test_e2e_owned_benchmark_measures_both_targets_and_reproduces_offline(
    e2e_base_config: ResolvedConfig, tmp_path: Path, task_artifact_dir: Path | None
) -> None:
    # The outer deadline covers startup, sequential warmup and HTTP deadlines;
    # observed performance does not determine the verdict.
    assert isinstance(CASE, OwnedBenchCase)
    workdir = (task_artifact_dir if task_artifact_dir is not None else tmp_path) / "benchmark"
    workdir.mkdir()
    prompts = workdir / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt_id": "fixed", "input_ids": [1] * 16}) + "\n", encoding="utf-8")
    requests = tuple(
        ScheduledRequest(
            request_id=str(target.model_id),
            model_id=target.model_id,
            arrival_seconds=0.0,
            prompt_id="fixed",
            max_new_tokens=16,
        )
        for target in CASE.targets
    )
    trace = workdir / "trace.jsonl"
    trace.write_text("".join(request.model_dump_json() + "\n" for request in requests), encoding="utf-8")
    case = CASE.model_copy(
        update={
            "arrivals": TraceArrivals(kind="jsonl", path=trace, duration_seconds=0.1),
            "targets": tuple(
                target.model_copy(
                    update={
                        "prompts": JsonlPrompts(kind="jsonl", path=prompts),
                        "sampling": NativeSampling(ignore_eos=True),
                        "output_tokens": None,
                    }
                )
                for target in CASE.targets
            ),
        }
    )
    catalog = workdir / "benches" / "benches.toml"
    catalog.parent.mkdir()
    source = catalog.parent / "suites/serving/multi_model.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ROOT / "benches/suites/serving/multi_model.py", source)
    deployment = resolve_deployment_path(
        catalog, tuple(target.model_id for target in case.targets), case.deployment.stem
    )
    deployment.parent.mkdir(parents=True)
    shutil.copyfile(case.deployment, deployment)
    declaration = case.model_dump(mode="json", exclude_none=True, exclude={"id"})
    declaration["deployment"] = case.deployment.stem
    catalog.write_text(
        tomli_w.dumps({"serving_cases": {case.id: declaration}}),
        encoding="utf-8",
    )
    assigned = visible_gpu_uuids(query_physical_gpus())
    result_root = workdir / "runs"
    completed = subprocess.run(
        ["uv", "run", "--no-sync", "xbench", "run", "--catalog", str(catalog), "--result-root", str(result_root)],
        cwd=REPOSITORY_ROOT,
        env=dict(os.environ, SGLANG_PLUGINS="__none__", HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0"),
        capture_output=True,
        text=True,
        timeout=4000,
    )
    (workdir / "cli.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = Path(completed.stdout.strip())
    manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
    assert manifest.finished and manifest.result_code == 0
    repetition = run / "cases" / case.id / "repetition-0001"
    series = load_series(repetition, "owned")
    assert series.workload.requests == requests
    assert series.summary.cleanup_verified
    assert not (repetition / "report").exists()
    assert series.summary.execution_complete and series.summary.evidence_complete
    assert series.summary.outcomes["success"] == 2
    for target in case.targets:
        assert series.summary.targets[str(target.model_id)].output_tokens == 16
    inventory = series.environment["local_gpu_inventory"]
    assert isinstance(inventory, list) and len(inventory) == 2
    assert set(inventory) <= set(assigned)
    assert series.environment["serving_metadata_source"] == "observed"
    assert series.environment["environment_source"] == "effective_serving_launch"
    environment = series.environment["environment"]
    assert isinstance(environment, dict)
    assert environment["SGLANG_PLUGINS"] == "xpool"
    assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
    serving = series.environment["serving_metadata"]
    assert isinstance(serving, dict) and isinstance(serving["gpus"], list)
    assert {gpu["uuid"] for gpu in serving["gpus"] if isinstance(gpu, dict)} == set(inventory)
    assert isinstance(serving["target_gpu_uuids"], dict) and set(serving["target_gpu_uuids"]) == {
        str(target.model_id) for target in case.targets
    }
    assert serving["role_gpu_uuids"] == {"atn": [inventory[0]], "ffn": [inventory[1]]}
    assert isinstance(series.environment["capture_errors"], list)
    endpoints = series.environment["endpoints"]
    assert isinstance(endpoints, dict) and set(endpoints) == {str(target.model_id) for target in case.targets}

    output = repetition / "report"
    reported = subprocess.run(
        ["uv", "run", "--no-sync", "xbench", "report", str(run)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert reported.returncode == 0, reported.stderr
    projection = json.loads((output / "summary.json").read_bytes())
    assert projection["summary"] == series.summary.model_dump(mode="json")
    assert (output / "throughput.pdf").is_file()
