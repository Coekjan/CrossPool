from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import partial
from pathlib import Path

import pytest
import tomli_w

import xtest
from xbench.harness.serving.case import (
    BenchCatalog,
    JsonlPrompts,
    OwnedBenchCase,
    Sampling,
    TraceArrivals,
    owned_requirements,
)
from xbench.harness.serving.measure import BenchRunManifest
from xbench.harness.serving.report import load_series, retained_repetitions
from xbench.harness.serving.workload import ScheduledRequest
from xkit.case import CaseId
from xkit.deployment import resolve_deployment_path
from xkit.envs import DETERMINISTIC_MODE, DETERMINISTIC_PATH
from xkit.task import get_task_root
from xpool.utils.device import visible_uuids
from xpool.utils.sighandler import defer_signal_exceptions
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.support.config import e2e_base_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CASE = BenchCatalog.from_file(REPOSITORY_ROOT / "benches/benches.toml").select(
    ("58fcaacc-d3bc-4a7f-af68-a3eb1dbd79c1",)
)[0]
assert isinstance(CASE, OwnedBenchCase)


@pytest.mark.usefixtures(e2e_base_config.__name__)
@xtest.requirements(partial(owned_requirements, case=CASE))
@pytest.mark.estimated_duration(seconds=90)
@pytest.mark.timeout(4200)
def test_e2e_owned_benchmark_measures_both_targets_with_assigned_devices(
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
            "id": CaseId.generate(),
            "arrivals": TraceArrivals(kind="jsonl", path=trace, duration_seconds=0.1),
            "targets": tuple(
                target.model_copy(
                    update={
                        "prompts": JsonlPrompts(kind="jsonl", path=prompts),
                        "sampling": Sampling(ignore_eos=True),
                        "output_tokens": None,
                    }
                )
                for target in CASE.targets
            ),
        }
    )
    catalog = workdir / "benches" / "benches.toml"
    catalog.parent.mkdir()
    deployment = resolve_deployment_path(
        catalog, tuple(target.model_id for target in case.targets), case.deployment.stem
    )
    deployment.parent.mkdir(parents=True)
    shutil.copyfile(case.deployment, deployment)
    declaration = case.model_dump(mode="json", exclude_none=True, exclude={"id"})
    declaration["deployment"] = case.deployment.stem
    catalog.write_text(
        tomli_w.dumps({"serving_cases": {str(case.id): declaration}}),
        encoding="utf-8",
    )
    assigned = visible_uuids()
    result_root = workdir / "cache/bench-runs"
    root = get_task_root()
    scope = None
    process: subprocess.Popen[str] | None = None
    stdout_path = workdir / "cli.stdout.log"
    stderr_path = workdir / "cli.stderr.log"
    sampling_path = workdir / "ambient-sampling.jsonl"
    environment = dict(
        os.environ, SGLANG_PLUGINS="xpool,xkit_deterministic", HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0"
    )
    DETERMINISTIC_MODE.set("record", env=environment)
    DETERMINISTIC_PATH.set(str(sampling_path), env=environment)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            if root is not None:
                with defer_signal_exceptions():
                    scope = root.register_scope()
                root.activate()
            with defer_signal_exceptions():
                process = subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "--no-sync",
                        "--no-env-file",
                        "xbench",
                        "run",
                        "--all",
                        "--catalog",
                        str(catalog),
                        "--cache-root",
                        str(result_root.parent),
                    ],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    start_new_session=True,
                )
            returncode = process.wait()
        finally:
            if process is not None:
                if process.poll() is None:
                    # Only the CLI receives cancellation. Its existing runner
                    # closes owned serving scopes and reaps the whole task
                    # domain before returning its original verdict.
                    process.terminate()
                process.wait()
            if scope is not None:
                scope.complete()
    output_text = stdout_path.read_text(encoding="utf-8")
    errors = stderr_path.read_text(encoding="utf-8")
    assert returncode == 0, output_text + errors
    assert not sampling_path.exists()
    run = Path(output_text.strip())
    manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
    assert manifest.finished and manifest.result_code == 0
    attempts, _, complete = retained_repetitions(run, manifest)
    assert complete and len(attempts) == 1
    series = load_series(attempts[0], "owned")
    assert series.workload.requests == requests
    assert series.summary.cleanup_verified
    assert series.summary.execution_complete and series.summary.evidence_complete
    assert series.summary.outcomes["success"] == 2
    for target in case.targets:
        assert series.summary.targets[str(target.model_id)].output_tokens == 16
    serving = series.environment["serving"]
    assert isinstance(serving, dict) and isinstance(serving["devices"], list)
    inventory = [device["uuid"] for device in serving["devices"] if isinstance(device, dict)]
    assert len(inventory) == 2 and set(inventory) <= set(assigned)
    assert serving["role_device_uuids"] == {"atn": [inventory[0]], "ffn": [inventory[1]]}
