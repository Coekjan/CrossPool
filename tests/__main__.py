"""Canonical xpool Python test-suite composition root."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from tests.harness.runner.collection import CollectionFailure, CollectionWorker
from tests.harness.runner.ctest import CtestSuite
from tests.harness.runner.gpu import GpuLease, GpuPool
from tests.harness.runner.plan import TestPlan
from tests.harness.runner.results import TestResultStore
from tests.harness.runner.suite import SuiteRunner
from tests.harness.runner.supervisor import SupervisedTaskScope, TaskCompletionKind, TaskScopeFailure
from tests.harness.sglang.serving.alignment import ServingGraphAdapter
from xpool.mps import probe_mps_controller
from xpool.utils.sighandler import sighandle

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MPS_POOL_PROBE_TIMEOUT_SECONDS = 60.0
SUITE_ORDER = ("cext", "unit", "integration", "e2e")


def run_mps_pool_probe() -> int:
    """Initialize and synchronize every runner-visible GPU in one fresh client."""

    import torch

    visible = tuple(entry for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry)
    if not visible:
        raise RuntimeError("MPS pool probe requires nonempty CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != len(visible):
        raise RuntimeError(f"MPS pool probe expected {len(visible)} visible GPUs, received {torch.cuda.device_count()}")
    for device_index in range(len(visible)):
        with torch.cuda.device(device_index):
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
    return 0


def main(arguments: list[str] | None = None) -> int:
    """Collect, plan, and execute the selected Python suite."""

    parser = argparse.ArgumentParser(description="run the complete supervised xpool Python test suite")
    parser.add_argument("--strict-requirements", action="store_true")
    parser.add_argument("--suite", action="append", choices=SUITE_ORDER, dest="suites")
    parser.add_argument("--mps-pool-probe", action="store_true", help=argparse.SUPPRESS)
    options, selectors = parser.parse_known_args(arguments)
    if options.mps_pool_probe:
        if selectors or options.strict_requirements:
            parser.error("--mps-pool-probe is an internal standalone mode")
        return run_mps_pool_probe()

    selected_suites = tuple(options.suites or SUITE_ORDER)
    if len(selected_suites) != len(set(selected_suites)):
        parser.error("--suite cannot select the same suite more than once")
    if selectors and all(suite == "cext" for suite in selected_suites):
        parser.error("pytest selectors require a Python suite")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.monotonic_ns()}"
    try:
        result_store = TestResultStore.from_environment(REPOSITORY_ROOT / ".xpool-cache" / "test-runs")
        test_run = result_store.start(run_id)
    except (OSError, ValueError) as error:
        print(f"xpool test result setup failure: {error}", file=sys.stderr)
        return 2
    try:
        return execute_test_run(
            selected_suites,
            tuple(selectors),
            strict_requirements=options.strict_requirements,
            run_directory=test_run.directory,
        )
    finally:
        try:
            test_run.complete()
        except OSError as error:
            print(f"xpool test result completion failure: {error}", file=sys.stderr)
        result_store.cleanup()


def execute_test_run(
    selected_suites: tuple[str, ...],
    selectors: tuple[str, ...],
    *,
    strict_requirements: bool,
    run_directory: Path,
) -> int:
    """Collect, schedule, and fully reap one durable test run."""

    print(f"xpool test run directory: {run_directory}")
    plan: TestPlan | None = None
    try:
        python_suites = tuple(suite for suite in selected_suites if suite != "cext")
        if python_suites:
            collection_selectors = tuple(selectors) or tuple(f"tests/suites/{suite}" for suite in python_suites)
            plan = CollectionWorker(
                repository_root=REPOSITORY_ROOT,
                run_directory=run_directory / "collection",
                selectors=collection_selectors,
                strict_requirements=strict_requirements,
            ).collect()
            unexpected_stages = tuple(case.nodeid for case in plan.cases if case.stage.value not in selected_suites)
            if unexpected_stages:
                raise ValueError(f"collected tests outside selected suites: {unexpected_stages}")
    except (CollectionFailure, ValueError) as error:
        print(f"xpool test collection failure: {error}", file=sys.stderr)
        return 2

    needs_gpu_pool = "cext" in selected_suites or (
        plan is not None and any(case.requirements.cuda_count for case in plan.cases)
    )
    gpu_pool: GpuPool | None = None
    runner: SuiteRunner | None = None
    gpu_resources_releasable = True
    try:
        if needs_gpu_pool:
            gpu_pool = GpuPool.from_environment()
            prove_gpu_pool(gpu_pool, run_directory)
        if "cext" in selected_suites:
            assert gpu_pool is not None
            ctest_result = CtestSuite().run(gpu_pool=gpu_pool, run_directory=run_directory / "cext")
            print(
                f"STAGE cext: code={ctest_result.result_code}; "
                f"log={ctest_result.log_path} junit={ctest_result.junit_path}"
            )
            if ctest_result.result_code:
                return ctest_result.result_code
        if plan is None:
            return 0
        runner = SuiteRunner(
            plan,
            repository_root=REPOSITORY_ROOT,
            run_directory=run_directory,
            strict_requirements=strict_requirements,
            artifact_group_adapters=(ServingGraphAdapter(),),
            gpu_pool=gpu_pool,
        )
        with ExitStack() as stack:
            stack.enter_context(sighandle(signal.SIGINT, runner.request_stop))
            stack.enter_context(sighandle(signal.SIGTERM, runner.request_stop))
            return runner.run()
    except TaskScopeFailure as error:
        gpu_resources_releasable = False
        print(f"xpool test cannot release GPU resources after unproven task cleanup: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool test infrastructure failure: {error}", file=sys.stderr)
        return 2
    finally:
        if gpu_pool is not None:
            runner_resources_releasable = runner is None or runner.resources_releasable
            if gpu_resources_releasable and runner_resources_releasable and not gpu_pool.active_leases:
                gpu_pool.close()
            else:
                print("xpool test could not prove GPU resources releasable", file=sys.stderr)


def prove_gpu_pool(gpu_pool: GpuPool, run_directory: Path) -> None:
    """Prove MPS controller readiness and CUDA usability over the complete pool."""

    mps = probe_mps_controller()
    if not mps.online:
        raise RuntimeError(f"MPS is unhealthy during initial suite preflight: {mps.diagnostic}")
    probe_directory = run_directory / "mps-pool-probe"
    probe_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(GpuLease(gpu_pool.uuids).uuids)
    environment["PYTHONPYCACHEPREFIX"] = str(REPOSITORY_ROOT / ".xpool-cache" / "pycache")
    completion = SupervisedTaskScope.run(
        "mps-pool-probe",
        [sys.executable, "-m", "tests", "--mps-pool-probe"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        log_path=probe_directory / "probe.log",
        timeout_seconds=MPS_POOL_PROBE_TIMEOUT_SECONDS,
    )
    if completion.kind is not TaskCompletionKind.EXITED or completion.returncode != 0:
        raise RuntimeError(
            f"MPS pool usability probe failed ({completion.kind.value}, {completion.returncode}); "
            f"see {probe_directory / 'probe.log'}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
