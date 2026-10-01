"""Source-checkout execution composition for the installed test command."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from xkit.gpu import GpuLease, GpuPool
from xkit.results import RunStore
from xkit.supervisor import SupervisedTaskScope, TaskCompletionKind, TaskScopeFailure
from xpool.mps import probe_mps_controller
from xpool.utils.sighandler import sighandle
from xtest.harness.report import TaskReportRecord, TestResultWriter, TestRunManifest
from xtest.harness.runner.collection import CollectionFailure
from xtest.harness.runner.console import configure_console
from xtest.harness.runner.ctest import CtestSuite, ctest_case_reports
from xtest.harness.runner.selection import collect_plan, select_suites
from xtest.harness.runner.suite import SuiteRunner
from xtest.harness.runner.task import compile_execution_tasks
from xtest.harness.sglang.serving.alignment import ServingGraphAdapter

MPS_POOL_PROBE_TIMEOUT_SECONDS = 60.0


def run_mps_pool_probe() -> int:
    """Initialize and synchronize every runner-visible GPU in one fresh client."""

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


def run_tests(options: argparse.Namespace, selectors: tuple[str, ...], parser: argparse.ArgumentParser) -> int:
    """Collect, plan, and execute the selected test suites."""

    if options.mps_pool_probe:
        if selectors or options.strict_requirements:
            parser.error("--mps-pool-probe is an internal standalone mode")
        return run_mps_pool_probe()

    configure_console()

    try:
        selected_suites = select_suites(Path.cwd(), tuple(options.suites or ()), selectors, options.integration)
    except ValueError as error:
        parser.error(str(error))
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.monotonic_ns()}"
    try:
        result_store = RunStore(options.result_root.expanduser())
        test_run = result_store.start(run_id)
    except (OSError, ValueError) as error:
        print(f"xpool test result setup failure: {error}", file=sys.stderr)
        return 2
    result_code = 2
    try:
        packages = {}
        for name in ("xpool-dev", "xpool"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        writer = TestResultWriter(
            test_run.directory,
            TestRunManifest(
                run_id=run_id,
                selected_suites=selected_suites,
                selectors=selectors,
                integration=options.integration,
                strict_requirements=options.strict_requirements,
                tool_software={
                    "source": "local_distribution_metadata",
                    "python": platform.python_version(),
                    "packages": packages,
                },
            ),
        )
        result_code = execute_test_run(
            selected_suites,
            tuple(selectors),
            strict_requirements=options.strict_requirements,
            run_directory=test_run.directory,
            integration=options.integration,
            result_writer=writer,
        )
        writer.finish(result_code, cleanup_verified=writer.results.cleanup_verified)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool test result recording failure: {error}", file=sys.stderr)
        result_code = 2
    finally:
        try:
            test_run.complete()
        except OSError as error:
            print(f"xpool test result completion failure: {error}", file=sys.stderr)
            result_code = 2
    return result_code


def execute_test_run(
    selected_suites: tuple[str, ...],
    selectors: tuple[str, ...],
    *,
    strict_requirements: bool,
    run_directory: Path,
    integration: str | None = None,
    result_writer: TestResultWriter | None = None,
) -> int:
    """Collect, schedule, and fully reap one durable test run."""

    print(f"xpool test run directory: {run_directory}")
    repository_root = Path.cwd().resolve()
    catalogue_path = repository_root / "tests/tests.toml"
    try:
        plan = collect_plan(
            repository_root,
            selected_suites,
            selectors,
            integration=integration,
            strict_requirements=strict_requirements,
            directory=run_directory / "collection",
            catalogue_path=catalogue_path,
        )
    except (CollectionFailure, ValueError) as error:
        print(f"xpool test collection failure: {error}", file=sys.stderr)
        if result_writer is not None:
            result_writer.fail(str(error))
        return 2

    needs_gpu_pool = "cext" in selected_suites or (
        plan is not None and any(case.requirements.cuda_count for case in plan.cases)
    )
    gpu_pool: GpuPool | None = None
    runner: SuiteRunner | None = None
    gpu_resources_releasable = True
    try:
        if result_writer is not None:
            native_cases = CtestSuite(Path.cwd()).inventory() if "cext" in selected_suites else ()
            tasks = (
                {task.key: tuple(case.nodeid for case in task.cases) for task in compile_execution_tasks(plan)}
                if plan is not None
                else {}
            )
            if "cext" in selected_suites:
                tasks["cext"] = native_cases
            result_writer.inventory(
                python_cases=tuple(case.nodeid for case in plan.cases) if plan is not None else (),
                native_cases=native_cases,
                tasks=tasks,
            )
        if needs_gpu_pool:
            gpu_pool = GpuPool.from_environment()
            prove_gpu_pool(gpu_pool, run_directory)
        if "cext" in selected_suites:
            assert gpu_pool is not None
            ctest_result = CtestSuite(Path.cwd()).run(gpu_pool=gpu_pool, run_directory=run_directory / "cext")
            if result_writer is not None:
                result_writer.task(
                    TaskReportRecord(
                        key="cext",
                        stage="cext",
                        completion=ctest_result.completion,
                        result_code=ctest_result.result_code,
                        elapsed_seconds=ctest_result.elapsed_seconds,
                        cases=ctest_case_reports(ctest_result.junit_path) if ctest_result.junit_path.exists() else (),
                        artifact_directory="cext",
                    )
                )
                result_writer.stage("cext", ctest_result.result_code)
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
            repository_root=repository_root,
            run_directory=run_directory,
            strict_requirements=strict_requirements,
            artifact_group_adapters=(ServingGraphAdapter(),),
            gpu_pool=gpu_pool,
            result_writer=result_writer,
            catalogue_path=catalogue_path,
        )
        with ExitStack() as stack:
            stack.enter_context(sighandle(signal.SIGINT, runner.request_stop))
            stack.enter_context(sighandle(signal.SIGTERM, runner.request_stop))
            return runner.run()
    except TaskScopeFailure as error:
        gpu_resources_releasable = False
        print(f"xpool test cannot release GPU resources after unproven task cleanup: {error}", file=sys.stderr)
        if result_writer is not None:
            result_writer.fail(str(error))
        return 2
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool test infrastructure failure: {error}", file=sys.stderr)
        if result_writer is not None:
            result_writer.fail(str(error))
        return 2
    finally:
        runner_resources_releasable = runner is None or runner.resources_releasable
        cleanup_verified = gpu_resources_releasable and runner_resources_releasable
        if gpu_pool is not None:
            cleanup_verified = cleanup_verified and not gpu_pool.active_leases
            if cleanup_verified:
                gpu_pool.close()
            else:
                print("xpool test could not prove GPU resources releasable", file=sys.stderr)
        if result_writer is not None:
            result_writer.cleanup(cleanup_verified)


def prove_gpu_pool(gpu_pool: GpuPool, run_directory: Path) -> None:
    """Prove MPS controller readiness and CUDA usability over the complete pool."""

    mps = probe_mps_controller()
    if not mps.online:
        raise RuntimeError(f"MPS is unhealthy during initial suite preflight: {mps.diagnostic}")
    probe_directory = run_directory / "mps-pool-probe"
    probe_directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(GpuLease(gpu_pool.uuids).uuids)
    environment["PYTHONPYCACHEPREFIX"] = str(Path.cwd() / ".xpool-cache" / "pycache")
    completion = SupervisedTaskScope.run(
        "mps-pool-probe",
        [sys.executable, "-m", "xtest.cli", "run", "--mps-pool-probe"],
        cwd=Path.cwd(),
        env=environment,
        log_path=probe_directory / "probe.log",
        timeout_seconds=MPS_POOL_PROBE_TIMEOUT_SECONDS,
    )
    if completion.kind is not TaskCompletionKind.EXITED or completion.returncode != 0:
        raise RuntimeError(
            f"MPS pool usability probe failed ({completion.kind.value}, {completion.returncode}); "
            f"see {probe_directory / 'probe.log'}"
        )
