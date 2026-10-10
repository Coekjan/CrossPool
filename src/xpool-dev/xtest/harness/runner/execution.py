"""Source-checkout execution composition for the installed test command."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from xkit.config import XpoolDevConfig, get_global_config
from xkit.device import DevicePool
from xkit.results import RunStore
from xkit.supervisor import TaskScopeFailure
from xpool.utils.sighandler import sighandle
from xtest.harness.report import TaskReportRecord, TestResultWriter, TestRunManifest
from xtest.harness.runner.collection import CollectionFailure
from xtest.harness.runner.console import configure_console
from xtest.harness.runner.ctest import CtestSuite, ctest_case_reports
from xtest.harness.runner.selection import collect_plan, select_suites
from xtest.harness.runner.suite import SuiteRunner
from xtest.harness.runner.task import compile_execution_tasks
from xtest.harness.sglang.serving.alignment import ServingGraphAdapter


def run_tests(selectors: tuple[str, ...], config: XpoolDevConfig) -> int:
    """Collect, plan, and execute the selected test suites."""

    configure_console()

    selected_suites = select_suites(Path.cwd(), config.xtest.suites)
    tool_config = config.record()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.monotonic_ns()}"
    try:
        result_store = RunStore(config.cache_root / "test-runs")
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
                tool_config=tool_config,
                selected_suites=selected_suites,
                selectors=selectors,
                strict_requirements=config.xtest.strict_requirements,
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
            strict_requirements=config.xtest.strict_requirements,
            run_directory=test_run.directory,
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
    result_writer: TestResultWriter | None = None,
) -> int:
    """Collect, schedule, and fully reap one durable test run."""

    print(f"xpool test run directory: {run_directory}")
    repository_root = Path.cwd().resolve()
    catalogue_path = get_global_config().xtest.catalog
    try:
        plan = collect_plan(
            repository_root,
            selected_suites,
            selectors,
            strict_requirements=strict_requirements,
            directory=run_directory / "collection",
            catalogue_path=catalogue_path,
        )
    except (CollectionFailure, ValueError) as error:
        print(f"xpool test collection failure: {error}", file=sys.stderr)
        if result_writer is not None:
            result_writer.fail(str(error))
        return 2

    if plan is not None:
        selected_suites = plan.selected_suites
        if result_writer is not None:
            result_writer.selection(selected_suites)

    needs_device_pool = "cext" in selected_suites or (
        plan is not None and any(case.requirements.device_count for case in plan.cases)
    )
    device_pool: DevicePool | None = None
    runner: SuiteRunner | None = None
    device_resources_releasable = True
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
        if needs_device_pool:
            device_pool = DevicePool.from_environment()
        if "cext" in selected_suites:
            assert device_pool is not None
            ctest_result = CtestSuite(Path.cwd()).run(device_pool=device_pool, run_directory=run_directory / "cext")
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
            device_pool=device_pool,
            result_writer=result_writer,
            catalogue_path=catalogue_path,
            tool_config_path=run_directory / "collection/tool-config.json",
        )
        with ExitStack() as stack:
            stack.enter_context(sighandle(signal.SIGINT, runner.request_stop))
            stack.enter_context(sighandle(signal.SIGTERM, runner.request_stop))
            return runner.run()
    except TaskScopeFailure as error:
        device_resources_releasable = False
        print(f"xpool test cannot release device resources after unproven task cleanup: {error}", file=sys.stderr)
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
        cleanup_verified = device_resources_releasable and runner_resources_releasable
        if device_pool is not None:
            cleanup_verified = cleanup_verified and not device_pool.active_leases
            if cleanup_verified:
                device_pool.close()
            else:
                print("xpool test could not prove device resources releasable", file=sys.stderr)
        if result_writer is not None:
            result_writer.cleanup(cleanup_verified)
