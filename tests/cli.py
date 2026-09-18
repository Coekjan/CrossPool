"""Canonical CrossPool test command implementation."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from tests.harness.runner.collection import CollectionFailure, CollectionWorker
from tests.harness.runner.console import configure_console
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
INTEGRATIONS = ("sglang",)
DEFAULT_KEEP_RUNS = 20


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
    """Run the source-checkout test command."""

    parser = argparse.ArgumentParser(prog="xtest", description="CrossPool test command")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the supervised CrossPool test suite")
    run_parser.add_argument("--strict-requirements", action="store_true")
    run_parser.add_argument("--suite", action="append", dest="suites")
    run_parser.add_argument("--integration", choices=INTEGRATIONS)
    run_parser.add_argument("--mps-pool-probe", action="store_true", help=argparse.SUPPRESS)

    clean_parser = subparsers.add_parser("clean", help="clean durable test results")
    clean_selection = clean_parser.add_mutually_exclusive_group()
    clean_selection.add_argument("--all", action="store_true", dest="remove_all")
    clean_selection.add_argument("--keep", type=positive_integer, metavar="N")
    clean_parser.add_argument("--dry-run", action="store_true")

    options, remaining = parser.parse_known_args(arguments)
    if options.command == "clean":
        if remaining:
            clean_parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        return clean_test_results(
            keep_runs=0 if options.remove_all else options.keep or DEFAULT_KEEP_RUNS,
            dry_run=options.dry_run,
        )
    return run_tests(options, tuple(remaining), run_parser)


def positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def clean_test_results(*, keep_runs: int, dry_run: bool) -> int:
    """Clean inactive durable results using one explicit retention policy."""

    try:
        cleanup = TestResultStore(REPOSITORY_ROOT / ".xpool-cache" / "test-runs").cleanup(
            keep_runs=keep_runs,
            dry_run=dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"xpool test result cleanup failure: {error}", file=sys.stderr)
        return 1
    for path in cleanup.removable:
        print(f"remove {path}")
    for path in cleanup.retained:
        print(f"keep {path}")
    for path in cleanup.active:
        print(f"active {path}")
    return 0


def run_tests(options: argparse.Namespace, selectors: tuple[str, ...], parser: argparse.ArgumentParser) -> int:
    """Collect, plan, and execute the selected test suites."""

    if options.mps_pool_probe:
        if selectors or options.strict_requirements:
            parser.error("--mps-pool-probe is an internal standalone mode")
        return run_mps_pool_probe()

    configure_console()

    selected_suites = tuple(options.suites or SUITE_ORDER)
    if len(selected_suites) != len(set(selected_suites)):
        parser.error("--suite cannot select the same suite more than once")
    model_suites = tuple(suite for suite in selected_suites if suite not in (*SUITE_ORDER, "models"))
    if model_suites:
        models_root = REPOSITORY_ROOT / "tests/suites/models"
        unknown = tuple(
            model_id
            for model_id in model_suites
            if len(parts := model_id.split("/")) != 2
            or any(part in ("", ".", "..") for part in parts)
            or not models_root.joinpath(*parts).is_dir()
        )
        if unknown:
            parser.error(f"unknown model suite: {', '.join(unknown)}")
    if model_suites and "models" in selected_suites:
        parser.error("--suite models cannot be combined with individual model suites")
    if selectors and all(suite == "cext" for suite in selected_suites):
        parser.error("pytest selectors require a Python suite")
    if options.integration is not None and selectors:
        parser.error("--integration cannot be combined with explicit pytest selectors")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.monotonic_ns()}"
    try:
        result_store = TestResultStore(REPOSITORY_ROOT / ".xpool-cache" / "test-runs")
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
            integration=options.integration,
        )
    finally:
        try:
            test_run.complete()
        except OSError as error:
            print(f"xpool test result completion failure: {error}", file=sys.stderr)


def execute_test_run(
    selected_suites: tuple[str, ...],
    selectors: tuple[str, ...],
    *,
    strict_requirements: bool,
    run_directory: Path,
    integration: str | None = None,
) -> int:
    """Collect, schedule, and fully reap one durable test run."""

    print(f"xpool test run directory: {run_directory}")
    model_suites = tuple(suite for suite in selected_suites if suite not in (*SUITE_ORDER, "models"))
    plan: TestPlan | None = None
    try:
        python_suites = tuple(suite for suite in selected_suites if suite != "cext")
        if python_suites:
            collection_selectors = tuple(selectors) or tuple(
                path
                for suite in python_suites
                for path in suite_selectors(suite, model_suites=model_suites, integration=integration)
            )
            plan = CollectionWorker(
                repository_root=REPOSITORY_ROOT,
                run_directory=run_directory / "collection",
                selectors=collection_selectors,
                strict_requirements=strict_requirements,
            ).collect()
            unexpected_stages = tuple(
                case.nodeid
                for case in plan.cases
                if case.stage.value not in selected_suites
                and not (
                    case.stage.value == "models"
                    and any(case.path.startswith(f"tests/suites/models/{model_id}/") for model_id in model_suites)
                )
            )
            if unexpected_stages:
                raise ValueError(f"collected tests outside selected suites: {unexpected_stages}")
            if integration is not None:
                unexpected_integrations = tuple(
                    case.nodeid
                    for case in plan.cases
                    if integration_for_file(Path(case.path), case.stage.value) not in (None, integration)
                )
                if unexpected_integrations:
                    raise ValueError(f"collected tests outside selected integration: {unexpected_integrations}")
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


def suite_selectors(suite: str, *, model_suites: tuple[str, ...], integration: str | None) -> tuple[str, ...]:
    """Select pytest files before importing an unselected serving engine."""

    root = Path(f"tests/suites/models/{suite}" if suite in model_suites else f"tests/suites/{suite}")
    if integration is None or suite == "unit":
        return (root.as_posix(),)
    selected = tuple(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in sorted((REPOSITORY_ROOT / root).rglob("test_*.py"))
        if integration_for_file(path, "models" if suite in model_suites else suite) in (None, integration)
    )
    if not selected:
        raise ValueError(f"no tests for integration {integration!r} in suite {suite!r}")
    return selected


def integration_for_file(path: Path, stage: str) -> str | None:
    """Identify engine-owned files by their directory or model-test filename."""

    for name in INTEGRATIONS:
        if path.name.startswith(f"test_{name}_") or (stage != "models" and name in path.parts):
            return name
    return None


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
        [sys.executable, "-m", "tests.cli", "run", "--mps-pool-probe"],
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
