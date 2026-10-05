"""Installed CrossPool test inventory, execution, reports and retention commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from xkit.cli import positive_integer, print_cleanup
from xkit.results import RunStore
from xtest.harness.report import report_test_runs
from xtest.harness.runner import execution
from xtest.harness.runner.collection import CollectionFailure
from xtest.harness.runner.ctest import CtestSuite
from xtest.harness.runner.selection import INTEGRATIONS, SUITE_ORDER, collect_plan, select_suites

__all__ = ["main"]


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one test-tool command; reports and cleanup need only retained evidence.

    Source inventory/execution require the checkout and its development tools.
    Configuration, collection and report errors return two. Original test-run
    verdicts are retained independently of successful report generation.
    """

    parser = argparse.ArgumentParser(prog="xtest", description="CrossPool test command")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("list", "list concrete selected tests"), ("run", "run supervised tests")):
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--suite", action="append", dest="suites")
        command_parser.add_argument("--integration", choices=INTEGRATIONS)
        if name == "run":
            command_parser.add_argument("--result-root", type=Path, default=Path(".xpool-cache/test-runs"))
            command_parser.add_argument("--strict-requirements", action="store_true")
    report_parser = subparsers.add_parser("report", help="report retained test results offline")
    report_parser.add_argument("inputs", nargs="+", type=Path)
    report_parser.add_argument("--output", required=True, type=Path)
    report_parser.add_argument("--label", action="append", dest="labels")
    clean_parser = subparsers.add_parser("clean", help="clean inactive durable test results")
    clean_parser.add_argument("--result-root", type=Path, default=Path(".xpool-cache/test-runs"))
    clean_selection = clean_parser.add_mutually_exclusive_group()
    clean_selection.add_argument("--all", action="store_true", dest="remove_all")
    clean_selection.add_argument("--keep", type=positive_integer, default=20, metavar="N")
    clean_parser.add_argument("--dry-run", action="store_true")

    options, selectors = parser.parse_known_args(None if arguments is None else list(arguments))
    if options.command in {"report", "clean"} and selectors:
        parser.error(f"unrecognized arguments: {' '.join(selectors)}")
    try:
        if options.command == "report":
            report_test_runs(options.inputs, output=options.output, labels=options.labels)
            print(options.output.resolve())
            return 0
        if options.command == "clean":
            cleanup = RunStore(options.result_root.expanduser()).cleanup(
                keep_runs=0 if options.remove_all else options.keep, dry_run=options.dry_run
            )
            print_cleanup(cleanup)
            return 0
        repository_root = Path.cwd()
        if not (repository_root / "pyproject.toml").is_file() or not (repository_root / "tests/tests.toml").is_file():
            print("xtest must run from the xpool repository root", file=sys.stderr)
            return 2
        match options.command:
            case "list":
                try:
                    suites = select_suites(
                        repository_root, tuple(options.suites or ()), tuple(selectors), options.integration
                    )
                except ValueError as error:
                    parser.error(str(error))
                return list_tests(repository_root, suites, tuple(selectors), integration=options.integration)
            case "run":
                return execution.run_tests(options, tuple(selectors), parser)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool test {options.command} failure: {error}", file=sys.stderr)
        return 2
    return 2


def list_tests(
    repository_root: Path,
    suites: tuple[str, ...],
    selectors: tuple[str, ...],
    *,
    integration: str | None,
) -> int:
    with TemporaryDirectory(prefix="xpool-test-inventory-") as temporary:
        directory = Path(temporary) / "collection"
        try:
            plan = collect_plan(
                repository_root,
                suites,
                selectors,
                integration=integration,
                strict_requirements=False,
                directory=directory,
                catalogue_path=(repository_root / "tests/tests.toml").resolve(),
            )
        except CollectionFailure:
            # Inventory scratch disappears on return; expose diagnostics now.
            log = directory / "collection.log"
            if log.is_file():
                print(log.read_text(encoding="utf-8")[-16_384:], file=sys.stderr)
            raise
        if "cext" in suites:
            for name in CtestSuite(repository_root).inventory():
                print(f"cext\t{name}")
        if plan is not None:
            for stage in (*SUITE_ORDER, "models"):
                for case in plan.cases:
                    if case.stage.value == stage:
                        requirement = case.requirements
                        print(
                            f"{stage}\t{case.nodeid}\tdevices={requirement.device_count} "
                            f"config={requirement.requires_config} "
                            f"models={','.join(str(model_id) for model_id in requirement.model_ids)}"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
