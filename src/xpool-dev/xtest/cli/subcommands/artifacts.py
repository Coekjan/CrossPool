"""Exact artifact discovery, offline reports and explicit retention."""

import argparse
from pathlib import Path

from xkit.config import XpoolDevConfig
from xkit.results import RUN_ID_PATTERN, RunStore
from xpool.config import XpoolConfig
from xpool.utils.cli import RunnableCliCommand
from xtest.harness.report import list_test_artifacts, report_test_runs


class TestReportCommand(RunnableCliCommand[XpoolDevConfig]):
    """Discover eligible evidence or generate reports by exact artifact address."""

    name = "report"
    help = "list reportable artifact IDs or generate independent offline reports"
    order = 30
    config_settings = ("config_path",)

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("inputs", nargs="*", metavar="ARTIFACT_ID")
        parser.add_argument("--list", action="store_true", dest="list_artifacts")
        parser.add_argument(
            "--output", type=Path, help="optional export root; default reports stay with their evidence"
        )
        XpoolConfig.add_cli_args(parser, names=("cache_root",))

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        root = config.cache_root / "test-runs"
        if args.list_artifacts:
            if args.inputs or args.output is not None:
                raise ValueError("report --list cannot be combined with artifact inputs or --output")
            for identity in list_test_artifacts(root):
                print(identity)
            return 0
        if not args.inputs:
            raise ValueError("supply artifact IDs, or use report --list")
        paths = []
        for identity in dict.fromkeys(args.inputs):
            if RUN_ID_PATTERN.fullmatch(identity) is None:
                raise ValueError(f"invalid test artifact ID: {identity!r}")
            paths.append(root / identity)
        for destination in report_test_runs(paths, output=args.output):
            print(destination)
        return 0


class TestCleanCommand(RunnableCliCommand[XpoolDevConfig]):
    """Remove only inactive run entries under the resolved cache owner."""

    name = "clean"
    help = "clean inactive runs using configured retention"
    order = 40
    config_settings = ("config_path", "keep_runs")

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--all", action="store_true", dest="remove_all")
        parser.add_argument("--dry-run", action="store_true")
        XpoolConfig.add_cli_args(parser, names=("cache_root",))

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        if args.remove_all and "keep_runs" in vars(args):
            raise ValueError("clean --all and --keep are mutually exclusive")
        cleanup = RunStore(config.cache_root / "test-runs").cleanup(
            keep_runs=0 if args.remove_all else config.keep_runs,
            dry_run=args.dry_run,
        )
        cleanup.print()
        return 0
