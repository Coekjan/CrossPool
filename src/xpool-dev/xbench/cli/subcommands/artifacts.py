"""Exact artifact discovery, offline reports and explicit retention."""

import argparse
from pathlib import Path, PurePosixPath

from xbench.harness.serving.measure import REPETITION_DIRECTORY_PATTERN
from xbench.harness.serving.report import list_bench_artifacts, report_bench_runs
from xkit.case import CaseId
from xkit.config import XpoolDevConfig
from xkit.results import RUN_ID_PATTERN, RunStore
from xpool.config import XpoolConfig
from xpool.utils.cli import RunnableCliCommand


class BenchReportCommand(RunnableCliCommand[XpoolDevConfig]):
    """Discover eligible evidence or generate reports by exact artifact address."""

    name = "report"
    help = "list reportable artifact IDs or generate independent offline reports"
    order = 30
    config_settings = (
        "config_path",
        "xbench_report_layout",
        "xbench_report_formats",
        "xbench_report_ppi",
        "xbench_report_legend_visible",
    )

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("inputs", nargs="*", metavar="ARTIFACT_ID")
        parser.add_argument("--list", action="store_true", dest="list_artifacts")
        parser.add_argument(
            "--output", type=Path, help="optional export root; default reports stay with their evidence"
        )
        parser.add_argument("--width", type=float, default=argparse.SUPPRESS, help="canvas width in inches")
        parser.add_argument("--height", type=float, default=argparse.SUPPRESS, help="canvas height in inches")
        XpoolConfig.add_cli_args(parser, names=("cache_root",))

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        root = config.cache_root / "bench-runs"
        if args.list_artifacts:
            if args.inputs or args.output is not None:
                raise ValueError("report --list cannot be combined with artifact inputs or --output")
            for identity in list_bench_artifacts(root):
                print(identity)
            return 0
        if not args.inputs:
            raise ValueError("supply artifact IDs, or use report --list")
        paths = []
        for identity in dict.fromkeys(args.inputs):
            address = PurePosixPath(identity)
            parts = address.parts
            if (
                address.is_absolute()
                or address.as_posix() != identity
                or len(parts) not in {1, 4}
                or RUN_ID_PATTERN.fullmatch(parts[0]) is None
            ):
                raise ValueError(f"invalid benchmark artifact ID: {identity!r}")
            if len(parts) == 4:
                CaseId(parts[2])
                if parts[1] != "cases" or REPETITION_DIRECTORY_PATTERN.fullmatch(parts[3]) is None:
                    raise ValueError(f"invalid benchmark repetition address: {identity!r}")
            paths.append(root / identity)
        for destination in report_bench_runs(paths, output=args.output):
            print(destination)
        return 0


class BenchCleanCommand(RunnableCliCommand[XpoolDevConfig]):
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
        cleanup = RunStore(config.cache_root / "bench-runs").cleanup(
            keep_runs=0 if args.remove_all else config.keep_runs,
            dry_run=args.dry_run,
        )
        cleanup.print()
        return 0
