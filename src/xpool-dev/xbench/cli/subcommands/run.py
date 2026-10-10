"""Execute explicitly selected benchmark cases."""

import argparse
from pathlib import Path

from xbench.harness.serving.case import BenchCatalog
from xbench.harness.serving.runner import run_benchmarks
from xkit.config import XpoolDevConfig
from xpool.config import XpoolConfig
from xpool.utils.cli import RunnableCliCommand


class BenchRunCommand(RunnableCliCommand[XpoolDevConfig]):
    """Require all cases or a unique-prefix selection before execution."""

    name = "run"
    help = "measure all cases or selected case prefixes"
    order = 20
    config_settings = ("config_path", "xbench_catalog", "xbench_repetitions")

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        selection = parser.add_mutually_exclusive_group(required=True)
        selection.add_argument("--all", action="store_true")
        selection.add_argument("--case", action="append", dest="cases")
        parser.add_argument(
            "--continue",
            type=Path,
            dest="continue_dir",
            help="continue the original prepared cases in an existing artifact directory",
        )
        parser.add_argument(
            "--rerun-failure",
            action="store_true",
            help="with --continue, execute unsuccessful attempted repetitions once",
        )
        parser.add_argument(
            "--fast-fail",
            action="store_true",
            help="stop admitting new repetitions after a new failure; let active tasks finish",
        )
        XpoolConfig.add_cli_args(parser, names=("cache_root",))

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        catalog = BenchCatalog.from_file(config.xbench.catalog)
        return run_benchmarks(
            catalog.select(tuple(args.cases or ())),
            root=config.cache_root / "bench-runs",
            catalogue_path=catalog.path,
            fast_fail=args.fast_fail,
        )
