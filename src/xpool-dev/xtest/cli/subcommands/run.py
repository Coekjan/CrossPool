"""Execute selected test suites and pytest inputs."""

import argparse

from xkit.config import XpoolDevConfig
from xpool.config import XpoolConfig
from xpool.utils.cli import RunnableCliCommand
from xtest.harness.runner.execution import run_tests


class TestRunCommand(RunnableCliCommand[XpoolDevConfig]):
    """Run the effective scope established by isolated pytest collection."""

    name = "run"
    help = "run selected suites and pytest filters"
    order = 20
    config_settings = ("config_path", "xtest_catalog", "xtest_suites", "xtest_strict_requirements")

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        XpoolConfig.add_cli_args(parser, names=("cache_root",))

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        return run_tests(args.pytest_selectors, config)
