"""Test inventory and catalogue authoring."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from xkit.case import CaseFamily
from xkit.config import XpoolDevConfig
from xpool.utils.cli import RunnableCliCommand
from xtest.harness.runner.collection import CollectionFailure
from xtest.harness.runner.ctest import CtestSuite
from xtest.harness.runner.selection import SUITE_ORDER, collect_plan, select_suites
from xtest.harness.sglang.catalog import TestCatalog


class TestListCommand(RunnableCliCommand[XpoolDevConfig]):
    """List concrete test rows from isolated source collection."""

    name = "list"
    help = "list selected node IDs, resources and known case conditions"
    order = 10
    config_settings = ("config_path", "xtest_catalog", "xtest_suites")

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        root = Path.cwd()
        suites = select_suites(root, config.xtest.suites)
        with TemporaryDirectory(prefix="xpool-test-inventory-") as temporary:
            directory = Path(temporary) / "collection"
            try:
                plan = collect_plan(
                    root,
                    suites,
                    args.pytest_selectors,
                    strict_requirements=False,
                    directory=directory,
                    catalogue_path=config.xtest.catalog,
                )
            except CollectionFailure:
                log = directory / "collection.log"
                if log.is_file():
                    print(log.read_text(encoding="utf-8")[-16_384:], file=sys.stderr)
                raise
            if plan is not None:
                suites = plan.selected_suites
            if "cext" in suites:
                for name in CtestSuite(root).inventory():
                    print(f"cext\t{name}")
            if plan is not None:
                for stage in (*SUITE_ORDER, "models"):
                    for case in plan.cases:
                        if case.stage.value == stage:
                            resources = case.requirements
                            print(
                                f"{stage}\t{case.nodeid}\tdevices={resources.device_count} "
                                f"config={resources.requires_config} "
                                f"models={','.join(str(model_id) for model_id in resources.model_ids)}"
                            )
                            if case.inspection is not None:
                                print(f"  {case.inspection.model_dump_json()}")
        return 0


class TestCaseGenCommand(RunnableCliCommand[XpoolDevConfig]):
    """Append a serving or topology declaration with a permanent UUID."""

    name = "case-gen"
    help = "clone a case with a new UUID and print its edit location"
    order = 60
    config_settings = ("config_path", "xtest_catalog")

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--type", required=True, type=CaseFamily, choices=tuple(CaseFamily), dest="family")
        parser.add_argument("--from", required=True, dest="source")

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        location = TestCatalog.from_file(config.xtest.catalog).clone(args.source, family=args.family)
        print(f"{location.id}\nEdit {location.path}:{location.first_line}-{location.last_line}")
        print("The copied scenario is unchanged. Edit its conditions and description before running it.")
        return 0
