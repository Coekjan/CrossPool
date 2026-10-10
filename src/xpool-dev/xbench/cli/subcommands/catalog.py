"""Benchmark inventory and catalogue authoring."""

import argparse

from xbench.harness.collection import collect_programs
from xbench.harness.serving.case import BenchCatalog, OwnedBenchCase
from xkit.case import CaseFamily
from xkit.config import DeploymentConfig, XpoolDevConfig
from xpool.utils.cli import RunnableCliCommand


class BenchListCommand(RunnableCliCommand[XpoolDevConfig]):
    """Explain cases and their portable experiment conditions."""

    name = "list"
    help = "list case IDs, descriptions, models, topology and workloads"
    order = 10
    config_settings = ("config_path", "xbench_catalog")

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        catalog = BenchCatalog.from_file(config.xbench.catalog)
        programs = collect_programs(catalog.path, catalog.cases)
        for case, program in zip(catalog.cases, programs, strict=True):
            print(f"{case.id}\t{case.description}")
            print(f"  mode={case.mode} devices={program.requirements.device_count} seed={case.seed}")
            deployment = (
                DeploymentConfig.from_file(case.deployment, model_ids=tuple(target.model_id for target in case.targets))
                if isinstance(case, OwnedBenchCase)
                else None
            )
            if deployment is not None:
                print(
                    f"  attention={deployment.atn.devices} ffn={deployment.ffn.devices} "
                    f"lanes={deployment.ffn_concurrency} slo={deployment.slo.model_dump_json()}"
                )
            print(f"  arrivals={case.arrivals.model_dump_json()}")
            for target in case.targets:
                print(
                    f"  model={target.model_id} prompts={target.prompts.model_dump_json()} "
                    f"output_tokens={target.output_tokens}"
                )
                if deployment is not None:
                    print(f"    geometry={deployment.model_by_id[target.model_id].model_dump_json(exclude={'path'})}")
        return 0


class BenchCaseGenCommand(RunnableCliCommand[XpoolDevConfig]):
    """Clone one validated declaration, preserving the original catalogue text."""

    name = "case-gen"
    help = "clone a case with a new UUID and print its edit location"
    order = 60
    config_settings = ("config_path", "xbench_catalog")

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--type", required=True, type=CaseFamily, choices=(CaseFamily.SERVING,), dest="family")
        parser.add_argument("--from", required=True, dest="source")

    def run(self, args: argparse.Namespace, config: XpoolDevConfig) -> int:
        location = BenchCatalog.from_file(config.xbench.catalog).clone(args.source, family=args.family)
        print(f"{location.id}\nEdit {location.path}:{location.first_line}-{location.last_line}")
        print("The copied experiment is unchanged. Edit its conditions and description before running it.")
        return 0
