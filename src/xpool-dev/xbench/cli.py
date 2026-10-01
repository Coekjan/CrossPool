"""Installed CrossPool benchmark inventory, execution, report and retention commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xbench.harness.collection import collect_programs
from xbench.harness.serving.case import BenchCatalog
from xbench.harness.serving.report import report_bench_runs
from xbench.harness.serving.runner import run_benchmarks
from xkit.cli import positive_integer, print_cleanup
from xkit.results import RunStore

__all__ = ["main"]


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one benchmark command from the checkout or an explicit external catalog.

    Inventory validates declarations only. Offline reporting preserves source
    verdicts and returns zero even for failed source runs. Invalid input and
    infrastructure failures return two; run request failures return one.
    """

    parser = argparse.ArgumentParser(prog="xbench", description="CrossPool benchmark command")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "run"):
        command = commands.add_parser(
            name,
            description="Retains resolved prompts and arrival traces for replay; cleanup is explicit."
            if name == "run"
            else None,
        )
        command.add_argument("--catalog", type=Path, default=Path("benches/benches.toml"))
        command.add_argument("--case", action="append", dest="cases")
        if name == "run":
            command.add_argument("--result-root", type=Path, default=Path(".xpool-cache/bench-runs"))
    report = commands.add_parser("report")
    report.add_argument("inputs", nargs="+", type=Path)
    report.add_argument("--label", action="append", dest="labels")
    report.add_argument("--layout", choices=("single", "double"), default="single")
    clean = commands.add_parser("clean")
    clean.add_argument("--result-root", type=Path, default=Path(".xpool-cache/bench-runs"))
    selection = clean.add_mutually_exclusive_group()
    selection.add_argument("--keep", type=positive_integer, default=20)
    selection.add_argument("--all", action="store_true", dest="remove_all")
    clean.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(None if arguments is None else list(arguments))
    try:
        match options.command:
            case "list":
                catalogue_path = options.catalog.expanduser().resolve()
                catalog = BenchCatalog.load(catalogue_path)
                cases = catalog.select(tuple(options.cases or ()))
                programs = collect_programs(catalogue_path, cases)
                for case, program in zip(cases, programs, strict=True):
                    models = ",".join(str(target.model_id) for target in case.targets)
                    resources = program.requirements
                    print(
                        f"{case.id}\tmode={case.mode} targets={models} "
                        f"gpus={resources.cuda_count} mps={resources.requires_mps} config={resources.requires_config}"
                    )
            case "run":
                catalogue_path = options.catalog.expanduser().resolve()
                catalog = BenchCatalog.load(catalogue_path)
                return run_benchmarks(
                    catalog.select(tuple(options.cases or ())), root=options.result_root, catalogue_path=catalogue_path
                )
            case "report":
                for output in report_bench_runs(options.inputs, labels=options.labels or (), layout=options.layout):
                    print(output)
            case "clean":
                cleanup = RunStore(options.result_root.expanduser()).cleanup(
                    keep_runs=0 if options.remove_all else options.keep, dry_run=options.dry_run
                )
                print_cleanup(cleanup)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"xpool benchmark {options.command} failure: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
