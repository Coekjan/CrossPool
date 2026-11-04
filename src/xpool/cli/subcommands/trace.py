"""Explicit offline trace commands; raw artifact publication belongs to runtime owners."""

import argparse
import sys
from pathlib import Path

from xpool.config import XpoolConfig
from xpool.observability import ObservationError, build_calibration_report
from xpool.observability.trace import (
    QueryName,
    TraceError,
    export_traces,
    load_inputs,
    setup_processor,
    write_queries,
)
from xpool.observability.trace.export import encoded
from xpool.observability.trace.inputs import prepare_output
from xpool.utils.cli import CliCommandGroup, RunnableCliCommand


class TraceCommand(CliCommandGroup[XpoolConfig]):
    """Group offline validation, protobuf export and pinned processor queries."""

    name = "trace"
    help = "validate, export and query CrossPool observation artifacts"
    subparser_dest = "trace_command"


class TraceSetupCommand(RunnableCliCommand[XpoolConfig]):
    """Explicitly prepare the qualified Trace Processor binary."""

    name = "setup"
    parent = "trace"
    help = "download and verify pinned trace processor"
    config_settings = ("cache_root",)

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Allow an explicit tool directory instead of the configured cache."""
        parser.add_argument("--tool-dir", type=Path)
        parser.set_defaults(trace_handler=self.run_offline)

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Dispatch the offline command without consuming deployment state."""
        return self.run_offline(args)

    def run_offline(self, args: argparse.Namespace) -> int:
        """Download only when explicitly invoked; report preparation failures."""
        try:
            print(setup_processor(args.tool_dir or XpoolConfig.resolve_cache_root(cli=vars(args))[0] / "perfetto"))
        except TraceError as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0


class TraceValidateCommand(RunnableCliCommand[XpoolConfig]):
    """Validate explicit raw inputs and publish calibration/quality evidence."""

    name = "validate"
    parent = "trace"
    help = "validate raw artifacts and calibration declarations"
    config_settings = ()

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Require separate input, manifest and output paths."""
        parser.add_argument("--input-dir", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.set_defaults(trace_handler=self.run_offline)

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Dispatch the offline command without consuming deployment state."""
        return self.run_offline(args)

    def run_offline(self, args: argparse.Namespace) -> int:
        """Keep evidence failures structured and return a CLI validation status."""
        try:
            view, index, digests = load_inputs(args.input_dir, args.manifest)
            prepare_output(args.output_dir, (args.input_dir, args.manifest))
            (args.output_dir / "calibration.json").write_text(build_calibration_report(index, view).to_json())
            (args.output_dir / "quality.json").write_text(
                encoded(
                    {
                        "findings": view.quality.findings,
                        "checkpoints": tuple(view.quality.checkpoints.values()),
                        "inputs": digests,
                    }
                )
                + "\n"
            )
        except (TraceError, ObservationError) as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0


class TraceExportCommand(TraceValidateCommand):
    """Export verified reconstruction and mappings into a new trace bundle."""

    name = "export"
    help = "export standard Perfetto protobuf traces"

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Dispatch the offline command without consuming deployment state."""
        return self.run_offline(args)

    def run_offline(self, args: argparse.Namespace) -> int:
        """Publish only derived artifacts, leaving every input unchanged."""
        try:
            view, index, digests = load_inputs(args.input_dir, args.manifest)
            export_traces(view, index, args.output_dir, protected=(args.input_dir, args.manifest), inputs=digests)
        except (TraceError, ObservationError) as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0


class TraceQueryCommand(RunnableCliCommand[XpoolConfig]):
    """Query each trace member and publish JSON results."""

    name = "query"
    parent = "trace"
    help = "run built-in queries or SQL on an exported bundle"
    config_settings = ("cache_root",)

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Require the bundle and separate result directory."""
        parser.add_argument("--input-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.set_defaults(trace_handler=self.run_offline)
        parser.add_argument("--processor", type=Path)
        selection = parser.add_mutually_exclusive_group()
        selection.add_argument("--name", action="append", choices=tuple(QueryName), type=QueryName)
        selection.add_argument("--sql", type=Path)

    def run(self, args: argparse.Namespace, config: XpoolConfig) -> int:
        """Dispatch the offline command without consuming deployment state."""
        return self.run_offline(args)

    def run_offline(self, args: argparse.Namespace) -> int:
        """Use a prepared local binary; never implicitly download dependencies."""
        try:
            write_queries(
                args.input_dir,
                args.processor or XpoolConfig.resolve_cache_root(cli=vars(args))[0] / "perfetto/trace_processor_shell",
                args.output_dir,
                names=tuple(args.name) if args.name else tuple(QueryName),
                sql_file=args.sql,
            )
        except TraceError as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0
