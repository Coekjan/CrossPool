"""Headless paper figures and read-protected, checkout-independent benchmark reports."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import fill

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import JsonValue, TypeAdapter

from xbench.harness.serving.measure import (
    REPETITION_DIRECTORY_PATTERN,
    BenchCaseManifest,
    BenchRunManifest,
    BenchSummary,
    CdfPoint,
    RepetitionManifest,
    ThroughputBucket,
    load_measurement,
    retained_file,
)
from xbench.harness.serving.workload import PreparedWorkload
from xkit.config import FigureConfig, XbenchReportConfig, get_global_config
from xkit.results import RunStore, write_json


@dataclass(frozen=True, slots=True)
class ReportSeries:
    label: str
    directory: Path
    workload: PreparedWorkload
    summary: BenchSummary
    case_manifest: BenchCaseManifest
    repetition_manifest: RepetitionManifest
    environment: dict[str, JsonValue] = field(default_factory=dict)


def write_metric_csv(summary: BenchSummary, directory: Path) -> None:
    for name, values, fields in (
        (
            "cdf.csv",
            summary.cdf,
            tuple(CdfPoint.model_fields),
        ),
        (
            "throughput.csv",
            summary.throughput,
            tuple(ThroughputBucket.model_fields),
        ),
    ):
        with (directory / name).open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(value.model_dump() for value in values)


def load_series(directory: Path, label: str) -> ReportSeries:
    """Aggregate retained request metrics and event samples under a run lock.

    Diagnostic metadata may be unavailable; replay, raw observations and the
    incomplete archive/cleanup evidence stays visible beside valid samples.
    """

    case_directory = directory.parent
    case_manifest = BenchCaseManifest.model_validate_json(retained_file(case_directory, "case.json").read_bytes())
    workload = case_manifest.load_workload(case_directory)
    manifest, summary = load_measurement(directory.resolve(), workload)
    if not (directory.parents[2] / ".completed").is_file():
        summary = summary.model_copy(update={"evidence_complete": False})
    environment: dict[str, JsonValue]
    try:
        environment = TypeAdapter(dict[str, JsonValue]).validate_json(
            retained_file(directory, "environment.json").read_bytes()
        )
    except (OSError, ValueError) as error:
        environment = {"unavailable": True, "capture_errors": [str(error)]}
    return ReportSeries(label, directory, workload, summary, case_manifest, manifest, environment)


def retained_repetitions(
    directory: Path, manifest: BenchRunManifest
) -> tuple[tuple[Path, ...], tuple[Path, ...], bool]:
    """Inspect parent-declared repetition metadata, returning complete-run eligibility.

    This boundary checks sealing, addresses and required target declarations.
    Raw observations and digests are read only during generation.
    """
    if manifest.run_id != directory.name or not manifest.finished or manifest.result_code is None:
        raise ValueError(f"benchmark invocation is not sealed: {directory.name}")
    if not (directory / ".completed").is_file():
        raise ValueError(f"benchmark invocation is not sealed: {directory.name}")
    repetitions = []
    historical = []
    complete = len(manifest.case_directories) == len(manifest.selected_cases)
    seen_cases = set()
    for reference in manifest.case_directories:
        case_directory = (directory / reference).resolve()
        if not case_directory.is_relative_to(directory / "cases") or case_directory.parent != directory / "cases":
            raise ValueError("benchmark case reference escapes its invocation")
        case = BenchCaseManifest.model_validate_json((case_directory / "case.json").read_bytes())
        if (
            str(case.case.id) != case_directory.name
            or case.case.id not in manifest.selected_cases
            or case.case.id in seen_cases
        ):
            raise ValueError("benchmark retained case does not match its parent selection")
        seen_cases.add(case.case.id)
        effective = case.effective_attempts(case_directory)
        count = manifest.tool_config.settings.xbench.repetitions
        if any(number > count for number in effective):
            raise ValueError("retained repetition exceeds the original configured count")
        complete = complete and set(effective) == set(range(1, count + 1))
        for repetition_directory in effective.values():
            try:
                repetition = RepetitionManifest.from_directory(repetition_directory)
            except (OSError, ValueError):
                complete = False
                continue
            if repetition.sealed:
                repetitions.append(repetition_directory)
            else:
                complete = False
        for attempt in sorted(case_directory.iterdir()):
            if (
                attempt.is_symlink()
                or attempt in effective.values()
                or REPETITION_DIRECTORY_PATTERN.fullmatch(attempt.name) is None
            ):
                continue
            try:
                checkpoint = RepetitionManifest.from_directory(attempt)
            except (OSError, ValueError):
                continue
            if checkpoint.repetition <= count and checkpoint.sealed:
                historical.append(attempt)
    return tuple(repetitions), tuple(historical), complete and bool(repetitions)


def list_bench_artifacts(root: Path) -> tuple[str, ...]:
    """List sealed inactive target-format addresses using metadata only."""
    artifacts = []
    store = RunStore(root)
    for entry in store.inactive_runs():
        try:
            with store.read(entry.name) as directory:
                manifest = BenchRunManifest.model_validate_json((directory / "run.json").read_bytes())
                repetitions, historical, complete = retained_repetitions(directory, manifest)
                if complete:
                    artifacts.append(entry.name)
                else:
                    artifacts.extend(f"{entry.name}/{path.relative_to(directory).as_posix()}" for path in repetitions)
                artifacts.extend(f"{entry.name}/{path.relative_to(directory).as_posix()}" for path in historical)
        except (ValueError, FileNotFoundError, BlockingIOError):
            continue
    return tuple(artifacts)


def report_bench_runs(inputs: Sequence[Path], *, output: Path | None = None) -> tuple[Path, ...]:
    """Validate retained parents and generate independent repetition reports.

    The CLI supplies exact artifact paths. No catalogue or deployment sources
    are reopened. Presentation settings come from the installed aggregate.
    """
    if not inputs:
        raise ValueError("supply at least one benchmark artifact ID")
    config = get_global_config().xbench.report
    outputs = []
    seen = set()
    for address in (path.expanduser().absolute() for path in inputs):
        repetition_match = REPETITION_DIRECTORY_PATTERN.fullmatch(address.name)
        root = address.parents[2] if repetition_match is not None else address
        with RunStore(root.parent).read(root.name, exclusive=True) as protected:
            directory = address.resolve()
            manifest = BenchRunManifest.model_validate_json((protected / "run.json").read_bytes())
            repetitions, historical, complete = retained_repetitions(protected, manifest)
            if repetition_match is not None:
                if directory not in (*repetitions, *historical) or (
                    address.is_symlink() and directory not in repetitions
                ):
                    raise ValueError(f"benchmark repetition is unsealed or not declared by its parent: {directory}")
                if directory.parent != address.parent.resolve() or (
                    address.is_symlink()
                    and (
                        repetition_match[2] is not None
                        or RepetitionManifest.from_directory(directory).repetition != int(repetition_match[1])
                    )
                ):
                    raise ValueError("benchmark repetition link does not identify its current same-case attempt")
                selected = (directory,)
            else:
                if not complete:
                    raise ValueError(f"benchmark run has no complete reportable repetition set: {directory.name}")
                selected = repetitions
            for repetition in selected:
                if repetition in seen:
                    continue
                seen.add(repetition)
                item = load_series(repetition, f"{root.name}/{repetition.relative_to(protected).as_posix()}")
                destination = (
                    None
                    if output is None
                    else output.expanduser().resolve()
                    / "xbench"
                    / root.name
                    / repetition.relative_to(protected)
                    / "report"
                )
                outputs.append(render_report(item, config=config, output=destination, run=manifest))
    return tuple(outputs)


def render_report(
    item: ReportSeries,
    *,
    config: XbenchReportConfig,
    output: Path | None = None,
    run: BenchRunManifest | None = None,
) -> Path:
    """Render exact configured canvases from retained metrics under the run lock.

    Captions are Markdown text. Regeneration replaces owned files and removes
    obsolete formats, preserving raw evidence and unrelated output files.
    """
    destination = (item.directory / "report" if output is None else output).resolve()
    if destination.is_relative_to(item.directory.resolve()) and destination != item.directory.resolve() / "report":
        raise ValueError("benchmark report export overlaps source evidence")
    destination.mkdir(parents=True, exist_ok=True)
    write_metric_csv(item.summary, destination)
    settings = {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "grid.color": "0.85",
        "grid.linewidth": 0.4,
        "grid.linestyle": ":",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "text.usetex": False,
        "savefig.dpi": config.ppi,
    }
    colors = ("#1f4e79", "#b34a33", "#39745a", "#77558f", "#222222")
    line_styles = ("-", "--", "-.", ":")
    preset_width = {"single": 3.3, "double": 6.8, "half": 1.65}[config.layout]
    identities = (*(str(model_id) for model_id in item.workload.model_ids), "aggregate")
    styles = {
        identity: {
            "color": colors[index % len(colors)],
            "linestyle": line_styles[index % len(line_styles)],
            "linewidth": 1.5 if identity == "aggregate" else 1.0,
        }
        for index, identity in enumerate(identities)
    }
    match item.summary.window_kind:
        case "interrupted":
            window_note = "Interrupted measurement"
        case "observed_prefix":
            window_note = "Observed prefix; actual stop unknown"
        case _:
            window_note = None
    rendered: dict[str, JsonValue] = {}

    def save(figure: Figure, name: str, figure_config: FigureConfig) -> None:
        """Fit the shared legend inside this canvas and publish selected formats."""
        if config.legend.visible:
            handles_by_label = {}
            for axis in figure.axes:
                handles, labels = axis.get_legend_handles_labels()
                handles_by_label.update(zip(labels, handles, strict=True))
            if handles_by_label:
                legend_args = {
                    "ncols": config.legend.columns,
                    "frameon": False,
                }
                if config.legend.location == "best":
                    figure.axes[0].legend(
                        tuple(handles_by_label.values()), tuple(handles_by_label), loc="best", **legend_args
                    )
                else:
                    legend = figure.legend(
                        tuple(handles_by_label.values()),
                        tuple(handles_by_label),
                        loc=config.legend.location,
                        **legend_args,
                    )
                    figure.canvas.draw()
                    bounds = legend.get_window_extent().transformed(figure.transFigure.inverted())
                    rect = [0.0, 0.0, 1.0, 1.0]
                    if config.legend.location.startswith("upper"):
                        rect[3] = max(0.2, 1.0 - bounds.height - 0.02)
                    elif config.legend.location.startswith("lower"):
                        rect[1] = min(0.8, bounds.height + 0.02)
                    elif config.legend.location in {"center left", "center right", "right"}:
                        if config.legend.location == "center left":
                            rect[0] = min(0.8, bounds.width + 0.02)
                        else:
                            rect[2] = max(0.2, 1.0 - bounds.width - 0.02)
                    figure.tight_layout(rect=(rect[0], rect[1], rect[2], rect[3]))
        for extension in ("pdf", "svg", "png"):
            artifact = destination / f"{name}.{extension}"
            if extension in config.formats:
                figure.savefig(artifact, dpi=config.ppi)
            else:
                artifact.unlink(missing_ok=True)
        rendered[name] = {
            "width_inches": float(figure.get_figwidth()),
            "height_inches": float(figure.get_figheight()),
            "columns": figure_config.columns,
            "outputs": [f"{name}.{extension}" for extension in config.formats],
        }

    with matplotlib.rc_context(settings):
        for name, metrics, figure_config in (
            ("ttft-cdf", (("http_ttft_seconds", "HTTP TTFT"), ("arrival_ttft_seconds", "Arrival TTFT")), config.ttft),
            (
                "itl-cdf",
                (
                    ("itl_observed", "Observed ITL"),
                    ("itl_estimated", "Token-estimated ITL"),
                    ("itl_combined", "Combined token-weighted ITL"),
                ),
                config.itl,
            ),
            (
                "throughput",
                (
                    ("input_tokens_per_second", "Logical input (including cache hits)"),
                    ("output_tokens_per_second", "Observed successful output"),
                ),
                config.throughput,
            ),
        ):
            rows = math.ceil(len(metrics) / figure_config.columns)
            width = figure_config.width_inches or preset_width
            height = figure_config.height_inches or 2.4 * rows
            text_width = max(10, int(width * 9 / figure_config.columns))
            figure = Figure(figsize=(width, height))
            FigureCanvasAgg(figure)
            if window_note is not None:
                figure.suptitle(window_note, fontsize=8)
            axes = figure.subplots(rows, figure_config.columns, squeeze=False)
            for axis, (metric, title) in zip(axes.flat, metrics):
                unavailable = []
                for identity in identities:
                    label = fill(identity, width=max(10, int(width * 9 / config.legend.columns)))
                    if name == "throughput":
                        buckets = tuple(bucket for bucket in item.summary.throughput if bucket.target_id == identity)
                        if not buckets:
                            unavailable.append(label)
                            continue
                        axis.stairs(
                            [
                                bucket.input_tokens_per_second
                                if metric == "input_tokens_per_second"
                                else bucket.output_tokens_per_second
                                for bucket in buckets
                            ],
                            [*(bucket.start_seconds for bucket in buckets), buckets[-1].end_seconds],
                            baseline=None,
                            label=label,
                            **styles[identity],
                        )
                    else:
                        points = tuple(
                            point
                            for point in item.summary.cdf
                            if point.target_id == identity and point.metric == metric
                        )
                        if not points:
                            unavailable.append(label)
                            continue
                        axis.step(
                            [points[0].value_seconds * 1000, *(point.value_seconds * 1000 for point in points)],
                            [0.0, *(point.cumulative_probability for point in points)],
                            where="post",
                            label=label,
                            **styles[identity],
                        )
                if name == "throughput":
                    axis.set(
                        xlabel=fill("Time since origin (s)", width=text_width),
                        ylabel="Tokens / second",
                        title=fill(title, width=text_width),
                    )
                    if item.summary.window_end_seconds is not None:
                        axis.set_xlim(right=item.summary.window_end_seconds)
                        if item.workload.arrival_horizon_seconds <= item.summary.window_end_seconds:
                            axis.axvline(
                                item.workload.arrival_horizon_seconds, color="0.3", linewidth=0.6, linestyle=":"
                            )
                            if item.workload.arrival_horizon_seconds < item.summary.window_end_seconds:
                                axis.axvspan(
                                    item.workload.arrival_horizon_seconds,
                                    item.summary.window_end_seconds,
                                    color="0.94",
                                    zorder=-1,
                                )
                                axis.text(
                                    0.98,
                                    0.98,
                                    "shaded: queue drain",
                                    transform=axis.transAxes,
                                    ha="right",
                                    va="top",
                                    fontsize=6,
                                )
                else:
                    axis.set(xlabel=fill(f"{title} (ms)", width=text_width), ylabel="Empirical CDF", ylim=(0, 1.02))
                axis.xaxis.labelpad += axis.xaxis.label.get_fontsize()
                axis.set_xlim(left=0)
                axis.grid(True)
                axis.set_axisbelow(True)
                if unavailable:
                    axis.text(
                        0.02,
                        0.98,
                        "Unavailable:\n" + "\n".join(unavailable),
                        transform=axis.transAxes,
                        va="top",
                        fontsize=6,
                    )
            for axis in tuple(axes.flat)[len(metrics) :]:
                axis.set_visible(False)
            figure.tight_layout()
            save(figure, name, figure_config)

    write_json(
        destination / "render.json",
        {
            "matplotlib_version": matplotlib.__version__,
            "settings": config.model_dump(mode="json"),
            "figures": rendered,
            "font": "DejaVu Serif",
            "rc_params": settings,
            "artifact_id": item.label,
            "series_styles": [{"target_id": identity, **styles[identity]} for identity in identities],
            "latency_figure_units": "milliseconds",
            "throughput_units": "tokens per second",
        },
    )
    write_report_projection(item, output=destination, run=run, config=config)
    return destination


def write_report_projection(
    item: ReportSeries, *, output: Path, config: XbenchReportConfig, run: BenchRunManifest | None = None
) -> None:
    """Export one repetition's summary and original execution/cleanup verdicts."""

    write_json(
        output / "summary.json",
        {
            "original_run": run.model_dump(mode="json") if run is not None else None,
            "label": item.label,
            "directory": str(item.directory),
            "prompt_sha256": item.workload.prompt_sha256,
            "trace_sha256": item.workload.trace_sha256,
            "environment": item.environment,
            "warmup_sha256": item.workload.warmup_sha256,
            "deployment": item.case_manifest.deployment.model_dump(mode="json")
            if item.case_manifest.deployment is not None
            else None,
            "serving_metadata": item.case_manifest.serving_metadata.model_dump(mode="json")
            if item.case_manifest.serving_metadata is not None
            else None,
            "original_result_code": item.repetition_manifest.result_code,
            "summary": item.summary.model_dump(mode="json"),
        },
    )
    lines = [
        f"# {config.title or 'CrossPool Benchmark Report'}",
        "",
        "Main distributions contain successful requests. Partial output remains separate.",
        "",
    ]
    if run is not None:
        lines.extend(
            [
                f"## Invocation: {run.run_id}",
                "",
                f"Original result: {run.result_code}; execution finished: {run.finished}.",
                f"Selected cases: {', '.join(str(identity) for identity in run.selected_cases)}.",
                f"Retained cases: {', '.join(run.case_directories) or 'none'}.",
                f"Infrastructure error: {run.infrastructure_error or 'none'}.",
                "",
            ]
        )
    summary = item.summary
    lines.extend(
        [
            f"## {item.label}",
            "",
            f"Evidence: {item.directory}",
            "",
            f"Execution complete: {summary.execution_complete}; cleanup verified: {summary.cleanup_verified}; "
            f"evidence complete: {summary.evidence_complete}.",
            "",
            f"Outcomes: {summary.outcomes}. Arrival horizon: {summary.arrival_horizon_seconds}s; "
            f"window end: {summary.window_end_seconds}s; window kind: {summary.window_kind}.",
            "",
            f"Prompt digest: {item.workload.prompt_sha256}; trace digest: {item.workload.trace_sha256}.",
            "",
        ]
    )
    if summary.window_kind == "observed_prefix":
        lines.extend(["Throughput covers only the retained observation prefix; the actual stop time is unknown.", ""])
    elif summary.window_kind == "interrupted":
        lines.extend(["Throughput covers the measured interval through interruption, excluding teardown.", ""])
    if "aggregate" in summary.targets:
        distributions = summary.targets["aggregate"].distributions
        lines.extend(
            [
                f"ITL represented intervals: {distributions['itl_combined'].sample_count}; "
                f"observed: {distributions['itl_observed'].sample_count}; "
                f"token-estimated: {distributions['itl_estimated'].sample_count}; "
                f"mean represented-interval coverage: {distributions['itl_coverage'].mean}.",
                "",
            ]
        )
    for figure, settings in (("ttft-cdf", config.ttft), ("itl-cdf", config.itl), ("throughput", config.throughput)):
        lines.extend([f"## {figure}", ""])
        lines.extend(f"[{figure}.{extension}]({figure}.{extension})" for extension in config.formats)
        if settings.caption is not None:
            lines.extend(["", settings.caption])
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
