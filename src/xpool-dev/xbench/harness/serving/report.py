"""Headless paper figures and read-protected, checkout-independent benchmark reports."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from textwrap import fill
from typing import Literal

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import JsonValue, TypeAdapter

from xbench.harness.serving.measure import (
    BenchCaseManifest,
    BenchRunManifest,
    BenchSummary,
    CdfPoint,
    MeasurementOrigin,
    RepetitionManifest,
    RequestRecord,
    StreamEvent,
    ThroughputBucket,
    summarize,
    unavailable_summary,
)
from xbench.harness.serving.workload import PreparedWorkload, file_digest, read_jsonl
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


def retained_file(directory: Path, reference: str) -> Path:
    """Resolve a retained artifact inside its owning directory, rejecting escapes."""

    path = (directory / reference).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"benchmark artifact is missing or escapes its owner: {reference}")
    return path


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

    manifest = RepetitionManifest.model_validate_json(retained_file(directory, "repetition.json").read_bytes())
    case_directory = directory.parent
    case_manifest = BenchCaseManifest.model_validate_json(retained_file(case_directory, "case.json").read_bytes())
    workload = PreparedWorkload.load(case_directory, case=case_manifest.case)
    if (
        workload.prompt_sha256 != case_manifest.prompt_sha256
        or workload.trace_sha256 != case_manifest.trace_sha256
        or workload.warmup_sha256 != case_manifest.warmup_sha256
    ):
        raise ValueError("retained replay content identities disagree")
    if workload.bucket_seconds != case_manifest.case.bucket_seconds or (
        case_manifest.case.arrivals.duration_seconds is not None
        and workload.arrival_horizon_seconds != case_manifest.case.arrivals.duration_seconds
    ):
        raise ValueError("retained workload declarations disagree with the case")
    required = {"events.jsonl", "requests.jsonl"}
    if (
        manifest.window_end_seconds is not None
        or "measurement.json" in manifest.artifact_sha256
        or (directory / "measurement.json").is_file()
    ):
        required.add("measurement.json")
    if not required <= manifest.artifact_sha256.keys():
        raise ValueError("benchmark repetition lacks required evidence digests")
    for reference in required:
        artifact = retained_file(directory, reference)
        if file_digest(artifact) != manifest.artifact_sha256[reference]:
            raise ValueError(f"retained benchmark artifact digest mismatch: {reference}")
    if "measurement.json" in required:
        MeasurementOrigin.model_validate_json(retained_file(directory, "measurement.json").read_bytes())
    requests = read_jsonl(directory / "requests.jsonl", RequestRecord)
    events = read_jsonl(directory / "events.jsonl", StreamEvent)
    if manifest.window_end_seconds is None and events:
        raise ValueError("benchmark events require an available measurement window")
    if (manifest.window_end_seconds is None) != (manifest.window_kind is None):
        raise ValueError("measurement window end and kind must be available together")
    summary = (
        summarize(
            workload,
            requests,
            events,
            window_end_seconds=manifest.window_end_seconds,
            window_kind=manifest.window_kind,
        )
        if manifest.window_end_seconds is not None and manifest.window_kind is not None
        else unavailable_summary(workload, requests)
    ).model_copy(
        update={"cleanup_verified": manifest.cleanup_verified, "infrastructure_error": manifest.infrastructure_error}
    )
    if manifest.result_code == 0 and (
        not summary.execution_complete
        or not summary.measurement_available
        or not summary.evidence_complete
        or not summary.cleanup_verified
        or manifest.raw_evidence_complete is False
        or summary.infrastructure_error is not None
        or any(summary.outcomes[kind] for kind in ("failed", "cancelled", "not_sent"))
    ):
        raise ValueError("successful benchmark checkpoint contradicts its completeness or outcomes")
    if (
        not manifest.finished
        or manifest.result_code is None
        or manifest.cleanup_verified is None
        or manifest.raw_evidence_complete is not True
        or not (directory.parents[2] / ".completed").is_file()
    ):
        summary = summary.model_copy(update={"evidence_complete": False})
    if manifest.raw_evidence_complete is False:
        summary = summary.model_copy(update={"execution_complete": False})
    environment: dict[str, JsonValue]
    try:
        environment = TypeAdapter(dict[str, JsonValue]).validate_json(
            retained_file(directory, "environment.json").read_bytes()
        )
    except (OSError, ValueError) as error:
        environment = {"unavailable": True, "capture_errors": [str(error)]}
    return ReportSeries(label, directory, workload, summary, case_manifest, manifest, environment)


def report_bench_runs(
    inputs: Sequence[Path], *, labels: Sequence[str], layout: Literal["single", "double"]
) -> tuple[Path, ...]:
    """Regenerate an independent report inside each selected repetition.

    Run inputs expand to all retained repetitions. Exclusive run protection
    covers loading and publication, coordinating report writers and cleanup.
    """

    if labels and len(labels) != len(inputs):
        raise ValueError("supply exactly one label per benchmark input directory")
    directories = tuple(path.expanduser().resolve() for path in inputs)
    outputs = []
    for index, directory in enumerate(directories):
        repetition_input = (directory / "repetition.json").is_file()
        root = directory.parents[2] if repetition_input else directory
        with RunStore(root.parent).read(root.name, exclusive=True) as protected:
            manifest = BenchRunManifest.model_validate_json((protected / "run.json").read_bytes())
            if repetition_input:
                selected = (directory,)
            else:
                selected_list: list[Path] = []
                for reference in manifest.case_directories:
                    case_directory = (protected / reference).resolve()
                    if not case_directory.is_relative_to(protected):
                        raise ValueError("benchmark case reference escapes its invocation")
                    case = BenchCaseManifest.model_validate_json((case_directory / "case.json").read_bytes())
                    for reference in case.repetitions:
                        repetition_directory = (case_directory / reference).resolve()
                        if not repetition_directory.is_relative_to(case_directory):
                            raise ValueError("benchmark repetition reference escapes its case")
                        selected_list.append(repetition_directory)
                selected = tuple(selected_list)
            for repetition in selected:
                suffix = f"{repetition.parent.name}/{repetition.name}"
                label = f"{labels[index]} / {suffix}" if labels else f"{root.name} / {suffix}"
                item = load_series(repetition, label)
                if not manifest.finished:
                    item = replace(item, summary=item.summary.model_copy(update={"evidence_complete": False}))
                outputs.append(render_report(item, layout=layout, run=manifest))
    return tuple(outputs)


def render_report(
    item: ReportSeries,
    *,
    layout: Literal["single", "double"] = "single",
    run: BenchRunManifest | None = None,
) -> Path:
    """Regenerate one repetition's report using retained ECDF/bucket data.

    Headless PDF/SVG/300-DPI PNG use a local paper style, embedded fonts and
    unsmoothed observations. The caller holds the run's exclusive lock.
    Generated files are overwritten; errors may leave a partially updated
    report, while measurement files and unrelated report files remain intact.
    """

    output = (item.directory / "report").resolve()
    if not output.is_relative_to(item.directory.resolve()):
        raise ValueError("benchmark report directory escapes its repetition")
    output.mkdir(exist_ok=True)
    write_metric_csv(item.summary, output)
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
        "savefig.dpi": 300,
    }
    colors = ("#1f4e79", "#b34a33", "#39745a", "#77558f", "#222222")
    line_styles = ("-", "--", "-.", ":")
    width = 3.3 if layout == "single" else 6.8
    identities = (*(str(model_id) for model_id in item.workload.model_ids), "aggregate")
    styles = {
        id: {
            "color": colors[index % len(colors)],
            "linestyle": line_styles[index % len(line_styles)],
            "linewidth": 1.5 if id == "aggregate" else 1.0,
        }
        for index, id in enumerate(identities)
    }
    match item.summary.window_kind:
        case "interrupted":
            window_note = "Interrupted measurement"
        case "observed_prefix":
            window_note = "Observed prefix; actual stop unknown"
        case _:
            window_note = None

    def save(figure: Figure, name: str) -> None:
        legend = {}
        for axis in figure.axes:
            handles, labels = axis.get_legend_handles_labels()
            legend.update(zip(labels, handles, strict=True))
        if legend:
            figure.legend(
                tuple(legend.values()),
                tuple(legend),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncols=1 if layout == "single" else 2,
                frameon=False,
            )
        for extension in ("pdf", "svg", "png"):
            figure.savefig(output / f"{name}.{extension}", bbox_inches="tight")

    with matplotlib.rc_context(settings):
        for name, metrics in (
            ("ttft-cdf", (("http_ttft_seconds", "HTTP TTFT"), ("arrival_ttft_seconds", "Arrival TTFT"))),
            (
                "itl-cdf",
                (
                    ("itl_observed", "Observed ITL"),
                    ("itl_estimated", "Token-estimated ITL"),
                    ("itl_combined", "Combined token-weighted ITL"),
                ),
            ),
        ):
            rows, columns = (len(metrics), 1) if layout == "single" else (1, len(metrics))
            figure = Figure(figsize=(width, 2.05 * rows))
            FigureCanvasAgg(figure)
            if window_note is not None:
                figure.suptitle(window_note, fontsize=8)
            axes = figure.subplots(rows, columns, squeeze=False)
            for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
                unavailable: list[str] = []
                for id in identities:
                    label = fill(id, width=32 if layout == "single" else 42)
                    points = tuple(
                        point for point in item.summary.cdf if point.target_id == id and point.metric == metric
                    )
                    if not points:
                        unavailable.append(label)
                        continue
                    axis.step(
                        [points[0].value_seconds * 1000, *(point.value_seconds * 1000 for point in points)],
                        [0.0, *(point.cumulative_probability for point in points)],
                        where="post",
                        label=label,
                        **styles[id],
                    )
                axis.set(xlabel=f"{title} (ms)", ylabel="Empirical CDF", ylim=(0, 1.02))
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
            figure.tight_layout()
            save(figure, name)

        figure = Figure(figsize=(width, 4.1))
        FigureCanvasAgg(figure)
        if window_note is not None:
            figure.suptitle(window_note, fontsize=8)
        axes = figure.subplots(2, 1, sharex=True)
        for axis, metric, title in zip(
            axes,
            ("input_tokens_per_second", "output_tokens_per_second"),
            ("Logical input (including cache hits)", "Observed successful output"),
            strict=True,
        ):
            for id in identities:
                buckets = tuple(bucket for bucket in item.summary.throughput if bucket.target_id == id)
                if not buckets:
                    continue
                edges = [*(bucket.start_seconds for bucket in buckets), buckets[-1].end_seconds]
                axis.stairs(
                    [getattr(bucket, metric) for bucket in buckets],
                    edges,
                    baseline=None,
                    label=fill(id, width=32 if layout == "single" else 42),
                    **styles[id],
                )
            if (
                item.summary.window_end_seconds is not None
                and item.workload.arrival_horizon_seconds <= item.summary.window_end_seconds
            ):
                axis.axvline(
                    item.workload.arrival_horizon_seconds,
                    color=styles["aggregate"]["color"],
                    linewidth=0.6,
                    linestyle=":",
                )
                if item.summary.window_end_seconds > item.workload.arrival_horizon_seconds:
                    axis.axvspan(
                        item.workload.arrival_horizon_seconds,
                        item.summary.window_end_seconds,
                        color="0.94",
                        zorder=-1,
                    )
            axis.set(ylabel="Tokens / second", title=title)
            axis.grid(True)
            axis.set_axisbelow(True)
            if not axis.patches:
                axis.text(0.5, 0.5, "Unavailable: no measurement window", transform=axis.transAxes, ha="center")
        axes[-1].set_xlabel("Time since origin (s)")
        axes[-1].set_xlim(left=0)
        if item.summary.window_end_seconds is not None:
            axes[-1].set_xlim(right=item.summary.window_end_seconds)
        axes[-1].text(0.98, 0.98, "shaded: queue drain", transform=axes[-1].transAxes, ha="right", va="top", fontsize=6)
        figure.tight_layout()
        save(figure, "throughput")

    write_json(
        output / "render.json",
        {
            "matplotlib_version": matplotlib.__version__,
            "layout": layout,
            "width_inches": width,
            "font": "DejaVu Serif",
            "rc_params": settings,
            "label": item.label,
            "series_styles": [{"target_id": id, **styles[id]} for id in identities],
            "latency_figure_units": "milliseconds",
            "throughput_units": "tokens per second",
            "legend_location": "below figure",
        },
    )
    write_report_projection(item, output=output, run=run)
    return output


def write_report_projection(item: ReportSeries, *, output: Path, run: BenchRunManifest | None = None) -> None:
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
        "# CrossPool Benchmark Report",
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
                f"Selected cases: {', '.join(run.selected_cases)}.",
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
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
