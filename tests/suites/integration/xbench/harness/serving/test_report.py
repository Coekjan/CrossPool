from __future__ import annotations

import csv
import json
import shutil
import struct
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal

import matplotlib
import numpy
import pytest
from matplotlib.figure import Figure
from pydantic import JsonValue, TypeAdapter

import xbench.cli
import xbench.harness.serving.runner
from xbench.harness.serving.api import create_api_adapter
from xbench.harness.serving.case import BenchCase
from xbench.harness.serving.client import MeasurementRecorder, RequestState
from xbench.harness.serving.measure import (
    BenchCaseManifest,
    BenchRunManifest,
    RepetitionManifest,
    RequestRecord,
)
from xbench.harness.serving.report import (
    list_bench_artifacts,
    load_series,
    render_report,
    report_bench_runs,
)
from xbench.harness.serving.workload import (
    PreparedWorkload,
    ResolvedPrompt,
    ScheduledRequest,
    content_digest,
    file_digest,
    read_jsonl,
)
from xkit.config import XpoolDevConfig, get_global_config
from xkit.results import RunEntry, RunStore, write_json, write_jsonl
from xtest.harness.support.config import (
    TEST_CASE_ID,
    TEST_MODEL_ID,
    development_config,
    reset_development_config,
    tool_config_record,
)

pytestmark = pytest.mark.usefixtures(reset_development_config.__name__, development_config.__name__)


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, Figure]:
    figures: dict[str, Figure] = {}

    def save(figure: Figure, filename: Path, *, dpi: int) -> None:
        figures[filename.stem] = figure

    monkeypatch.setattr(Figure, "savefig", save)
    return figures


def test_report_exports_all_formats_with_embedded_fonts(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    output = render_report(load_series(repetition, "the retained run"), config=get_global_config().xbench.report)
    for name in ("ttft-cdf", "itl-cdf", "throughput"):
        assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF-")
        assert b"/FontFile2" in (output / f"{name}.pdf").read_bytes()
        assert "<svg" in (output / f"{name}.svg").read_text()
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_report_regeneration_applies_presentation_without_changing_evidence(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    series = load_series(repetition, "the retained run")
    output = render_report(series, config=get_global_config().xbench.report)
    (output / "notes.txt").write_text("retain these notes", encoding="utf-8")
    evidence = {path: path.read_bytes() for path in repetition.iterdir() if path.is_file()}
    config = XpoolDevConfig.from_mapping(
        {
            "xbench": {
                "report": {
                    "layout": "half",
                    "formats": ["png"],
                    "ppi": 100,
                    "legend": {"visible": False},
                    "ttft": {"width_inches": 4, "height_inches": 3, "columns": 2, "caption": "Latency comparison."},
                    "itl": {"columns": 2},
                }
            }
        }
    ).xbench.report
    render_report(series, config=config)
    assert struct.unpack(">II", (output / "ttft-cdf.png").read_bytes()[16:24]) == (400, 300)
    assert struct.unpack(">II", (output / "itl-cdf.png").read_bytes()[16:24]) == (165, 480)
    assert {path.name for path in output.glob("*.png")} == {"ttft-cdf.png", "itl-cdf.png", "throughput.png"}
    assert not tuple(output.glob("*.pdf")) and not tuple(output.glob("*.svg"))
    assert "Latency comparison." in (output / "report.md").read_text()
    assert (output / "notes.txt").read_text() == "retain these notes"
    assert evidence == {path: path.read_bytes() for path in evidence}


@pytest.mark.parametrize("layout", ["single", "double", "half"])
def test_report_renders_labeled_units_weights_and_layout(
    layout: Literal["single", "double", "half"], tmp_path: Path, captured_figures: dict[str, Figure]
) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    series = load_series(repetition, "the retained run")
    series = replace(
        series,
        summary=series.summary.model_copy(
            update={
                "cdf": tuple(
                    point.model_copy(update={"value_seconds": point.value_seconds * 1_000_000})
                    if point.metric == "itl_combined"
                    else point
                    for point in series.summary.cdf
                )
            }
        ),
    )
    original_font = matplotlib.rcParams["font.family"]
    output = repetition / "report"
    assert (
        render_report(series, config=get_global_config().xbench.report.model_copy(update={"layout": layout})) == output
    )
    if layout == "single":
        (output / "notes.txt").write_text("keep these notes", encoding="utf-8")
        (output / "cdf.csv").write_text("old generated CSV", encoding="utf-8")
        render_report(series, config=get_global_config().xbench.report.model_copy(update={"layout": layout}))
        assert (output / "notes.txt").read_text() == "keep these notes"
        assert "old generated CSV" not in (output / "cdf.csv").read_text()
    assert matplotlib.rcParams["font.family"] == original_font
    metadata = json.loads((output / "render.json").read_text())
    assert metadata["font"] == "DejaVu Serif"
    assert metadata["rc_params"]["savefig.dpi"] == 300
    assert metadata["figures"]["ttft-cdf"]["width_inches"] == {"single": 3.3, "double": 6.8, "half": 1.65}[layout]
    assert metadata["settings"]["legend"]["location"] == "upper center"
    ttft_figure = captured_figures["ttft-cdf"]
    itl_figure = captured_figures["itl-cdf"]
    throughput_figure = captured_figures["throughput"]
    assert ttft_figure.legends and all(axis.get_legend() is None for axis in ttft_figure.axes)
    assert len({value["linestyle"] for value in metadata["series_styles"]}) == 2
    assert all(axis.get_xlabel().endswith("(ms)") for axis in ttft_figure.axes)
    observed_itl = next(axis for axis in itl_figure.axes if " ".join(axis.get_xlabel().split()) == "Observed ITL (ms)")
    assert any("Unavailable" in note.get_text() for note in observed_itl.texts)
    http_ttft = next(axis for axis in ttft_figure.axes if " ".join(axis.get_xlabel().split()) == "HTTP TTFT (ms)")
    cdf_line = next(line for line in http_ttft.lines if str(line.get_label()).replace("\n", "") == str(TEST_MODEL_ID))
    assert numpy.asarray(cdf_line.get_xydata()).tolist() == [[100, 0], [100, 1]]
    assert any("queue drain" in note.get_text() for axis in throughput_figure.axes for note in axis.texts)
    for figure in captured_figures.values():
        figure.canvas.draw()
        for axis in figure.axes:
            for text in (axis.title, axis.xaxis.label):
                bounds = text.get_window_extent()
                assert figure.bbox.contains(bounds.x0, bounds.y0)
                assert figure.bbox.contains(bounds.x1, bounds.y1)
            offset = axis.xaxis.get_offset_text()
            if offset.get_text():
                assert not axis.xaxis.label.get_window_extent().overlaps(offset.get_window_extent())
    assert "mean represented-interval coverage: 0.6" in (output / "report.md").read_text()


def test_report_expands_repetitions_and_preserves_failed_source(
    tmp_path: Path, captured_figures: dict[str, Figure], capsys: pytest.CaptureFixture[str]
) -> None:
    run, repetition = retained_run(tmp_path, failed=True)
    output = repetition / "report"
    with pytest.raises(BlockingIOError):
        report_bench_runs((run.directory,))
    run.complete()
    assert list_bench_artifacts(run.directory.parent) == (run.directory.name,)
    (repetition / "environment.json").unlink()
    (repetition / "warmup.json").unlink()
    second = repetition.with_name("repetition-0002")
    shutil.copytree(repetition, second)
    checkpoint = RepetitionManifest.model_validate_json((second / "repetition.json").read_bytes())
    write_json(second / "repetition.json", checkpoint.model_copy(update={"repetition": 2}).model_dump(mode="json"))
    case_path = repetition.parent / "case.json"
    manifest = BenchCaseManifest.model_validate_json(case_path.read_bytes())
    run_path = run.directory / "run.json"
    run_manifest = BenchRunManifest.model_validate_json(run_path.read_bytes())
    settings = XpoolDevConfig.from_mapping({}, cli={"xbench_repetitions": 2}, env={})
    write_json(
        run_path,
        run_manifest.model_copy(
            update={
                "tool_config": run_manifest.tool_config.model_copy(
                    update={"settings": settings, "sources": settings.sources}
                )
            }
        ).model_dump(mode="json"),
    )
    write_json(
        case_path,
        manifest.model_copy(
            update={
                "repetitions": (repetition.name, second.name),
            }
        ).model_dump(mode="json"),
    )
    before = {path.relative_to(run.directory): path.read_bytes() for path in run.directory.rglob("*") if path.is_file()}
    assert xbench.cli.main(["report", run.directory.name]) == 0
    outputs = (output, second / "report")
    assert capsys.readouterr().out.splitlines() == [str(path) for path in outputs]
    after = {
        path.relative_to(run.directory): path.read_bytes()
        for path in run.directory.rglob("*")
        if path.is_file() and not any(path.is_relative_to(directory) for directory in outputs)
    }
    assert before == after
    for directory in outputs:
        report = json.loads((directory / "summary.json").read_text())
        assert report["summary"]["outcomes"]["failed"] == 1
        assert not report["summary"]["measurement_available"]
        assert report["environment"]["unavailable"] is True
    assert report_bench_runs((run.directory, repetition)) == outputs

    for unavailable in (
        checkpoint.model_copy(update={"repetition": 2, "finished": False, "result_code": None}).model_dump(mode="json"),
        {"repetition": "malformed"},
    ):
        write_json(second / "repetition.json", unavailable)
        assert list_bench_artifacts(run.directory.parent) == (
            f"{run.directory.name}/{repetition.relative_to(run.directory).as_posix()}",
        )
        with pytest.raises(ValueError, match="complete reportable repetition set"):
            report_bench_runs((run.directory,))
    export = tmp_path / "export"
    assert report_bench_runs((repetition,), output=export) == (
        export / "xbench" / run.directory.name / repetition.relative_to(run.directory) / "report",
    )


@pytest.mark.parametrize("failure", ["csv", "render"])
def test_report_output_failure_preserves_original_verdict_and_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, captured_figures: dict[str, Figure]
) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    before = {path.relative_to(run.directory): path.read_bytes() for path in run.directory.rglob("*") if path.is_file()}
    with monkeypatch.context() as patch:
        if failure == "csv":

            def write_rows(writer: csv.DictWriter, rows: Iterable[Mapping[str, JsonValue]]) -> None:
                raise OSError("report output unavailable")

            patch.setattr(csv.DictWriter, "writerows", write_rows)
        else:

            def save(figure: Figure, filename: Path, *, dpi: int) -> None:
                assert RunStore(run.directory.parent).cleanup(keep_runs=0, dry_run=True).active == (run.directory,)
                with pytest.raises(BlockingIOError):
                    with RunStore(run.directory.parent).read(run.directory.name, exclusive=True):
                        pytest.fail("report publication lost exclusive protection")
                raise OSError("report output unavailable")

            patch.setattr(Figure, "savefig", save)
        with pytest.raises(OSError, match="report output unavailable"):
            report_bench_runs((run.directory,))
    output = repetition / "report"
    assert report_bench_runs((repetition,)) == (output,)
    assert before == {
        path.relative_to(run.directory): path.read_bytes()
        for path in run.directory.rglob("*")
        if path.is_file() and not path.is_relative_to(output)
    }
    assert load_series(repetition, "original").summary.execution_complete
    assert json.loads((run.directory / "run.json").read_bytes())["result_code"] == 0


def test_report_rejects_corrupt_hashes_and_resealed_success_without_raw_evidence(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    manifest_path = repetition / "repetition.json"
    manifest = RepetitionManifest.model_validate_json(manifest_path.read_bytes())
    (repetition / "requests.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_series(repetition, "input")
    (repetition / "requests.jsonl").write_text("", encoding="utf-8")
    (repetition / "events.jsonl").write_text("", encoding="utf-8")
    digests = dict(manifest.artifact_sha256)
    for name in ("requests.jsonl", "events.jsonl"):
        digests[name] = file_digest(repetition / name)
    write_json(manifest_path, manifest.model_copy(update={"artifact_sha256": digests}).model_dump(mode="json"))
    with pytest.raises(ValueError, match="successful benchmark checkpoint"):
        load_series(repetition, "input")


@pytest.mark.parametrize("name", ["events.jsonl", "measurement.json"])
def test_report_requires_digests_for_its_raw_evidence(name: str, tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    path = repetition / "repetition.json"
    manifest = RepetitionManifest.model_validate_json(path.read_bytes())
    digests = dict(manifest.artifact_sha256)
    del digests[name]
    write_json(path, manifest.model_copy(update={"artifact_sha256": digests}).model_dump(mode="json"))
    with pytest.raises(ValueError, match=r"required.*digest"):
        load_series(repetition, "incomplete seal")


def test_report_marks_an_unavailable_final_checkpoint_incomplete(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    path = repetition / "repetition.json"
    manifest = json.loads(path.read_bytes())
    del manifest["raw_evidence_complete"]
    write_json(path, manifest)
    series = load_series(repetition, "missing checkpoint")
    assert series.repetition_manifest.result_code == 0
    assert series.summary.outcomes["success"] == 1 and not series.summary.evidence_complete


@pytest.mark.parametrize("name", ["prompts.jsonl", "trace.jsonl", "warmup.jsonl"])
def test_report_rejects_replay_content_that_disagrees_with_prepared_workload(name: str, tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    path = repetition.parent / name
    value = json.loads(path.read_text().splitlines()[0])
    value["text" if name == "prompts.jsonl" else "max_new_tokens"] = "changed prompt" if name == "prompts.jsonl" else 7
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content digest"):
        load_series(repetition, "changed replay")


def test_report_rejects_success_checkpoint_without_cleanup_proof(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    path = repetition / "repetition.json"
    manifest = RepetitionManifest.model_validate_json(path.read_bytes())
    write_json(path, manifest.model_copy(update={"cleanup_verified": False}).model_dump(mode="json"))
    with pytest.raises(ValueError, match=r"successful.*checkpoint"):
        load_series(repetition, "contradictory success")


@pytest.mark.parametrize("changed", [{"completion_tokens": 5}, {"first_token_at_seconds": 0.11}])
def test_report_rejects_resealed_request_claims_that_disagree_with_events(
    changed: dict[str, int | float], tmp_path: Path
) -> None:
    run, repetition = retained_run(tmp_path)
    run.complete()
    requests_path = repetition / "requests.jsonl"
    value = json.loads(requests_path.read_text())
    value.update(changed)
    requests_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    path = repetition / "repetition.json"
    manifest = RepetitionManifest.model_validate_json(path.read_bytes())
    digests = {**manifest.artifact_sha256, "requests.jsonl": file_digest(requests_path)}
    write_json(path, manifest.model_copy(update={"artifact_sha256": digests}).model_dump(mode="json"))
    with pytest.raises(ValueError, match="differ from accepted stream evidence"):
        load_series(repetition, "resealed contradiction")


def test_finalization_reports_abnormal_worker_exit_with_successful_requests(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    case = BenchCaseManifest.model_validate_json((repetition.parent / "case.json").read_bytes()).case
    workload = PreparedWorkload.load(repetition.parent, case=case)
    raw = (repetition / "requests.jsonl").read_bytes()
    assert (
        xbench.harness.serving.runner.finalize_repetition(
            repetition, workload, cleanup_verified=True, worker_code=1, infrastructure_error=None
        )
        == 2
    )
    run.complete()
    series = load_series(repetition, "worker failed")
    assert series.summary.outcomes["success"] == 1
    assert series.summary.infrastructure_error == "benchmark worker exited with code 1"
    assert (repetition / "requests.jsonl").read_bytes() == raw


def test_owner_loss_reports_only_retained_observation_prefix(
    tmp_path: Path, captured_figures: dict[str, Figure]
) -> None:
    run, repetition = retained_run(tmp_path, horizon=1000.0)
    case = BenchCaseManifest.model_validate_json((repetition.parent / "case.json").read_bytes()).case
    workload = PreparedWorkload.load(repetition.parent, case=case)
    write_json(repetition / "repetition.json", RepetitionManifest(repetition=1).model_dump(mode="json"))
    assert (
        xbench.harness.serving.runner.finalize_repetition(
            repetition, workload, cleanup_verified=True, worker_code=2, infrastructure_error=None
        )
        == 2
    )
    run.complete()
    series = load_series(repetition, "recovered prefix")
    assert series.summary.window_kind == "observed_prefix"
    assert series.summary.window_end_seconds == 0.18
    assert not series.summary.execution_complete
    assert series.summary.targets["aggregate"].distributions["http_ttft_seconds"].sample_count == 1
    output = render_report(series, config=get_global_config().xbench.report)
    assert all(axis.get_xlim()[1] == 0.18 for axis in captured_figures["throughput"].axes)
    assert "actual stop time is unknown" in (output / "report.md").read_text()


def test_finalization_preserves_raw_evidence_and_reports_recording_failure(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    case = BenchCaseManifest.model_validate_json((repetition.parent / "case.json").read_bytes()).case
    workload = PreparedWorkload.load(repetition.parent, case=case)
    raw = (repetition / "requests.jsonl").read_bytes()
    with (repetition / "events.jsonl").open("a", encoding="utf-8") as output:
        output.write('{"incomplete":')
    assert (
        xbench.harness.serving.runner.finalize_repetition(
            repetition, workload, cleanup_verified=True, worker_code=0, infrastructure_error=None
        )
        == 2
    )
    run.complete()
    series = load_series(repetition, "failure")
    assert series.summary.cleanup_verified and series.summary.infrastructure_error is not None
    assert (repetition / "requests.jsonl").read_bytes() == raw
    manifest = RepetitionManifest.model_validate_json((repetition / "repetition.json").read_bytes())
    assert manifest.finished and manifest.result_code == 2
    assert (repetition / "events.jsonl.partial").read_text().endswith('{"incomplete":')
    assert not series.summary.evidence_complete and not series.summary.execution_complete


def test_finalization_does_not_claim_success_after_terminal_events_are_lost(tmp_path: Path) -> None:
    run, repetition = retained_run(tmp_path)
    case = BenchCaseManifest.model_validate_json((repetition.parent / "case.json").read_bytes()).case
    workload = PreparedWorkload.load(repetition.parent, case=case)
    requests = (repetition / "requests.jsonl").read_bytes()
    events = (repetition / "events.jsonl").read_text().splitlines(keepends=True)
    (repetition / "events.jsonl").write_text("".join(events[:-1]) + '{"unfinished":', encoding="utf-8")
    assert (
        xbench.harness.serving.runner.finalize_repetition(
            repetition, workload, cleanup_verified=True, worker_code=0, infrastructure_error=None
        )
        == 2
    )
    run.complete()
    series = load_series(repetition, "lost terminal evidence")
    assert not series.summary.execution_complete and not series.summary.evidence_complete
    assert not series.summary.measurement_available
    assert series.summary.outcomes["failed"] == 1
    assert series.summary.targets["aggregate"].partial_output_tokens == 6
    assert (repetition / "requests.jsonl.unverified").read_bytes() == requests
    assert (repetition / "events.jsonl.partial").read_text().endswith('{"unfinished":')


@pytest.mark.parametrize("horizon,cleanup_verified,worker_code", [(0.0, True, 2), (10.0, False, 143)])
def test_owner_loss_after_origin_remains_reportable_without_a_measurement_window(
    tmp_path: Path, horizon: float, cleanup_verified: bool, worker_code: int, captured_figures: dict[str, Figure]
) -> None:
    run, repetition = retained_run(tmp_path, horizon=horizon)
    case = BenchCaseManifest.model_validate_json((repetition.parent / "case.json").read_bytes()).case
    workload = PreparedWorkload.load(repetition.parent, case=case)
    for name in ("requests.jsonl", "events.jsonl"):
        (repetition / name).write_text("", encoding="utf-8")
    write_json(repetition / "repetition.json", RepetitionManifest(repetition=1).model_dump(mode="json"))
    origin = repetition / "measurement.json"
    original_origin = origin.read_bytes()
    assert (
        xbench.harness.serving.runner.finalize_repetition(
            repetition, workload, cleanup_verified=cleanup_verified, worker_code=worker_code, infrastructure_error=None
        )
        == worker_code
    )
    run_manifest = BenchRunManifest.model_validate_json((run.directory / "run.json").read_bytes())
    write_json(
        run.directory / "run.json", run_manifest.model_copy(update={"result_code": worker_code}).model_dump(mode="json")
    )
    run.complete()

    requests = read_jsonl(repetition / "requests.jsonl", RequestRecord)
    assert len(requests) == 1 and requests[0].error_kind == "evidence_missing"
    assert (
        requests[0].enqueued_at_seconds is requests[0].http_started_at_seconds is requests[0].ended_at_seconds is None
    )
    series = load_series(repetition, "owner lost")
    assert series.repetition_manifest.finished and series.repetition_manifest.result_code == worker_code
    assert series.summary.cleanup_verified == cleanup_verified
    assert series.summary.infrastructure_error is not None
    assert series.summary.window_end_seconds is None
    assert not series.summary.measurement_available and not series.summary.evidence_complete
    assert not series.summary.execution_complete
    assert series.summary.outcomes["failed"] == 1
    assert not series.summary.targets and not series.summary.cdf and not series.summary.throughput
    assert "measurement.json" in series.repetition_manifest.artifact_sha256
    if not cleanup_verified:
        return
    assert report_bench_runs((run.directory,)) == (repetition / "report",)
    assert origin.read_bytes() == original_origin
    origin.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"digest mismatch: measurement\.json"):
        load_series(repetition, "changed origin")
    digests = {**series.repetition_manifest.artifact_sha256, "measurement.json": file_digest(origin)}
    write_json(
        repetition / "repetition.json",
        series.repetition_manifest.model_copy(update={"artifact_sha256": digests}).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="monotonic_origin_seconds"):
        load_series(repetition, "invalid origin")


def retained_run(tmp_path: Path, *, failed: bool = False, horizon: float = 0.1) -> tuple[RunEntry, Path]:
    store = RunStore(tmp_path / ".xpool-cache/bench-runs")
    run = store.start("20260927-180000-123-456")
    case_directory = run.directory / "cases" / str(TEST_CASE_ID)
    repetition = case_directory / "repetition-0001"
    repetition.mkdir(parents=True)
    prompts = (ResolvedPrompt(prompt_id="p", model_id=TEST_MODEL_ID, text="offline replay data"),)
    scheduled = ScheduledRequest(
        request_id="r", model_id=TEST_MODEL_ID, prompt_id="p", arrival_seconds=0.0, max_new_tokens=6
    )
    workload = PreparedWorkload(
        case_id=TEST_CASE_ID,
        model_ids=(TEST_MODEL_ID,),
        model_contexts={},
        prompts=prompts,
        requests=(scheduled,),
        warmup=(scheduled,),
        arrival_horizon_seconds=horizon,
        bucket_seconds=0.1,
        seeds={},
        prompt_sha256=content_digest(prompts),
        trace_sha256=content_digest((scheduled,)),
        warmup_sha256=content_digest((scheduled,)),
    )
    recorder = MeasurementRecorder(repetition)
    state = RequestState(scheduled, recorder, create_api_adapter("sglang"), enqueued_at=0.0, started_at=0.0, status=200)
    state.event("enqueue", 0.0)
    state.event("http_start", 0.0)
    state.consume(b'{"meta_info":{"completion_tokens":3,"prompt_tokens":5}}', 0.1)
    state.consume(b'{"meta_info":{"completion_tokens":6,"prompt_tokens":5,"finish_reason":{"type":"length"}}}', 0.16)
    if not failed:
        state.consume(b"[DONE]", 0.18)
    else:
        state.event("error", 0.18, error_kind="protocol", error_message="EOF without DONE")
    recorder.request(state.terminal("failed" if failed else "success", 0.18))
    recorder.close()
    write_json(repetition / "environment.json", {"mode": "client", "local_device_inventory": None})
    write_json(
        repetition / "measurement.json",
        {"monotonic_origin_seconds": 1000.0, "wall_clock_origin_seconds": 1000000000.0},
    )
    write_json(repetition / "warmup.json", {"requests": []})
    write_json(
        case_directory / "workload.json",
        {
            **workload.model_dump(mode="json", exclude={"case_id", "model_ids", "prompts", "requests", "warmup"}),
            "inputs": {"prompts": "prompts.jsonl", "requests": "trace.jsonl", "warmup": "warmup.jsonl"},
        },
    )
    write_jsonl(case_directory / "prompts.jsonl", (prompt.model_dump(mode="json") for prompt in workload.prompts))
    write_jsonl(case_directory / "trace.jsonl", (request.model_dump(mode="json") for request in workload.requests))
    write_jsonl(case_directory / "warmup.jsonl", (request.model_dump(mode="json") for request in workload.warmup))
    case = TypeAdapter(BenchCase).validate_json(
        json.dumps(
            {
                "id": str(TEST_CASE_ID),
                "description": "Render retained serving measurements without executing the workload.",
                "module": "serving.multi_model",
                "mode": "client",
                "arrivals": {"kind": "jsonl", "path": "trace.jsonl", "duration_seconds": horizon},
                "bucket_seconds": 0.1,
                "targets": [
                    {
                        "model_id": str(TEST_MODEL_ID),
                        "base_url": "http://localhost:8000",
                        "prompts": {"kind": "jsonl", "path": "prompts.jsonl"},
                    }
                ],
            }
        )
    )
    case_manifest = BenchCaseManifest(
        case=case,
        prompt_sha256=workload.prompt_sha256,
        trace_sha256=workload.trace_sha256,
        warmup_sha256=workload.warmup_sha256,
        repetitions=(repetition.name,),
    )
    write_json(case_directory / "case.json", case_manifest.model_dump(mode="json"))
    manifest = RepetitionManifest(
        repetition=1,
        finished=True,
        result_code=1 if failed else 0,
        cleanup_verified=True,
        raw_evidence_complete=True,
        window_end_seconds=0.18,
        window_kind="complete",
        artifact_sha256={
            name: file_digest(repetition / name) for name in ("requests.jsonl", "events.jsonl", "measurement.json")
        },
    )
    write_json(repetition / "repetition.json", manifest.model_dump(mode="json"))
    run_manifest = BenchRunManifest(
        run_id=run.directory.name,
        tool_config=tool_config_record(tmp_path / ".xpool-cache"),
        selected_cases=(TEST_CASE_ID,),
        case_directories=(f"cases/{TEST_CASE_ID}",),
        finished=True,
        result_code=1 if failed else 0,
    )
    write_json(run.directory / "run.json", run_manifest.model_dump(mode="json"))
    return run, repetition
