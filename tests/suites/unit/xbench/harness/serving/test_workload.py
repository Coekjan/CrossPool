from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

import xbench.harness.serving.workload
from xbench.harness.serving.case import BenchCase, ClientBenchCase
from xbench.harness.serving.workload import LocalMetadata, PreparedWorkload, prepare_workload
from xpool.model import ModelId


@pytest.mark.parametrize("random_prompts", [False, True])
@pytest.mark.parametrize("poisson_arrivals", [False, True])
def test_all_prompt_and_arrival_combinations_are_replayable_and_target_scoped(
    random_prompts: bool, poisson_arrivals: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = workload_case(tmp_path, random_prompts=random_prompts, poisson_arrivals=poisson_arrivals)
    stub_metadata(monkeypatch, tmp_path)
    workload = prepare_workload(case)
    assert workload == prepare_workload(case)
    assert PreparedWorkload.model_validate_json(workload.model_dump_json()) == workload
    prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
    for request in workload.requests:
        prompt = prompts[request.model_id, request.prompt_id]
        if random_prompts:
            assert prompt.input_ids is not None
            assert 2 <= len(prompt.input_ids) <= 5
            assert set(prompt.input_ids) <= {1, 3, 5}
        else:
            assert prompt.input_ids == (1, 2, 3)  # Explicit input may include a special token.
        assert 2 <= request.max_new_tokens <= 4
    if poisson_arrivals:
        assert workload.requests and all(0 < request.arrival_seconds < 20 for request in workload.requests)
        assert all(2 <= request.max_new_tokens <= 4 for request in workload.warmup)
    else:
        assert tuple(request.request_id for request in workload.requests) == ("first", "tied", "late")
        assert workload.arrival_horizon_seconds == 20
        assert workload.requests[0].prompt_id == workload.requests[2].prompt_id


def test_prompt_changes_and_warmup_do_not_perturb_arrivals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_metadata(monkeypatch, tmp_path)
    raw = workload_case(tmp_path, random_prompts=True, poisson_arrivals=True)
    baseline = prepare_workload(raw)
    changed = raw.model_dump(mode="json")
    changed["targets"][0]["prompts"]["input_tokens"] = 17
    changed["warmup_requests_per_target"] = 4
    altered = prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(changed)))
    assert baseline.requests == altered.requests
    assert baseline.trace_sha256 == altered.trace_sha256
    assert baseline.prompt_sha256 != altered.prompt_sha256
    assert len(altered.warmup) == 8


def test_declared_serving_metadata_validates_external_conditions(tmp_path: Path) -> None:
    case = workload_case(tmp_path, random_prompts=False, poisson_arrivals=False)
    path = tmp_path / "serving.json"
    raw = case.model_dump(mode="json")
    raw["serving_metadata_path"] = str(path)
    for target in raw["targets"]:
        target["model_metadata_path"] = None
    declared_case = TypeAdapter(BenchCase).validate_json(json.dumps(raw))
    assert isinstance(declared_case, ClientBenchCase)
    declaration = {
        "schema_version": 1,
        "gpus": [{"uuid": "GPU-remote", "total_memory_bytes": 81920 * 1024 * 1024}],
        "target_gpu_uuids": {"test/one": ["GPU-remote"]},
    }
    path.write_text(json.dumps(declaration), encoding="utf-8")
    metadata = declared_case.load_serving_metadata()
    assert metadata is not None and metadata.packages is None
    assert metadata.target_gpu_uuids == {ModelId("test/one"): ("GPU-remote",)}
    declaration["target_gpu_uuids"] = {"test/unknown": ["GPU-remote"]}
    path.write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown benchmark target"):
        declared_case.load_serving_metadata()
    declaration["gpus"][0]["total_memory_bytes"] = True
    path.write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="total_memory_bytes"):
        declared_case.load_serving_metadata()


def test_poisson_trace_replay_preserves_idle_horizon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_metadata(monkeypatch, tmp_path)
    case = workload_case(tmp_path, random_prompts=True, poisson_arrivals=True)
    workload = prepare_workload(case)
    trace = tmp_path / "replay.jsonl"
    trace.write_text("".join(request.model_dump_json() + "\n" for request in workload.requests), encoding="utf-8")
    raw = case.model_dump(mode="json")
    raw["arrivals"] = {"kind": "jsonl", "path": str(trace), "duration_seconds": 20}
    for target in raw["targets"]:
        target.pop("output_tokens")
    replayed = prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(raw)))
    assert replayed.requests == workload.requests
    assert replayed.arrival_horizon_seconds == workload.arrival_horizon_seconds
    assert replayed.prompt_sha256 == workload.prompt_sha256


def test_empty_poisson_is_preserved_and_rate_is_sane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_metadata(monkeypatch, tmp_path)
    raw = workload_case(tmp_path, random_prompts=True, poisson_arrivals=True).model_dump(mode="json")
    raw["arrivals"]["duration_seconds"] = 1e-10
    assert prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(raw))).requests == ()
    raw["arrivals"]["duration_seconds"] = 1000
    raw["arrivals"]["rates"] = {"test/one": 3, "test/two": 0}
    workload = prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(raw)))
    assert 2500 < len(workload.requests) < 3500
    assert {request.model_id for request in workload.requests} == {ModelId("test/one")}
    assert {request.model_id for request in workload.warmup} == {ModelId("test/one"), ModelId("test/two")}


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("duplicate", "request IDs must be unique"),
        ("unknown-target", "unknown target ID"),
        ("unknown-prompt", "unresolved prompt reference"),
        ("short-horizon", "arrival horizon"),
        ("boolean-token", "input_ids"),
    ],
)
def test_invalid_dataset_is_rejected_before_execution(
    failure: str, reason: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_metadata(monkeypatch, tmp_path)
    case = workload_case(tmp_path, random_prompts=False, poisson_arrivals=False)
    raw = case.model_dump(mode="json")
    path = tmp_path / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "duplicate":
        rows[1]["request_id"] = rows[0]["request_id"]
    elif failure == "unknown-target":
        rows[0]["model_id"] = "test/missing"
    elif failure == "unknown-prompt":
        rows[0]["prompt_id"] = "missing"
    elif failure == "short-horizon":
        raw["arrivals"]["duration_seconds"] = 1
    else:
        (tmp_path / "prompts.jsonl").write_text('{"prompt_id":"same","input_ids":[true]}\n', encoding="utf-8")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(raw)))


@pytest.mark.parametrize("random_prompts", [False, True])
def test_prompt_metadata_rejects_unselected_file_values_and_generated_input_overflow(
    random_prompts: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_metadata(monkeypatch, tmp_path)
    case = workload_case(tmp_path, random_prompts=random_prompts, poisson_arrivals=False)
    raw = case.model_dump(mode="json")
    if random_prompts:
        raw["targets"][0]["prompts"]["input_tokens"] = 129
    else:
        with (tmp_path / "prompts.jsonl").open("a", encoding="utf-8") as output:
            output.write('{"prompt_id":"unselected","input_ids":[20]}\n')
    with pytest.raises(ValueError, match=r"known model input limit|local model vocabulary"):
        prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(raw)))


def workload_case(tmp_path: Path, *, random_prompts: bool, poisson_arrivals: bool) -> BenchCase:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text('{"prompt_id":"same","input_ids":[1,2,3]}\n', encoding="utf-8")
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": id,
                    "model_id": target,
                    "arrival_seconds": arrival,
                    "prompt_id": "same",
                    "max_new_tokens": 3,
                }
            )
            for id, target, arrival in (("late", "test/one", 9), ("first", "test/one", 2), ("tied", "test/two", 2))
        )
        + "\n",
        encoding="utf-8",
    )
    prompts = (
        {"kind": "random", "input_tokens": {"min": 2, "max": 5}}
        if random_prompts
        else {"kind": "jsonl", "path": str(prompt_path)}
    )
    arrivals = (
        {
            "kind": "poisson",
            "duration_seconds": 20,
            "rates": {"test/one": 1, "test/two": 0.5},
        }
        if poisson_arrivals
        else {"kind": "jsonl", "path": str(trace_path), "duration_seconds": 20}
    )
    return TypeAdapter(BenchCase).validate_json(
        json.dumps(
            {
                "id": "case",
                "description": "Replay deterministic per-model traffic from declared input sources.",
                "module": "serving.multi_model",
                "mode": "client",
                "seed": 23,
                "arrivals": arrivals,
                "targets": [
                    {
                        "model_id": id,
                        "base_url": "http://localhost:8000",
                        "model_metadata_path": str(tmp_path),
                        "prompts": prompts,
                        "output_tokens": ({"min": 2, "max": 4} if id == "test/one" else 3)
                        if poisson_arrivals
                        else None,
                    }
                    for id in ("test/one", "test/two")
                ],
            }
        )
    )


def stub_metadata(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    metadata = LocalMetadata(vocab_size=20, max_input_tokens=128)
    monkeypatch.setattr(
        xbench.harness.serving.workload, "load_local_metadata", lambda path, *, prompts=(): (metadata, (1, 3, 5))
    )
