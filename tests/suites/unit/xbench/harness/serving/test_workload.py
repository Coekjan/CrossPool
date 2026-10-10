from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from xbench.harness.serving.case import BenchCase, ClientBenchCase, LogNormalTokens, TokenRange
from xbench.harness.serving.workload import prepare_workload, sample_tokens
from xpool.integrations.sglang.devkit.requests import LocalModelMetadata, SglangRequestLimits
from xpool.model import ModelId
from xtest.harness.support.config import TEST_CASE_ID


@pytest.mark.parametrize("random_prompts", [False, True])
@pytest.mark.parametrize("poisson_arrivals", [False, True])
def test_all_prompt_and_arrival_combinations_are_replayable_and_target_scoped(
    random_prompts: bool, poisson_arrivals: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = workload_case(tmp_path, random_prompts=random_prompts, poisson_arrivals=poisson_arrivals)
    stub_metadata(monkeypatch, tmp_path)
    workload = prepare_workload(case)
    assert workload == prepare_workload(case)
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


@pytest.mark.parametrize(("random_prompts", "ratio"), [(False, 2.0), (True, 2.0), (True, 0.25)])
def test_interval_relative_lengths_obey_joint_bounds_and_keep_isolated_model_streams(
    random_prompts: bool, ratio: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_metadata(monkeypatch, tmp_path)
    raw = workload_case(tmp_path, random_prompts=random_prompts, poisson_arrivals=True).model_dump(mode="json")
    for target in raw["targets"]:
        if random_prompts:
            target["prompts"]["input_tokens"] = {"kind": "lognormal", "median_fraction": 0.125, "sigma": 1.0}
        target["output_tokens"] = {
            "kind": "lognormal",
            "median_fraction": 0.03125,
            "sigma": 1.0,
            "max_output_input_ratio": ratio,
        }
    case = TypeAdapter(BenchCase).validate_json(json.dumps(raw))
    workload = prepare_workload(case)
    assert workload == prepare_workload(case)
    assert workload.model_contexts == {target.model_id: 128 for target in case.targets}
    prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
    for request in (*workload.requests, *workload.warmup):
        length = prompts[request.model_id, request.prompt_id].input_tokens
        assert length is not None and math.ceil(1 / ratio) <= length <= 121
        assert 1 <= request.max_new_tokens <= min(math.floor(ratio * length), 126 - length)
        if not random_prompts:
            assert length == 3
    remaining = case.targets[0].model_id
    raw["targets"] = raw["targets"][:1]
    raw["arrivals"]["rates"] = {str(remaining): raw["arrivals"]["rates"][str(remaining)]}
    isolated = prepare_workload(TypeAdapter(BenchCase).validate_json(json.dumps(raw)))
    assert isolated.requests == tuple(request for request in workload.requests if request.model_id == remaining)
    assert isolated.prompts == tuple(prompt for prompt in workload.prompts if prompt.model_id == remaining)


def test_token_sampling_conditions_intervals_without_rewriting_fixed_counts() -> None:
    for policy in (TokenRange(min=2, max=20), LogNormalTokens(kind="lognormal", median_fraction=0.125, sigma=1.0)):
        first, second = random.Random(0), random.Random(0)
        values = [sample_tokens(policy, first, minimum=7, maximum=9) for _ in range(16)]
        assert values == [sample_tokens(policy, second, minimum=7, maximum=9) for _ in range(16)]
        assert all(7 <= value <= 9 for value in values)
        assert sample_tokens(policy, first, minimum=9, maximum=9) == 9
        with pytest.raises(ValueError, match="no legal request interval"):
            sample_tokens(policy, first, minimum=10, maximum=9)
    assert sample_tokens(9, random.Random(0), minimum=7, maximum=9) == 9
    with pytest.raises(ValueError, match="outside the legal request interval"):
        sample_tokens(6, random.Random(0), minimum=7, maximum=9)


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
        "devices": [{"uuid": "GPU-remote", "total_memory_bytes": 81920 * 1024 * 1024}],
        "target_device_uuids": {"test/one": ["GPU-remote"]},
    }
    path.write_text(json.dumps(declaration), encoding="utf-8")
    metadata = declared_case.load_serving_metadata()
    assert metadata is not None and metadata.packages is None
    assert metadata.target_device_uuids == {ModelId("test/one"): ("GPU-remote",)}
    declaration["target_device_uuids"] = {"test/unknown": ["GPU-remote"]}
    path.write_text(json.dumps(declaration), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown benchmark target"):
        declared_case.load_serving_metadata()
    declaration["devices"][0]["total_memory_bytes"] = True
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
    with pytest.raises(ValueError, match=r"legal request interval|local model vocabulary"):
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
                "id": str(TEST_CASE_ID),
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
    metadata = LocalModelMetadata(
        vocab_size=20, limits=SglangRequestLimits.from_context(128), admissible_ids=(1, 3, 5), text_lengths={}
    )
    monkeypatch.setattr(LocalModelMetadata, "from_checkpoint", lambda path, *, text_prompts: metadata)
