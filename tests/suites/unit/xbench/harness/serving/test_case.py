import json
from pathlib import Path

import pytest
import tomli_w
from pydantic import TypeAdapter

from xbench.harness.serving.case import (
    BenchCase,
    BenchCatalog,
    ClientBenchCase,
    JsonlPrompts,
    OutputTokenCount,
    TokenCount,
    TraceArrivals,
)
from xkit.case import CaseId
from xpool.model import ModelId
from xtest.harness.support.config import TEST_CASE_ID, TEST_MODEL_ID


def test_log_normal_policies_validate_role_specific_ratios_and_interval_fractions() -> None:
    policy = {"kind": "lognormal", "median_fraction": 0.125, "sigma": 1.0}
    TypeAdapter(TokenCount).validate_python(policy)
    TypeAdapter(OutputTokenCount).validate_python({**policy, "max_output_input_ratio": 2.0})
    with pytest.raises(ValueError, match="max_output_input_ratio"):
        TypeAdapter(TokenCount).validate_python({**policy, "max_output_input_ratio": 2.0})
    with pytest.raises(ValueError, match="max_output_input_ratio"):
        TypeAdapter(OutputTokenCount).validate_python({"min": 1, "max": 8, "max_output_input_ratio": 2.0})
    for adapter in (TypeAdapter(TokenCount), TypeAdapter(OutputTokenCount)):
        for fraction in (0.0, 1.01):
            with pytest.raises(ValueError, match="median_fraction"):
                adapter.validate_python({**policy, "median_fraction": fraction})
    with pytest.raises(ValueError, match="max_output_input_ratio"):
        TypeAdapter(OutputTokenCount).validate_python({**policy, "max_output_input_ratio": 0.0})


def test_catalog_resolves_owning_paths_and_preserves_selection_order(tmp_path: Path) -> None:
    path = tmp_path / "catalog.toml"
    first = TEST_CASE_ID
    second = CaseId("550e8401-e29b-41d4-a716-446655440000")
    path.write_text(
        tomli_w.dumps(
            {
                "serving_cases": {
                    str(identity): {
                        "mode": "client",
                        "description": description,
                        "module": "serving.multi_model",
                        "arrivals": {"kind": "jsonl", "path": "trace.jsonl"},
                        "targets": [
                            {
                                "model_id": str(TEST_MODEL_ID),
                                "base_url": f"http://localhost:{port}",
                                "prompts": {"kind": "jsonl", "path": "prompts.jsonl"},
                            }
                        ],
                    }
                    for identity, description, port in (
                        (first, "First target.", 8000),
                        (second, "Second target.", 8001),
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    catalog = BenchCatalog.from_file(path)
    assert tuple(case.id for case in catalog.select(())) == (first, second)
    assert tuple(case.id for case in catalog.select(("550e8401", "550e8400"))) == (second, first)
    case = catalog.cases[0]
    assert isinstance(case, ClientBenchCase)
    assert isinstance(case.arrivals, TraceArrivals)
    assert isinstance(case.targets[0].prompts, JsonlPrompts)
    assert case.arrivals.path == tmp_path / "trace.jsonl"
    assert case.targets[0].prompts.path == tmp_path / "prompts.jsonl"
    assert case.request_timeout_seconds is None
    with pytest.raises(ValueError, match="unique"):
        catalog.select(("550e8400", str(first)))
    with pytest.raises(ValueError, match="unknown"):
        catalog.select(("missing",))


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"runtime_config": "/private/xpool.toml"}, "runtime_config"),
        ({"seed": "0"}, "seed"),
        ({"max_inflight": True}, "max_inflight"),
        ({"request_timeout_seconds": 0}, "request_timeout_seconds"),
        ({"id": "../outside"}, "id"),
        (
            {
                "arrivals": {
                    "kind": "poisson",
                    "duration_seconds": 10,
                    "rates": {str(TEST_MODEL_ID): 0},
                }
            },
            "positive rate",
        ),
        (
            {
                "arrivals": {
                    "kind": "poisson",
                    "duration_seconds": 10,
                    "rates": {"test/other": 1},
                }
            },
            "rates must cover",
        ),
        (
            {"arrivals": {"kind": "poisson", "duration_seconds": 10, "rates": {str(TEST_MODEL_ID): 1}}},
            "targets require output_tokens",
        ),
        (
            {
                "targets": [
                    {
                        "model_id": str(TEST_MODEL_ID),
                        "base_url": "http://localhost:8000",
                        "prompts": {"kind": "jsonl", "path": "p.jsonl"},
                        "output_tokens": 8,
                    }
                ]
            },
            "target output_tokens must be absent",
        ),
    ],
)
def test_case_rejects_mixed_modes_invalid_scalars_and_incomplete_maps(update: dict[str, object], reason: str) -> None:
    case: dict[str, object] = {
        "id": str(TEST_CASE_ID),
        "description": "Serving declaration validation.",
        "module": "serving.multi_model",
        "mode": "client",
        "arrivals": {"kind": "jsonl", "path": "trace.jsonl"},
        "targets": [
            {
                "model_id": str(TEST_MODEL_ID),
                "base_url": "http://localhost:8000",
                "prompts": {"kind": "jsonl", "path": "p.jsonl"},
            }
        ],
    }
    TypeAdapter(BenchCase).validate_json(json.dumps(case))
    case.update(update)
    with pytest.raises(ValueError, match=reason):
        TypeAdapter(BenchCase).validate_json(json.dumps(case))


def test_initial_catalog_declares_portable_two_model_deployment() -> None:
    catalog = BenchCatalog.from_file(Path("benches/benches.toml"))
    case = catalog.cases[0]
    assert case.mode == "owned"
    assert (
        case.deployment
        == Path("configs/deployments/Qwen%2FQwen2.5-0.5B+Qwen%2FQwen3-0.6B/atn1-ffn1-lanes2.toml").resolve()
    )
    assert case.deployment.is_file()
    assert case.runtime_config is None
    assert tuple(target.model_id for target in case.targets) == (
        ModelId("Qwen/Qwen2.5-0.5B"),
        ModelId("Qwen/Qwen3-0.6B"),
    )
    assert all(target.ignores_eos() for target in case.targets)


def test_random_client_requires_local_model_metadata() -> None:
    raw = {
        "id": str(TEST_CASE_ID),
        "description": "Random prompts require a local tokenizer.",
        "module": "serving.multi_model",
        "mode": "client",
        "arrivals": {"kind": "jsonl", "path": "trace.jsonl"},
        "targets": [
            {
                "model_id": str(TEST_MODEL_ID),
                "base_url": "http://localhost:8000",
                "prompts": {"kind": "random", "input_tokens": 8},
            }
        ],
    }
    with pytest.raises(ValueError, match="model_metadata_path"):
        TypeAdapter(BenchCase).validate_json(json.dumps(raw))


@pytest.mark.parametrize("mode", ["client", "owned"])
def test_case_requires_one_target_per_model(mode: str) -> None:
    target: dict[str, object] = {"model_id": str(TEST_MODEL_ID), "prompts": {"kind": "jsonl", "path": "prompts.jsonl"}}
    raw: dict[str, object] = {
        "id": str(TEST_CASE_ID),
        "description": "One endpoint for each model.",
        "module": "serving.multi_model",
        "mode": mode,
        "arrivals": {"kind": "jsonl", "path": "trace.jsonl"},
        "targets": [target],
    }
    if mode == "owned":
        raw["deployment"] = "atn1-ffn1-lanes1"
        target["graph_mode"] = "eager"
    else:
        target["base_url"] = "http://localhost:8000"
    case = TypeAdapter(BenchCase).validate_json(json.dumps(raw))
    assert case.targets[0].model_id == TEST_MODEL_ID
    raw["targets"] = [target, dict(target)]
    with pytest.raises(ValueError, match="unique Model IDs"):
        TypeAdapter(BenchCase).validate_json(json.dumps(raw))
