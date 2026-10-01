import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from xbench.harness.serving.case import (
    BenchCase,
    BenchCatalog,
    ClientBenchCase,
    JsonlPrompts,
    TraceArrivals,
)
from xpool.model import ModelId
from xtest.harness.support.config import TEST_MODEL_ID


def test_catalog_resolves_owning_paths_and_preserves_selection_order(tmp_path: Path) -> None:
    path = tmp_path / "catalog.toml"
    path.write_text(
        '[serving_cases.a]\nmode = "client"\ndescription = "First target."\nmodule = "serving.multi_model"\n'
        '[serving_cases.a.arrivals]\nkind = "jsonl"\npath = "trace.jsonl"\n'
        f'[[serving_cases.a.targets]]\nmodel_id = "{TEST_MODEL_ID}"\nbase_url = "http://localhost:8000"\n'
        'prompts = {kind = "jsonl", path = "prompts.jsonl"}\n'
        '[serving_cases.b]\nmode = "client"\ndescription = "Second target."\nmodule = "serving.multi_model"\n'
        '[serving_cases.b.arrivals]\nkind = "jsonl"\npath = "trace.jsonl"\n'
        f'[[serving_cases.b.targets]]\nmodel_id = "{TEST_MODEL_ID}"\nbase_url = "http://localhost:8001"\n'
        'prompts = {kind = "jsonl", path = "prompts.jsonl"}\n',
        encoding="utf-8",
    )
    catalog = BenchCatalog.load(path)
    assert tuple(case.id for case in catalog.select(())) == ("a", "b")
    assert tuple(case.id for case in catalog.select(("b", "a"))) == ("b", "a")
    case = catalog.cases[0]
    assert isinstance(case, ClientBenchCase)
    assert isinstance(case.arrivals, TraceArrivals)
    assert isinstance(case.targets[0].prompts, JsonlPrompts)
    assert case.arrivals.path == tmp_path / "trace.jsonl"
    assert case.targets[0].prompts.path == tmp_path / "prompts.jsonl"
    with pytest.raises(ValueError, match="unique"):
        catalog.select(("a", "a"))
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
        "id": "case",
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
    catalog = BenchCatalog.load(Path("benches/benches.toml"))
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
        "id": "case",
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
        "id": "case",
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
