from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from pathlib import Path
from typing import cast

import pytest
import tomli_w
from pydantic import ValidationError

from xpool.config import ConfigSource, MissingRequiredConfig, ModelConfig, XpoolConfig
from xpool.model import ModelId
from xtest.harness.support.config import TEST_MODEL_ID


def test_config_snapshot_round_trips_effective_fields_and_omits_env_only_debug(tmp_path: Path) -> None:
    environment = {
        "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
        "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(tmp_path / "observations"),
    }
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 900, "tbt_ms": 30}, "ffn_concurrency": 2},
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2, 3], "loader": {"parallelism": 2}},
            "models": [
                {
                    "id": "test/model-a",
                    "path": str(tmp_path / "a"),
                    "atn_tp_size": 1,
                    "atn_dp_size": 2,
                    "ffn_tp_size": 2,
                },
                {"id": "test/model-b", "path": str(tmp_path / "b"), "slo": {"ttft_ms": 800, "tbt_ms": 20}},
            ],
        },
        env=environment,
    )
    mapping = config.to_config_mapping()
    snapshot = tmp_path / "snapshot.toml"
    snapshot.write_text(tomli_w.dumps(mapping), encoding="utf-8")
    restored = XpoolConfig.from_file(snapshot, env=environment)

    assert "debug" not in mapping
    assert restored.model_dump(mode="json") == config.model_dump(mode="json")
    assert restored.sources != config.sources
    assert config.models[1].atn_tp_size is None
    assert config.models[1].atn_dp_size == 1
    sources = {record["name"]: record["source"] for record in config.sources}
    assert sources["models[0].atn_tp_size"] is ConfigSource.CONFIG
    assert sources["models[1].atn_tp_size"] is ConfigSource.UNSET
    assert sources["models[1].atn_dp_size"] is ConfigSource.DEFAULT


@pytest.mark.parametrize("field", ["atn_tp_size", "atn_dp_size"])
@pytest.mark.parametrize("value", [True, "2", 0, -1])
def test_model_attention_widths_require_positive_integers(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"id": str(TEST_MODEL_ID), field: value})


def test_independent_model_declaration_preserves_optional_widths() -> None:
    model = ModelConfig(id=TEST_MODEL_ID)
    assert model.atn_tp_size is None and model.ffn_tp_size is None
    assert model.atn_dp_size == 1


def test_omitted_attention_tp_partitions_the_world_by_dp() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/test/model", "atn_dp_size": 2}],
        }
    )
    assert config.models[0].atn_tp_size is None
    assert config.atn_tp_size_of(TEST_MODEL_ID) == 1
    assert config.atn_world_size == 2
    restored = XpoolConfig.from_mapping(config.to_config_mapping())
    assert restored.models[0].atn_tp_size is None


@pytest.mark.parametrize("geometry", [(1, 1), (2, 2), (None, 3)])
def test_attention_geometry_must_cover_the_world(geometry: tuple[int | None, int]) -> None:
    tp, dp = geometry
    with pytest.raises(ValidationError, match="attention"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0, 1]},
                "ffn": {"devices": [2]},
                "models": [
                    {"id": str(TEST_MODEL_ID), "path": "/models/test/model", "atn_tp_size": tp, "atn_dp_size": dp}
                ],
            }
        )


def test_example_config_defines_documented_topology() -> None:
    config = XpoolConfig.from_file(Path("configs/xpool.example.toml"))

    assert config.atn.devices == [0]
    assert config.atn.device_memory_utilization == 0.95
    assert config.ffn.devices == [1]
    assert config.ffn.loader.parallelism == 4
    assert config.ffn.device_memory_extra_margin_bytes == 0
    assert config.ffn.placement.parallelism == 4
    assert config.ffn.placement.timeout_seconds == 60
    assert config.ffn.device_memory_calibration is None
    assert config.models[0].id == ModelId("deepseek-ai/DeepSeek-V2-Lite-Chat")
    assert config.model_path_of(ModelId("deepseek-ai/DeepSeek-V2-Lite-Chat")) == Path(
        "/absolute/path/to/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )


def test_derived_placements_follow_declared_order() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2, 3, 4]},
            "models": [
                {"id": "test/m1", "path": "/models/m1"},
                {"id": "test/m2", "path": "/models/m2"},
            ],
        }
    )

    assert [(agent.rank, agent.device) for agent in config.atnagents] == [(0, 0), (1, 1)]
    assert [(agent.rank, agent.device) for agent in config.ffnagents] == [(0, 2), (1, 3), (2, 4)]
    assert [(instance.instance_index, instance.model_id) for instance in config.instances] == [
        (0, ModelId("test/m1")),
        (1, ModelId("test/m2")),
    ]
    assert config.atnagent_by_device[1].rank == 1
    assert config.ffnagent_by_device[3].rank == 1
    assert config.instance_by_model_id[ModelId("test/m2")].instance_index == 1


def test_latency_slo_uses_global_default_and_complete_model_override() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [
                {"id": "test/m1", "path": "/models/m1"},
                {"id": "test/m2", "path": "/models/m2", "slo": {"ttft_ms": 800, "tbt_ms": 40}},
            ],
        }
    )

    assert config.models[0].slo is None
    assert (config.scheduler.slo.ttft_ms, config.scheduler.slo.tbt_ms) == (1000, 50)
    assert config.models[1].slo is not None
    assert (config.models[1].slo.ttft_ms, config.models[1].slo.tbt_ms) == (800, 40)


@pytest.mark.parametrize(
    ("slo", "field"),
    [
        ({"ttft_ms": 1000}, "tbt_ms"),
        ({"ttft_ms": 0, "tbt_ms": 50}, "ttft_ms"),
        ({"ttft_ms": 1000, "tbt_ms": float("inf")}, "tbt_ms"),
        ({"ttft_ms": 1000, "tbt_ms": 50, "unknown": 1}, "unknown"),
    ],
)
def test_latency_slo_rejects_incomplete_nonpositive_nonfinite_and_extra_fields(
    slo: dict[str, object], field: str
) -> None:
    with pytest.raises(ValidationError) as error:
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": slo},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            }
        )
    assert any(item["loc"][-1] == field for item in error.value.errors())


def test_scheduler_latency_slo_is_required() -> None:
    with pytest.raises(MissingRequiredConfig, match="scheduler_slo"):
        XpoolConfig.from_mapping(
            {
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            }
        )


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError) as error:
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
                "unexpected": True,
            }
        )
    assert any(item["loc"][-1] == "unexpected" for item in error.value.errors())


def test_derived_placement_views_are_immutable() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )

    with pytest.raises(AttributeError):
        cast(MutableSequence[object], config.atnagents).append(config.atnagents[0])
    with pytest.raises(TypeError):
        cast(MutableMapping[int, object], config.atnagent_by_device)[2] = config.atnagents[0]
    with pytest.raises(TypeError):
        cast(MutableMapping[ModelId, object], config.instance_by_model_id)[ModelId("test/other")] = config.instances[0]


@pytest.mark.parametrize(
    ("atn_devices", "ffn_devices", "message"),
    [
        ([], [1], "at least 1 item"),
        ([-1], [1], "consecutive indices starting at zero"),
        ([0, 0], [1], "consecutive indices starting at zero"),
        ([1], [2], "consecutive indices starting at zero"),
        ([0, 2], [3], "consecutive indices starting at zero"),
        ([0], [7, 3], "nonnegative consecutive indices"),
        ([0], [1, 3], "nonnegative consecutive indices"),
    ],
)
def test_invalid_role_device_sequences_are_rejected(
    atn_devices: list[int],
    ffn_devices: list[int],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": atn_devices},
                "ffn": {"devices": ffn_devices},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            }
        )


@pytest.mark.parametrize("ffn_devices", [[0], [2]])
def test_role_blocks_must_be_adjacent(ffn_devices: list[int]) -> None:
    with pytest.raises(ValidationError, match="immediately follow"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": ffn_devices},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            }
        )


def test_missing_required_atn_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="atn_devices"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "ffn": {"devices": [1]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            },
            cli={},
        )


def test_missing_required_ffn_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="ffn_devices"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
            },
            cli={},
        )
