from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from xpool.config import MissingRequiredConfig, XpoolConfig


def test_example_config_defines_documented_topology() -> None:
    config = XpoolConfig.from_file(Path("configs/xpool.example.toml"))

    assert config.devices.atn_cuda_devices == [0]
    assert config.devices.ffn_cuda_devices == [1]
    assert config.models[0].id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path(
        "/absolute/path/to/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )


def test_derived_placements_follow_declared_order() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [2, 4], "ffn_cuda_devices": [3, 7, 10]},
            "models": [
                {"id": "m1", "path": "/models/m1"},
                {"id": "m2", "path": "/models/m2"},
            ],
        }
    )

    assert [(agent.rank, agent.cuda_device) for agent in config.atnagents] == [(0, 2), (1, 4)]
    assert [(agent.rank, agent.cuda_device) for agent in config.ffnagents] == [(0, 3), (1, 7), (2, 10)]
    assert [(instance.instance_index, instance.id) for instance in config.instances] == [(0, "m1"), (1, "m2")]
    assert config.atnagent_by_cuda_device[4].rank == 1
    assert config.ffnagent_by_cuda_device[7].rank == 1
    assert config.instance_by_id["m2"].instance_index == 1


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
                "unexpected": True,
            }
        )


def test_derived_placement_views_are_immutable() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    with pytest.raises(AttributeError):
        cast(MutableSequence[object], config.atnagents).append(config.atnagents[0])
    with pytest.raises(TypeError):
        cast(MutableMapping[int, object], config.atnagent_by_cuda_device)[2] = config.atnagents[0]
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], config.instance_by_id)["other"] = config.instances[0]


def test_duplicate_and_overlapping_devices_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"devices\.atn_cuda_devices must be unique"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0, 0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli={},
        )

    with pytest.raises(ValidationError, match="overlapping devices"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [0]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli={},
        )

    with pytest.raises(ValidationError, match=r"devices\.atn_cuda_devices must be sorted"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [4, 2], "ffn_cuda_devices": [7]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli={},
        )

    with pytest.raises(ValidationError, match=r"devices\.ffn_cuda_devices must be sorted"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [7, 3]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli={},
        )


def test_missing_required_top_level_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="devices"):
        XpoolConfig.from_mapping({"models": []}, cli={})


def test_missing_required_nested_field_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="ffn_cuda_devices"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli={},
        )
