from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from xpool.config import MissingRequiredConfig, XpoolConfig


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
    assert config.models[0].id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path(
        "/absolute/path/to/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )


def test_derived_placements_follow_declared_order() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [2, 4]},
            "ffn": {"devices": [3, 7, 10]},
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


def test_latency_slo_uses_global_default_and_complete_model_override() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [
                {"id": "m1", "path": "/models/m1"},
                {"id": "m2", "path": "/models/m2", "slo": {"ttft_ms": 800, "tbt_ms": 40}},
            ],
        }
    )

    assert config.models[0].slo is None
    assert (config.scheduler.slo.ttft_ms, config.scheduler.slo.tbt_ms) == (1000, 50)
    assert config.models[1].slo is not None
    assert (config.models[1].slo.ttft_ms, config.models[1].slo.tbt_ms) == (800, 40)


@pytest.mark.parametrize(
    "slo",
    [
        {"ttft_ms": 1000},
        {"ttft_ms": 0, "tbt_ms": 50},
        {"ttft_ms": 1000, "tbt_ms": float("inf")},
        {"ttft_ms": 1000, "tbt_ms": 50, "unknown": 1},
    ],
)
def test_latency_slo_rejects_incomplete_nonpositive_nonfinite_and_extra_fields(slo: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": slo},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )


def test_scheduler_latency_slo_is_required() -> None:
    with pytest.raises(MissingRequiredConfig, match="scheduler_slo"):
        XpoolConfig.from_mapping(
            {
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1]},
                "models": [{"id": "m", "path": "/models/m"}],
                "unexpected": True,
            }
        )


def test_derived_placement_views_are_immutable() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    with pytest.raises(AttributeError):
        cast(MutableSequence[object], config.atnagents).append(config.atnagents[0])
    with pytest.raises(TypeError):
        cast(MutableMapping[int, object], config.atnagent_by_cuda_device)[2] = config.atnagents[0]
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], config.instance_by_id)["other"] = config.instances[0]


@pytest.mark.parametrize(
    ("atn_devices", "ffn_devices", "message"),
    [
        ([], [1], "at least 1 item"),
        ([-1], [1], "non-negative CUDA device indices"),
        ([0, 0], [1], "devices must be unique"),
        ([0], [7, 3], "devices must be sorted in ascending order"),
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
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )


def test_overlapping_role_devices_are_rejected() -> None:
    with pytest.raises(ValidationError, match="overlapping devices"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [0]},
                "models": [{"id": "m", "path": "/models/m"}],
            }
        )


def test_missing_required_atn_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="atn_devices"):
        XpoolConfig.from_mapping({"models": []}, cli={})


def test_missing_required_ffn_devices_fails() -> None:
    with pytest.raises(MissingRequiredConfig, match="ffn_devices"):
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "models": [{"id": "m", "path": "/models/m"}],
            },
            cli={},
        )
