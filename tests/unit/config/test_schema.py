from __future__ import annotations

from collections.abc import MutableMapping

from pydantic import ValidationError

from tests.harness.config import (
    Path,
    pytest,
)
from xpool.config import DeviceRole, MissingRequiredConfig, XpoolConfig


def test_example_config_loads_schema_only() -> None:
    config = XpoolConfig.from_file(Path("configs/xpool.example.toml"))

    assert config.debug.shim_loopback.enable is False
    assert config.debug.transport_loopback.enable is False
    assert config.debug.graph_observer.enable is False
    assert config.debug.graph_observer.outdir is None
    assert config.scheduler.atn_concurrency == 1
    assert config.scheduler.ffn_concurrency == 1
    assert config.vendor.model_base_uri == Path("/absolute/path/to/models")
    assert config.devices.atn_cuda_devices == [0]
    assert config.devices.ffn_cuda_devices == [1]
    assert config.cuda_devices == (0, 1)
    assert [(agent.cuda_device, agent.role) for agent in config.devagents] == [
        (0, DeviceRole.ATN),
        (1, DeviceRole.FFN),
    ]
    assert config.models[0].id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert config.models[0].path is None
    assert config.model_path_of("deepseek-ai/DeepSeek-V2-Lite-Chat") == Path(
        "/absolute/path/to/models/deepseek-ai/DeepSeek-V2-Lite-Chat"
    )
    assert config.instances[0].id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
    assert config.instances[0].instance_index == 0


def test_instance_schema_does_not_duplicate_global_devices() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [2, 4], "ffn_cuda_devices": [6, 8, 10]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    instance = config.instances[0]
    payload = instance.model_dump(mode="json")

    assert payload == {"id": "m", "instance_index": 0}
    assert config.devices.atn_cuda_devices == [2, 4]
    assert config.devices.ffn_cuda_devices == [6, 8, 10]
    assert "atn_cuda_devices" not in payload
    assert "ffn_cuda_devices" not in payload
    assert "atn_world_size" not in payload
    assert "ffn_world_size" not in payload
    assert "ffn_tp_size" not in payload


def test_old_explicit_topology_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "devagents": [{"id": "gpu0", "cuda_device": 0, "nvshmem_rank": 0, "roles": ["atn"]}],
                "models": [{"id": "m", "path": "/models/m"}],
                "sglang_instances": [{"id": "m", "model": "m"}],
            },
            cli={},
        )


def test_devagents_follow_sorted_cuda_devices_and_roles_from_lists() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [2, 4], "ffn_cuda_devices": [3, 7]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    assert [(agent.cuda_device, agent.role) for agent in config.devagents] == [
        (2, DeviceRole.ATN),
        (3, DeviceRole.FFN),
        (4, DeviceRole.ATN),
        (7, DeviceRole.FFN),
    ]
    assert config.cuda_devices == (2, 3, 4, 7)
    assert "roles" not in config.devagents[0].model_dump(mode="json")
    assert config.devices.atn_cuda_devices == [2, 4]


def test_derived_config_views_are_cached_and_read_only() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    assert isinstance(config.cuda_devices, tuple)
    assert isinstance(config.devagents, tuple)
    assert isinstance(config.instances, tuple)
    assert not isinstance(config.devagent_by_cuda_device, MutableMapping)
    assert not isinstance(config.instance_by_id, MutableMapping)
    assert "devagents" not in config.model_dump(mode="json")
    assert "instances" not in config.model_dump(mode="json")


def test_model_metadata_fields_are_rejected_from_toml() -> None:
    with pytest.raises(ValidationError):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [
                    {
                        "id": "m",
                        "path": "/models/m",
                        "dtype": "bfloat16",
                        "family": "dense",
                        "hidden_size": 1,
                    }
                ],
            },
            cli={},
        )


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


def test_core_config_allows_non_arithmetic_sorted_atn_devices() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 2, 3], "ffn_cuda_devices": [7]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    assert config.devices.atn_cuda_devices == [0, 2, 3]
    assert config.instances[0].id == "m"


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
