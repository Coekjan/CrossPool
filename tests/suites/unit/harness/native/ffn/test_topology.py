"""Task-local config behavior for installed FFN qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.native.ffn.topology import materialize_ffn_cluster_launch
from xpool.config import XpoolConfig
from xpool.fabric import RandomSchedulerPolicy


def test_materialization_installs_exact_topology_without_integration_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGNAME", "test-user")
    monkeypatch.setenv("USER", "test-user")
    model_root = tmp_path / "models"
    model_root.mkdir()
    base_config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19000},
            "vendor": {"model_base_uri": str(model_root)},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "base"}],
        },
        env={},
    )

    launch = materialize_ffn_cluster_launch(
        base_config=base_config,
        model_tp_sizes=(("first", 4), ("second", 2)),
        atnagent_count=2,
        ffnagent_count=4,
        executor_lane_count=2,
        scheduler=RandomSchedulerPolicy(seed=17),
        daemon_port=19001,
        workdir=tmp_path / "run",
    )

    assert launch.config.devices.atn_cuda_devices == [0, 1]
    assert launch.config.devices.ffn_cuda_devices == [2, 3, 4, 5]
    assert tuple(model.ffn_tp_size for model in launch.config.models) == (4, 2)
    assert launch.config.scheduler.ffn_random_seed == 17
    assert "SGLANG_PLUGINS" not in launch.environment
    assert launch.environment["LOGNAME"] == "test-user"
    assert launch.environment["USER"] == "test-user"
