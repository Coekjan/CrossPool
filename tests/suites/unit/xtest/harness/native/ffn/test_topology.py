"""Task-local config behavior for installed FFN qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from xpool.config import XpoolConfig
from xpool.model import ModelId
from xtest.harness.native.ffn.topology import ffn_cluster_environment, materialize_ffn_cluster_launch


def test_materialization_freezes_selected_topology_and_model_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGNAME", "test-user")
    monkeypatch.setenv("USER", "test-user")
    model_root = tmp_path / "models"
    model_root.mkdir()
    workdir = tmp_path / "run"
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19000},
            "vendor": {"model_base_uri": str(model_root)},
            "scheduler": {
                "slo": {"ttft_ms": 1000, "tbt_ms": 50},
                "ffn_concurrency": 2,
                "ffn_policy": "random",
                "ffn_random_seed": 17,
            },
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2, 3, 4, 5], "loader": {"parallelism": 2}},
            "models": [
                {"id": "test/first", "path": str(tmp_path / "custom"), "ffn_tp_size": 4},
                {"id": "test/second", "ffn_tp_size": 2},
            ],
        },
        env=ffn_cluster_environment(config_path=workdir / "xpool.toml", observer_outdir=workdir / "observers"),
    )

    launch = materialize_ffn_cluster_launch(
        config=config,
        daemon_port=19001,
        workdir=workdir,
    )

    assert launch.config.atn.devices == [0, 1]
    assert launch.config.ffn.devices == [2, 3, 4, 5]
    assert tuple(model.ffn_tp_size for model in launch.config.models) == (4, 2)
    assert launch.config.scheduler.ffn_random_seed == 17
    assert launch.config.ffn.loader.parallelism == 2
    assert launch.config.model_path_of(ModelId("test/first")) == tmp_path / "custom"
    assert launch.config.model_path_of(ModelId("test/second")) == model_root / "test/second"
    assert launch.config.debug.graph_observer.enable and launch.config.debug.fabric_observer.enable
    assert "SGLANG_PLUGINS" not in launch.environment
    assert launch.environment["LOGNAME"] == "test-user"
    assert launch.environment["USER"] == "test-user"
