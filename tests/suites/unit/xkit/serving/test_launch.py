from pathlib import Path

from xkit.serving.launch import snapshot_cluster_launch
from xpool.config import XpoolConfig


def test_snapshot_preserves_complete_settings_and_freezes_environment(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    environment = {
        "XPOOL_LOG_LEVEL": "error",
        "XPOOL_FFN_LOADER_PARALLELISM": "7",
        "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
        "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(tmp_path / "debug"),
        "LD_LIBRARY_PATH": "/runtime/lib",
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
        "NCCL_DEBUG": "INFO",
        "SGLANG_PLUGINS": "unrelated-plugin",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
    }
    original_environment = environment.copy()
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"port": 19999},
            "vendor": {"model_base_uri": str(tmp_path / "models")},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1], "device_memory_calibration": str(calibration)},
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}, "ffn_concurrency": 2},
            "models": [
                {"id": "test/a", "path": str(tmp_path / "a"), "ffn_tp_size": 1, "slo": {"ttft_ms": 200, "tbt_ms": 10}},
                {"id": "test/b"},
            ],
        },
        env=environment,
    )
    launch = snapshot_cluster_launch(
        config, workdir=tmp_path / "launch", daemon_port=19810, environment=environment, cwd=tmp_path
    )
    expected = config.model_dump(mode="json")
    expected["daemon"]["port"] = 19810
    assert launch.config.model_dump(mode="json") == expected
    assert launch.config.sources != config.sources
    assert "XPOOL_LOG_LEVEL" not in launch.environment
    assert "XPOOL_FFN_LOADER_PARALLELISM" not in launch.environment
    assert launch.config.logging.level == "error"
    assert launch.config.ffn.loader.parallelism == 7
    assert launch.environment["SGLANG_PLUGINS"] == "unrelated-plugin"
    assert launch.environment["HF_HUB_OFFLINE"] == "0"
    assert launch.environment["TRANSFORMERS_OFFLINE"] == "0"
    assert launch.environment["LD_LIBRARY_PATH"] == "/runtime/lib"
    assert launch.environment["NCCL_DEBUG"] == "INFO"
    assert launch.config.debug.graph_observer.enable
    assert launch.cwd == tmp_path
    assert environment == original_environment
