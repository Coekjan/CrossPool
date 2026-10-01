from pathlib import Path

import pytest

from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xpool.config import XpoolConfig
from xpool.model import ModelId


def test_launch_requires_exact_instance_coverage(tmp_path: Path) -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": str(tmp_path)},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "models": [{"id": "test/a"}, {"id": "test/b"}],
        },
        env={},
    )
    a = SglangLaunchModel(ModelId("test/a"), SglangGraphMode.EAGER)
    b = SglangLaunchModel(ModelId("test/b"), SglangGraphMode.EAGER)
    assert ServingLaunch(config, {}, tmp_path, (b, a)).models == (b, a)
    for models in (
        (a,),
        (a, a),
        (a, SglangLaunchModel(ModelId("test/c"), SglangGraphMode.EAGER)),
    ):
        with pytest.raises(ValueError, match="all runtime"):
            ServingLaunch(config, {}, tmp_path, models)
