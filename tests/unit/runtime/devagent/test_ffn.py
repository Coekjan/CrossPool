from __future__ import annotations

from tests.harness.runtime.devagent import (
    XpoolConfig,
    create_devagent,
    pytest,
)


def test_ffn_devagent_fails_fast_until_runtime_exists() -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0, 1], "ffn_cuda_devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )

    with pytest.raises(NotImplementedError, match="FFN devagent runtime is not implemented"):
        create_devagent(config, cuda_device=2).run()
