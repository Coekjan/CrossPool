from __future__ import annotations

from tests.harness.runtime.devagent import (
    XpoolClientError,
    XpoolConfig,
    common_module,
    create_devagent,
    health_client_class,
    pytest,
)


def test_devagent_run_rejects_unhealthy_daemon_health(monkeypatch) -> None:
    config = XpoolConfig.from_file("configs/xpool.example.toml")
    monkeypatch.setattr(common_module, "XpoolClient", health_client_class(False))

    with pytest.raises(XpoolClientError, match="daemon health check failed"):
        create_devagent(config, cuda_device=0)
