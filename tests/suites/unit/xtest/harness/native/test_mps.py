from __future__ import annotations

import pytest

import xtest.harness.native.mps
from xpool.utils.mps import MpsEndpoint


def test_mps_query_preserves_server_clients_percentage_and_owner_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        "get_server_list": "20\n",
        "get_client_list 20": "10\n11\n",
        "get_active_thread_percentage 20": "50.0\n",
    }

    def run(self: MpsEndpoint, command: str, *, deadline: float) -> str:
        assert deadline == 1002.0
        return outputs[command].strip()

    monkeypatch.setattr(MpsEndpoint, "run_control", run)

    assert xtest.harness.native.mps.query_mps_servers(
        MpsEndpoint(("GPU-00000000-0000-0000-0000-000000000001",)), deadline=1002.0
    ) == (
        xtest.harness.native.mps.MpsServerObservation(
            process_id=20,
            client_process_ids=(10, 11),
            active_thread_percentage=50.0,
        ),
    )
