from __future__ import annotations

import subprocess

import pytest

import tests.harness.native.mps


def test_mps_query_preserves_server_clients_and_percentage(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        "get_server_list\n": "20\n",
        "get_client_list 20\n": "10\n11\n",
        "get_active_thread_percentage 20\n": "50.0\n",
    }

    def run(command: list[str], *, input: str, **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, outputs[input], "")

    monkeypatch.setattr(subprocess, "run", run)

    assert tests.harness.native.mps.query_mps_servers() == (
        tests.harness.native.mps.MpsServerObservation(
            process_id=20,
            client_process_ids=(10, 11),
            active_thread_percentage=50.0,
        ),
    )
