from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator

import pytest

from xpool.utils import device


@pytest.fixture
def inventory(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[int, str]]:
    mapping = {
        4: "GPU-00000000-0000-0000-0000-000000000004",
        2: "GPU-00000000-0000-0000-0000-000000000002",
    }
    device.resolve_visible_uuids.cache_clear()
    monkeypatch.setattr(device, "query_uuids_mapping", lambda: mapping)
    yield mapping
    device.resolve_visible_uuids.cache_clear()


def test_visibility_follows_selection_and_environment_changes(
    inventory: dict[int, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert device.visible_uuids() == (inventory[2], inventory[4])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,2")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/external/controller")
    assert device.visible_uuids() == (inventory[4], inventory[2])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,2"
    device.normalize_environment()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == f"{inventory[4]},{inventory[2]}"
    assert device.visible_uuids() == (inventory[4], inventory[2])
    assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/external/controller"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert device.visible_uuids() == ()


@pytest.mark.parametrize("selection", ["0", "-1", "MIG-invalid", "GPU-short", "2,,4", "2,2"])
def test_invalid_selection(inventory: dict[int, str], monkeypatch: pytest.MonkeyPatch, selection: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selection)
    with pytest.raises(ValueError):
        device.visible_uuids()


def test_duplicate_physical_identity(inventory: dict[int, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"2,{inventory[2]}")
    with pytest.raises(ValueError, match="duplicate physical"):
        device.visible_uuids()


def test_inventory_preserves_reported_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def query(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 0, "4, GPU-00000000-0000-0000-0000-000000000004\n2, GPU-00000000-0000-0000-0000-000000000002\n", ""
        )

    monkeypatch.setattr(device.subprocess, "run", query)
    assert device.query_uuids_mapping() == {
        4: "GPU-00000000-0000-0000-0000-000000000004",
        2: "GPU-00000000-0000-0000-0000-000000000002",
    }
