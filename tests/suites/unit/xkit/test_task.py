import pytest

import xkit.task
from xkit.envs import TASK_ROOT_CONTROL_FD
from xkit.task import TaskRoot


def test_direct_invocation_has_no_task_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TASK_ROOT_CONTROL_FD.name, raising=False)

    assert TaskRoot.from_environment() is None


def test_invalid_descriptor_metadata_is_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xkit.task, "current_task_root", None)
    monkeypatch.setenv(TASK_ROOT_CONTROL_FD.name, "not-a-descriptor")

    with pytest.raises(ValueError):
        TaskRoot.from_environment()

    assert TASK_ROOT_CONTROL_FD.get() is None
