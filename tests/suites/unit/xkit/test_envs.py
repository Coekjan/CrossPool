import os
from types import MappingProxyType

import pytest

from xkit.envs import EnvironmentVariable


@pytest.mark.parametrize("value", ["", "raw value"])
def test_explicit_environment_access_isolated_from_parent(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    variable = EnvironmentVariable("XKIT_TEST_VALUE")
    monkeypatch.setenv(variable.name, "parent")
    environment: dict[str, str] = {}

    assert variable.get(env=environment) is None
    assert variable.pop(env=environment) is None
    variable.clear(env=environment)
    variable.set(value, env=environment)
    assert variable.get(env=MappingProxyType(environment)) == value
    assert variable.pop(env=environment) == value
    assert variable.get(env=environment) is None
    variable.set(value, env=environment)
    variable.clear(env=environment)
    assert environment == {}
    assert os.environ[variable.name] == "parent"


def test_current_environment_access_reads_live_values(monkeypatch: pytest.MonkeyPatch) -> None:
    variable = EnvironmentVariable("XKIT_TEST_VALUE")
    monkeypatch.delenv(variable.name, raising=False)

    assert variable.get() is None
    assert variable.pop() is None
    variable.clear()
    variable.set("first")
    assert os.environ[variable.name] == "first"
    monkeypatch.setenv(variable.name, "second")
    assert variable.get() == "second"
    assert variable.pop() == "second"
    variable.set("")
    assert variable.get() == ""
    variable.clear()
    assert variable.name not in os.environ
