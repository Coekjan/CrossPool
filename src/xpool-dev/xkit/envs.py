"""Raw environment declarations for development process protocols."""

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

__all__ = ["TASK_ROOT_CONTROL_FD", "EnvironmentVariable"]


@dataclass(frozen=True, slots=True)
class EnvironmentVariable:
    """Immutable name with access to current or explicitly supplied environments.

    Values remain raw strings; protocol owners perform parsing and validation.
    An omitted environment selects ``os.environ``, while an explicit mapping,
    including an empty one, is accessed exclusively.
    """

    name: str

    def get(self, *, env: Mapping[str, str] | None = None) -> str | None:
        """Read without mutation, returning ``None`` only when absent."""

        return (os.environ if env is None else env).get(self.name)

    def set(self, value: str, *, env: MutableMapping[str, str] | None = None) -> None:
        """Write the supplied string, preserving empty values."""

        environment = os.environ if env is None else env
        environment[self.name] = value

    def clear(self, *, env: MutableMapping[str, str] | None = None) -> None:
        """Remove the entry if present, leaving other values unchanged."""

        environment = os.environ if env is None else env
        environment.pop(self.name, None)

    def pop(self, *, env: MutableMapping[str, str] | None = None) -> str | None:
        """Consume the entry, returning ``None`` only when absent."""

        return (os.environ if env is None else env).pop(self.name, None)


TASK_ROOT_CONTROL_FD = EnvironmentVariable("XKIT_TASK_ROOT_CONTROL_FD")
