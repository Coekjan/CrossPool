"""Pure function metadata consumed by the test and benchmark collection adapters."""

from __future__ import annotations

import keyword
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from xkit.requirements import ResourceRequirements
from xpool.model import ModelId

__all__ = ["Parameterization", "parameterize", "requirements"]


@dataclass(frozen=True, slots=True)
class Parameterization:
    """Declared parameter inputs before the consuming tool expands concrete rows.

    Absent values bind the function to its catalogue-assigned cases. A rows
    callback projects each input into tool-native parameter rows. Inputs and
    callbacks are heterogeneous source-program values; the owning collection
    adapter supplies their concrete types and execution policy.
    """

    argument_names: tuple[str, ...]
    values: tuple[object, ...] | None
    rows: Callable[..., Iterable[object]] | None


def parameterize[F: Callable[..., object]](
    argument_names: str | tuple[str, ...],
    values: Iterable[object] | None = None,
    *,
    rows: Callable[..., Iterable[object]] | None = None,
) -> Callable[[F], F]:
    """Declare parameters without changing the function or performing catalogue I/O.

    Values are explicit Python inputs; omission binds catalogue cases. A rows
    callback receives each input and returns tool-native rows. The adapter owns
    expansion and fixture handling. Decorators preserve the original callable
    signature and retain metadata on its function attributes.
    """

    names = (
        tuple(name.strip() for name in argument_names.split(",")) if isinstance(argument_names, str) else argument_names
    )
    if (
        not names
        or len(names) != len(set(names))
        or any(not name.isidentifier() or keyword.iskeyword(name) for name in names)
    ):
        raise ValueError("parameter names must be distinct Python identifiers")
    if values is None and "case" not in names:
        raise ValueError("catalogue-bound parameters must include case")
    declaration = Parameterization(names, None if values is None else tuple(values), rows)

    def decorate(function: F) -> F:
        if hasattr(function, "xpool_parameters"):
            raise ValueError("a function must have one parameter declaration")
        function.__dict__["xpool_parameters"] = declaration
        return function

    return decorate


def requirements[F: Callable[..., object]](
    callback: Callable[..., ResourceRequirements] | None = None,
    *,
    device_count: int = 0,
    requires_config: bool = False,
    model_ids: tuple[ModelId, ...] = (),
) -> Callable[[F], F]:
    """Declare static resources or derive them from concrete named parameter values.

    A callback receives all declared values in the collected row as keyword
    arguments, never fixtures or execution directories. It must return
    ResourceRequirements. Registration neither probes resources nor installs
    configuration, and preserves the decorated function's execution signature.
    """

    if callback is not None and (device_count or requires_config or model_ids):
        raise ValueError("requirements must use either a callback or static fields")
    declaration = callback if callback is not None else ResourceRequirements(device_count, requires_config, model_ids)

    def decorate(function: F) -> F:
        if hasattr(function, "xpool_requirements"):
            raise ValueError("a function must have one resource declaration")
        function.__dict__["xpool_requirements"] = declaration
        return function

    return decorate
