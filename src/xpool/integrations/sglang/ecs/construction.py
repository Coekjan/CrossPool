"""Lend the model-selected state factory to the original pool construction."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from xpool.integrations.sglang.ecs.interfaces import SglangStateAdapter

state_adapter_type_context = ContextVar[type[SglangStateAdapter] | None]("xpool_state_adapter_type", default=None)


@contextmanager
def state_adapter_scope(adapter_type: type[SglangStateAdapter]) -> Iterator[None]:
    token = state_adapter_type_context.set(adapter_type)
    try:
        yield
    finally:
        state_adapter_type_context.reset(token)


def current_state_adapter_type() -> type[SglangStateAdapter]:
    adapter_type = state_adapter_type_context.get()
    if adapter_type is None:
        raise RuntimeError("xpool state construction requires a model-selected implementation")
    return adapter_type
