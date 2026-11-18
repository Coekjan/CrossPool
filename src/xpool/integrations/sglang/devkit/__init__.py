"""Debug-only SGLang integration observers."""

from typing import Literal, TypedDict

from xpool.model import ModelId

__all__ = ("SglangGraphEvent",)


class SglangGraphEvent(TypedDict):
    """One process-local SGLang graph capture or execution event."""

    pid: int
    model_id: ModelId
    time_ns: int
    forward_phase: Literal["decode", "prefill"]
    backend_class: str
    event: str
