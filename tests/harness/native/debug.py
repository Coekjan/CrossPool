"""Native debug-option projection for component tests."""

from __future__ import annotations

from pathlib import Path

from xpool.config import DebugConfig, LoopbackSite

NATIVE_DEBUG_FIELDS = {
    "loopback": {"enable", "site"},
    "transport_observer": {"enable", "trace_capacity"},
    "fabric_observer": {"enable", "trace_capacity"},
}


def native_debug_options(
    *,
    loopback_site: LoopbackSite | None = None,
    transport_observer: bool = False,
    fabric_observer: bool = False,
    trace_capacity: int = 8192,
) -> str:
    """Project validated production debug settings to the native JSON boundary."""

    outdir = Path.cwd() if transport_observer or fabric_observer else None
    config = DebugConfig.model_validate(
        {
            "loopback": {
                "enable": loopback_site is not None,
                "site": loopback_site,
            },
            "transport_observer": {
                "enable": transport_observer,
                "outdir": outdir if transport_observer else None,
                "trace_capacity": trace_capacity,
            },
            "fabric_observer": {
                "enable": fabric_observer,
                "outdir": outdir if fabric_observer else None,
                "trace_capacity": trace_capacity,
            },
        }
    )
    return config.model_dump_json(include=NATIVE_DEBUG_FIELDS)
