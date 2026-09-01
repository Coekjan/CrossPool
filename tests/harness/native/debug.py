"""Native debug-option projection for component tests."""

from __future__ import annotations

from pathlib import Path

import xpool.native
from xpool.config import DebugConfig


def native_debug_options(
    *,
    transport_observer: bool = False,
    fabric_observer: bool = False,
    graph_observer: bool = False,
    ffn_routing_observer: bool = False,
    record_capacity: int = 8192,
    routing_record_capacity: int = 8,
) -> xpool.native.debug.Options:
    """Project validated production debug settings to native values."""

    outdir = Path.cwd() if transport_observer or fabric_observer or graph_observer or ffn_routing_observer else None
    config = DebugConfig.model_validate(
        {
            "transport_observer": {
                "enable": transport_observer,
                "outdir": outdir if transport_observer else None,
                "record_capacity": record_capacity,
            },
            "fabric_observer": {
                "enable": fabric_observer,
                "outdir": outdir if fabric_observer else None,
                "record_capacity": record_capacity,
            },
            "graph_observer": {
                "enable": graph_observer,
                "outdir": outdir if graph_observer else None,
            },
            "ffn_routing_observer": {
                "enable": ffn_routing_observer,
                "outdir": outdir if ffn_routing_observer else None,
                "record_capacity": routing_record_capacity,
            },
        }
    )
    return config.native_options()
