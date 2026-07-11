"""FFN-side devagent lifecycle."""

from __future__ import annotations

from xpool.runtime.devagent.common import Devagent

__all__ = ["FfnDevagent"]


class FfnDevagent(Devagent):
    """FFN-side devagent placeholder until the FFN executor runtime exists."""

    def run(self) -> None:
        """Fail fast because the FFN-side devagent runtime is not implemented."""

        raise NotImplementedError("FFN devagent runtime is not implemented yet")
