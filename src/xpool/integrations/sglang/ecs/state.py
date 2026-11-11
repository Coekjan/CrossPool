"""Runner ownership of the selected serving-state implementation."""

from dataclasses import dataclass

from xpool.integrations.sglang.ecs.interfaces import SglangStateAdapter


@dataclass(slots=True)
class SglangElasticContextState:
    """Retain the selected state adapter through quiesced runner teardown."""

    adapter: SglangStateAdapter

    def close(self) -> None:
        self.adapter.close()
