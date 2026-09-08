"""Own published SGLang configuration for component tests."""

from collections.abc import Iterator

import pytest
from sglang.srt.runtime_context import get_context

from tests.harness.support.sglang.fakes import server_args


@pytest.fixture
def published_sglang_config() -> Iterator[None]:
    """Publish a concrete baseline and restore the previous upstream context."""

    args = server_args()
    with get_context().override_server_args(cuda_graph_config=args.cuda_graph_config):
        yield
