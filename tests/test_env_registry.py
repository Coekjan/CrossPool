from __future__ import annotations

import re
from pathlib import Path


def test_xpool_source_env_refs_are_bootstrap_only() -> None:
    referenced: set[str] = set()
    for path in Path("src/xpool").rglob("*.py"):
        referenced.update(re.findall(r'"(XPOOL_[A-Z0-9_]+)"', path.read_text()))

    assert referenced == {"XPOOL_CONFIG"}


def test_env_example_contains_only_process_bootstrap_vars() -> None:
    referenced = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", Path(".env.example").read_text(), flags=re.MULTILINE))

    assert referenced == {"XPOOL_CONFIG", "SGLANG_PLUGINS"}
