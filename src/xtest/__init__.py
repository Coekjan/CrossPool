"""Installed entry point for the source-checkout xpool test command."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the source-owned test CLI from an xpool repository root.

    The command returns exit code 2 without importing test code when the
    current directory is not an xpool source checkout. A valid checkout is
    prepended to the process import path before delegation.
    """

    repository_root = Path.cwd()
    if not (repository_root / "pyproject.toml").is_file() or not (repository_root / "tests" / "cli.py").is_file():
        print("xtest must run from the xpool repository root", file=sys.stderr)
        return 2
    sys.path.insert(0, str(repository_root))

    # The test package intentionally remains source-owned rather than shipped
    # in the runtime wheel, so it becomes importable only after root validation.
    import tests.cli

    return tests.cli.main(None if arguments is None else list(arguments))
