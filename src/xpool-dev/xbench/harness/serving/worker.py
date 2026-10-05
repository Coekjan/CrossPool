"""Invoke one collected serving program inside its runner-owned supervision scope."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from xbench.harness.serving.case import BenchCase
from xbench.harness.serving.measure import BenchCaseManifest
from xkit.task import TaskRoot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--import-root", action="append", type=Path, default=[])
    options = parser.parse_args()
    sys.path[:0] = [str(path) for path in options.import_root]
    root = TaskRoot.from_environment()
    try:
        try:
            module = importlib.import_module(options.module)
            if module.__file__ is None or Path(module.__file__).resolve() != options.source_path:
                raise ValueError("benchmark worker imported a different source module")
            # Isolated collection validated this discovered entry and its invocation signature.
            entry = cast(Callable[..., None], getattr(module, options.entrypoint))
            manifest = BenchCaseManifest.model_validate_json((options.directory.parent / "case.json").read_bytes())
            case: BenchCase = manifest.case
            entry(case=case, workdir=options.directory)
            return 0
        except Exception as error:
            print(f"xbench program failure: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
    finally:
        if root is not None:
            root.finish()


if __name__ == "__main__":
    raise SystemExit(main())
