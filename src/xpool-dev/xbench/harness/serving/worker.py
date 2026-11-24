"""Execute one serving benchmark attempt under runner-owned supervision."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from xbench.harness.serving.execution import run_serving
from xbench.harness.serving.measure import BenchCaseManifest
from xkit.task import TaskRoot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    root = TaskRoot.from_environment()
    failure: Exception | None = None
    try:
        manifest = BenchCaseManifest.model_validate_json((options.directory.parent / "case.json").read_bytes())
        asyncio.run(run_serving(manifest, options.directory))
    except Exception as error:
        failure = error
        print(f"xbench attempt failure: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        if root is not None:
            # Wait for cleanup confirmation even if the benchmark failed.
            root.finish()
    return 0 if failure is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
