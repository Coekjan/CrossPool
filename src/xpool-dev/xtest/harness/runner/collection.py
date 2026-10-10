"""Owner of one isolated pytest collection process."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from xkit.config import ToolConfigRecord
from xkit.results import write_json
from xtest.harness.runner.plan import TestPlan


class CollectionFailure(RuntimeError):
    """Raised when isolated pytest discovery cannot publish a valid Test Plan."""


@dataclass(frozen=True, slots=True)
class CollectionWorker:
    """Own one isolated pytest collection process and its durable diagnostics."""

    repository_root: Path
    run_directory: Path
    selectors: tuple[str, ...]
    strict_requirements: bool
    catalogue_path: Path
    tool_config: ToolConfigRecord

    def collect(self) -> TestPlan:
        """Run final pytest discovery and strictly parse its atomic output."""

        self.run_directory.mkdir(parents=True, exist_ok=False)
        plan_path = self.run_directory / "test-plan.json"
        log_path = self.run_directory / "collection.log"
        tool_config_path = self.run_directory / "tool-config.json"
        write_json(tool_config_path, self.tool_config.model_dump(mode="json"))
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            *self.selectors,
            f"--xpool-test-catalog={self.catalogue_path}",
            f"--xpool-test-plan={plan_path}",
            f"--xpool-tool-config={tool_config_path}",
        ]
        if self.selectors:
            command.append("--xpool-pytest-inputs")
        if self.strict_requirements:
            command.append("--strict-requirements")
        environment = os.environ.copy()
        try:
            completed = subprocess.run(
                command,
                cwd=self.repository_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise CollectionFailure(f"failed to start pytest Collection Worker: {error}") from error
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise CollectionFailure(f"pytest Collection Worker exited with code {completed.returncode}; see {log_path}")
        try:
            return TestPlan.read(plan_path)
        except ValueError as error:
            raise CollectionFailure(f"pytest Collection Worker published an invalid Test Plan: {error}") from error
