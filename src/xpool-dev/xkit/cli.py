"""Shared argument parsing and run-retention presentation for development tools."""

from __future__ import annotations

import argparse

from xkit.results import RunCleanup

__all__ = ["positive_integer", "print_cleanup"]


def positive_integer(value: str) -> int:
    """Parse an integer above zero, leaving usage errors to argparse."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def print_cleanup(cleanup: RunCleanup) -> None:
    """Print planned removals, retained entries and active entries in that order."""

    for action, paths in (("remove", cleanup.removable), ("keep", cleanup.retained), ("active", cleanup.active)):
        for path in paths:
            print(f"{action} {path}")
