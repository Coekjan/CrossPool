"""Resolve deployment visibility using physical device inventory."""

from __future__ import annotations

import os
import subprocess
import uuid
from functools import cache

__all__ = ["normalize_environment", "query_uuids_mapping", "visible_uuids"]


def query_uuids_mapping() -> dict[int, str]:
    """Query physical nvidia-smi indices and full UUIDs, independent of visibility.

    The command has a ten-second bound and creates no CUDA context. Command
    failures, malformed inventory and duplicate identities raise RuntimeError.
    Process creation and timeout errors propagate. No result is cached.
    """

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"physical device query failed: {result.stderr.strip() or result.stdout.strip()}")
    mapping: dict[int, str] = {}
    identities: set[str] = set()
    for row in result.stdout.splitlines():
        fields = tuple(field.strip() for field in row.split(","))
        if len(fields) != 2 or not fields[0].isascii() or not fields[0].isdigit():
            raise RuntimeError(f"invalid physical device row: {row!r}")
        index, identity = int(fields[0]), fields[1]
        try:
            canonical = identity.startswith("GPU-") and identity == f"GPU-{uuid.UUID(identity[4:])}"
        except ValueError:
            canonical = False
        if not canonical or index in mapping or identity in identities:
            raise RuntimeError(f"invalid or duplicate physical device row: {row!r}")
        mapping[index] = identity
        identities.add(identity)
    if not mapping:
        raise RuntimeError("physical device query returned no devices")
    return mapping


def visible_uuids() -> tuple[str, ...]:
    """Return selected physical UUIDs in deployment order without changing env.

    Numeric selectors are nvidia-smi indices. Unset visibility selects all
    devices by ascending index; empty visibility selects none. Unknown selectors
    and duplicate physical selections raise ValueError. Successful resolutions
    are cached by the raw environment value, assuming stable physical inventory
    throughout the process lifetime.
    """

    return resolve_visible_uuids(os.environ.get("CUDA_VISIBLE_DEVICES"))


@cache
def resolve_visible_uuids(selection: str | None) -> tuple[str, ...]:
    if selection is not None and not selection.strip():
        return ()
    mapping = query_uuids_mapping()
    if selection is None:
        return tuple(mapping[index] for index in sorted(mapping))
    identities = set(mapping.values())
    resolved = []
    for selector in selection.split(","):
        selector = selector.strip()
        if selector.isascii() and selector.isdigit():
            identity = mapping.get(int(selector))
        else:
            identity = selector if selector in identities else None
        if identity is None:
            raise ValueError(f"unknown physical device selector: {selector!r}; use an index or full physical UUID")
        resolved.append(identity)
    if len(resolved) != len(set(resolved)):
        raise ValueError("device visibility contains duplicate physical devices")
    return tuple(resolved)


def normalize_environment() -> None:
    """Install the complete ordered UUID selection into CUDA_VISIBLE_DEVICES.

    Preserves selection order and all unrelated environment fields. Discovery
    and selector errors propagate before changing the environment.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_uuids())
