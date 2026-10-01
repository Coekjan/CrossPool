"""Resolve catalogue deployment references without runtime configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from xpool.model import ModelId


def resolve_deployment_path(catalogue_path: Path, model_ids: Sequence[ModelId], basename: str) -> Path:
    """Bind a suffix-free scene name to the catalogue's model-set directory.

    Model IDs are distinct, sorted by their complete spelling, URI-encoded as
    complete IDs and joined with '+'. The reference must be a single filename
    stem, not a path. This operation loads no runtime settings or model
    metadata and does not require the referenced file to exist.
    """

    if (
        not isinstance(basename, str)
        or not basename
        or basename in {".", ".."}
        or basename.endswith(".toml")
        or any(character in basename for character in "/\\\0")
    ):
        raise ValueError("deployment must be a suffix-free basename, without a directory or .toml extension")
    identity = "+".join(model_id.uri_encode() for model_id in sorted(set(model_ids)))
    root = catalogue_path.expanduser().resolve().parent.parent / "configs" / "deployments"
    return (root / identity / f"{basename}.toml").resolve()
