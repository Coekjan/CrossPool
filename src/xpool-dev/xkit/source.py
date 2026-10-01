"""Catalogue-relative source references shared by tooling collection adapters."""

import keyword
from pathlib import Path

__all__ = ["resolve_source_path"]


def resolve_source_path(catalogue_path: Path, module: str) -> Path:
    """Resolve a dotted module beside its catalogue under the sibling suites tree.

    The reference contains Python identifier components, not a filesystem path.
    Resolution requires neither importing the module nor an existing source file;
    the consuming collection adapter owns import and entry validation.
    """

    parts = module.split(".")
    if any(not part.isidentifier() or keyword.iskeyword(part) for part in parts):
        raise ValueError("source module must be a dotted Python module name")
    return (catalogue_path.expanduser().resolve().parent / "suites" / Path(*parts).with_suffix(".py")).resolve()
