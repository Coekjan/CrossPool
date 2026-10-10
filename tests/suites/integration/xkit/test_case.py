import shutil
import tomllib
from pathlib import Path
from typing import Literal

import pytest

from xkit.case import CaseFamily, CaseId
from xtest.harness.sglang.catalog import TestCatalog


@pytest.mark.parametrize(
    ("tool", "family"),
    [("xtest", CaseFamily.SERVING), ("xtest", CaseFamily.TOPOLOGY)],
)
def test_catalogue_clone_preserves_source_text_paths_and_published_line_range(
    tmp_path: Path, tool: Literal["xtest"], family: CaseFamily, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path.cwd()
    shutil.copytree(root / "configs/deployments", tmp_path / "configs/deployments")
    relative = Path("tests/tests.toml" if tool == "xtest" else "benches/benches.toml")
    path = tmp_path / relative
    path.parent.mkdir()
    original = ("# Author-owned comment.\n" + (root / relative).read_text(encoding="utf-8")).replace("\n", "\r\n")
    path.write_bytes(original.encode("utf-8"))
    source = next(iter(tomllib.loads(original)[family.table_name]))
    catalog = TestCatalog.from_file(path)
    location = catalog.clone(source[:8], family=family)
    published = path.read_bytes().decode("utf-8")
    assert published.startswith(original)
    assert location.path == path and location.id != CaseId(source)
    raw = tomllib.loads(published)[family.table_name]
    assert raw[str(location.id)] == raw[source]
    lines = published.splitlines()[location.first_line - 1 : location.last_line]
    assert tomllib.loads("\n".join(lines))[family.table_name][str(location.id)] == raw[source]
    other_family = CaseFamily.TOPOLOGY if family is CaseFamily.SERVING else CaseFamily.SERVING
    with pytest.raises(ValueError, match="does not belong"):
        catalog.clone(source[:8], family=other_family)
    assert path.read_bytes().decode("utf-8") == published

    def reject_candidate(candidate: Path) -> None:
        raise ValueError("candidate does not satisfy the catalogue contract")

    monkeypatch.setattr(type(catalog), "from_file", reject_candidate)
    with pytest.raises(ValueError, match="catalogue contract"):
        catalog.clone(source[:8], family=family)
    assert path.read_bytes().decode("utf-8") == published
