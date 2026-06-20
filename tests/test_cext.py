from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from xpool import cext as cext_module
from xpool.abi import ABI_VERSION
from xpool.cext import NativeLoadError, ensure_xpool_ops_loaded


@pytest.fixture(autouse=True)
def reset_cext_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cext_module, "_loaded", False)
    monkeypatch.setattr(cext_module, "import_module", lambda _name: None)


def test_native_loader_is_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_path = tmp_path / "libxpool_cext.so"
    library_path.write_bytes(b"")
    events: list[str] = []
    events_lock = Lock()

    def fake_load_library(_path: str) -> None:
        with events_lock:
            events.append("load")

    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: _FakeAsFileContext(library_path))
    monkeypatch.setattr(torch.ops, "load_library", fake_load_library)
    monkeypatch.setattr(torch.ops, "xpool", SimpleNamespace(abi_version=lambda: ABI_VERSION), raising=False)

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(lambda _index: ensure_xpool_ops_loaded(), range(32)))

    assert events == ["load"]


def test_native_loader_checks_native_abi_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_path = tmp_path / "libxpool_cext.so"
    library_path.write_bytes(b"")
    events: list[str] = []

    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: _FakeAsFileContext(library_path))
    monkeypatch.setattr(torch.ops, "load_library", lambda _path: events.append("load"))
    monkeypatch.setattr(torch.ops, "xpool", _FakeOpNamespace(events, abi_version=ABI_VERSION), raising=False)

    ensure_xpool_ops_loaded()
    ensure_xpool_ops_loaded()

    assert events == ["load", "preflight"]


def test_native_loader_prewarms_python_graph_ops_after_native_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = tmp_path / "libxpool_cext.so"
    library_path.write_bytes(b"")
    events: list[str] = []

    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: _FakeAsFileContext(library_path))
    monkeypatch.setattr(torch.ops, "load_library", lambda _path: events.append("load"))
    monkeypatch.setattr(torch.ops, "xpool", _FakeOpNamespace(events, abi_version=ABI_VERSION), raising=False)
    monkeypatch.setattr(cext_module, "import_module", lambda name: events.append(f"import:{name}"))

    ensure_xpool_ops_loaded()
    ensure_xpool_ops_loaded()

    assert events == ["load", "preflight", "import:xpool.ops"]


def test_native_loader_rejects_python_graph_wrapper_import_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = tmp_path / "libxpool_cext.so"
    library_path.write_bytes(b"")

    def fail_import(_name: str) -> object:
        raise RuntimeError("graph wrapper import failed")

    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: _FakeAsFileContext(library_path))
    monkeypatch.setattr(torch.ops, "load_library", lambda _path: None)
    monkeypatch.setattr(torch.ops, "xpool", SimpleNamespace(abi_version=lambda: ABI_VERSION), raising=False)
    monkeypatch.setattr(cext_module, "import_module", fail_import)

    with pytest.raises(NativeLoadError, match="Python graph wrapper ops"):
        ensure_xpool_ops_loaded()

    assert cext_module._loaded is False


def test_native_loader_loads_package_resource_while_as_file_context_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_path = tmp_path / "libxpool_cext.so"
    library_path.write_bytes(b"")
    context = _FakeAsFileContext(library_path)
    events: list[bool] = []

    def fake_load_library(_path: str) -> None:
        events.append(context.active)

    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: context)
    monkeypatch.setattr(torch.ops, "load_library", fake_load_library)
    monkeypatch.setattr(torch.ops, "xpool", SimpleNamespace(abi_version=lambda: ABI_VERSION), raising=False)

    ensure_xpool_ops_loaded()

    assert events == [True]


def test_native_loader_rejects_missing_abi_version_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: _FakeAsFileContext(Path(__file__)))
    monkeypatch.setattr(torch.ops, "load_library", lambda _path: None)
    monkeypatch.setattr(torch.ops, "xpool", SimpleNamespace(), raising=False)

    with pytest.raises(NativeLoadError, match="abi_version"):
        ensure_xpool_ops_loaded()


def test_native_loader_rejects_abi_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cext_module, "files", lambda package: _FakeFiles(package))
    monkeypatch.setattr(cext_module, "as_file", lambda _resource: _FakeAsFileContext(Path(__file__)))
    monkeypatch.setattr(torch.ops, "load_library", lambda _path: None)
    monkeypatch.setattr(torch.ops, "xpool", SimpleNamespace(abi_version=lambda: ABI_VERSION + 1), raising=False)

    with pytest.raises(NativeLoadError, match="does not match"):
        ensure_xpool_ops_loaded()


class _FakeFiles:
    def __init__(self, package: str) -> None:
        self.package = package

    def joinpath(self, name: str) -> object:
        assert self.package == "xpool"
        assert name == "libxpool_cext.so"
        return object()


class _FakeOpNamespace:
    def __init__(self, events: list[str], *, abi_version: int) -> None:
        self.events = events
        self.abi_version_value = abi_version

    def abi_version(self) -> int:
        self.events.append("preflight")
        return self.abi_version_value


class _FakeAsFileContext(AbstractContextManager[Path]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.active = False

    def __enter__(self) -> Path:
        self.active = True
        return self.path

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.active = False
        return None
