"""Explicit, checksum-pinned Trace Processor preparation for Linux x86-64."""

import hashlib
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import httpx

from xpool.observability.trace import versions
from xpool.observability.trace.models import TraceError, TraceErrorCode


def verify_processor(path: Path) -> Path:
    """Check the pinned executable locally, without discovery or network access.

    Raises TraceError for a missing, nonexecutable or different build. An
    explicit path obeys the same qualification pin as the managed cache.
    """
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise TraceError(TraceErrorCode.TOOL_VERSION, "qualified platform is linux x86_64", path=path)
    if version("perfetto") != versions.perfetto_version:
        raise TraceError(TraceErrorCode.TOOL_VERSION, f"requires perfetto {versions.perfetto_version}", path=path)
    if not path.is_file():
        raise TraceError(TraceErrorCode.DEPENDENCY_MISSING, "run xpool trace setup first", path=path)
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        raise TraceError(TraceErrorCode.DEPENDENCY_MISSING, "cannot read processor", path=path) from error
    if digest != versions.processor_sha256 or not os.access(path, os.X_OK):
        raise TraceError(
            TraceErrorCode.TOOL_VERSION,
            f"requires executable trace processor {versions.processor_version} (pinned sha256)",
            path=path,
        )
    return path.resolve()


def setup_processor(directory: Path) -> Path:
    """Explicitly download the official pinned binary and atomically publish it.

    Existing valid installations are reused. SHA-256 is checked before making
    the download executable; failed downloads never replace the target binary.
    """
    target = directory / "trace_processor_shell"
    if target.exists():
        return verify_processor(target)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", versions.processor_url, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    digest = hashlib.sha256()
                    for block in response.iter_bytes():
                        digest.update(block)
                        stream.write(block)
                    stream.flush()
                    if digest.hexdigest() != versions.processor_sha256:
                        raise TraceError(TraceErrorCode.TOOL_VERSION, "download checksum mismatch", path=target)
                    temporary.chmod(0o755)
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
    except (httpx.HTTPError, OSError) as error:
        raise TraceError(TraceErrorCode.TOOL_SETUP, "processor download failed", path=target) from error
    return verify_processor(target)
