"""Confinement and explicit file inventory at the offline CLI boundary."""

import hashlib
from pathlib import Path

from xpool.observability.artifacts import ArtifactInputs, AttachmentIndex, read_artifacts
from xpool.observability.calibration_index import CalibrationIndex, build_calibration
from xpool.observability.reconstruction import Reconstruction, reconstruct
from xpool.observability.trace.models import InputDigest, TraceError, TraceErrorCode, TraceInputs


def confined(root: Path, name: str) -> Path:
    """Resolve a relative input or bundle member, rejecting directory escapes."""
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise TraceError(TraceErrorCode.INPUT, "path must remain relative to its input directory", path=path)
    return path


def load_inputs(root: Path, manifest: Path) -> tuple[Reconstruction, CalibrationIndex, tuple[InputDigest, ...]]:
    """Read an explicit JSON inventory and apply artifact and calibration verification.

    The manifest may live outside the raw directory. No discovery or raw writes
    occur. Structural errors raise TraceError; Artifact and calibration evidence errors retain
    their original structured ObservationError categories and sources.
    """
    try:
        declaration = TraceInputs.model_validate_json(manifest.read_bytes())
    except (OSError, ValueError) as error:
        raise TraceError(TraceErrorCode.INPUT, "invalid input manifest", path=manifest) from error
    chunks = tuple(confined(root, name) for name in declaration.chunks)
    metadata = tuple(confined(root, name) for name in declaration.metadata)
    quality = tuple(confined(root, name) for name in declaration.quality)
    attachment_root = confined(root, declaration.attachment_root) if declaration.attachment_root is not None else root
    indices = tuple(
        AttachmentIndex(entry.run_id, entry.owner, entry.entries, entry.closing_complete)
        for entry in declaration.attachment_indices
    )
    view = reconstruct(read_artifacts(ArtifactInputs(chunks, metadata, quality, indices, attachment_root)))
    index = build_calibration(view, declaration.samplers)
    paths = {manifest.resolve(), *chunks, *metadata, *quality}
    paths.update(attachment.path for attachment in view.attachments if attachment.path is not None)
    digests = []
    for path in sorted(paths):
        with path.open("rb") as stream:
            digests.append(InputDigest(path=str(path), sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
    return view, index, tuple(digests)


def prepare_output(output: Path, protected: tuple[Path, ...]) -> None:
    """Create a new output directory disjoint from every protected input tree.

    Existing directories are rejected. Failed publication may leave diagnostics
    or partial files, but the completion manifest is written only on success.
    """
    resolved = output.resolve()
    for path in protected:
        source = path.resolve()
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise TraceError(TraceErrorCode.OUTPUT, "output overlaps input", path=output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise TraceError(TraceErrorCode.OUTPUT, "output must be a new separate directory", path=output) from error
