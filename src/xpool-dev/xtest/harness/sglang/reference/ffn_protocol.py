"""Picklable messages for one isolated SGLang FFN reference job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FfnReferenceCaseSpec:
    """One ordered FFN evaluation and its private tensor exchange paths."""

    case_id: str
    layer_id: int
    input_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class FfnReferenceJob:
    """One raw-model load followed by an ordered FFN case batch."""

    model_path: Path
    tensor_parallel_size: int
    workdir: Path
    cases: tuple[FfnReferenceCaseSpec, ...]


@dataclass(frozen=True, slots=True)
class FfnReferenceCompleted:
    """Successful completion of every case and TP-rank cleanup."""

    case_count: int
