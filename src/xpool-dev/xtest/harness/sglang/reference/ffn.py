"""Parent-owned interface for isolated original-SGLang FFN evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import safetensors.torch
import torch

from xkit.child import PythonChildProcess
from xtest.harness.sglang.reference import ffn_child, ffn_protocol
from xtest.harness.support.wait import remaining_seconds


@dataclass(frozen=True, slots=True)
class SglangFfnRouting:
    """CPU token-major routes before the model's overall routed scale."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    def __post_init__(self) -> None:
        ids, weights = self.topk_ids, self.topk_weights
        if (
            ids.device.type != "cpu"
            or weights.device.type != "cpu"
            or ids.dtype != torch.int32
            or weights.dtype != torch.float32
            or ids.ndim != 2
            or weights.shape != ids.shape
            or min(ids.shape) <= 0
            or not ids.is_contiguous()
            or not weights.is_contiguous()
            or not torch.isfinite(weights).all()
        ):
            raise ValueError("FFN reference routing requires matching contiguous CPU int32/FP32 matrices")
        sorted_ids = ids.sort(dim=-1).values
        if (ids < 0).any() or (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
            raise ValueError("FFN reference routing requires CPU int32/FP32 matrices with distinct nonnegative IDs")


@dataclass(frozen=True, slots=True)
class SglangFfnReferenceCase:
    """One FFN input, optionally prescribing routes instead of natural TopK."""

    case_id: str
    layer_id: int
    hidden_states: torch.Tensor
    routing: SglangFfnRouting | None = None


@dataclass(frozen=True, slots=True)
class SglangFfnReferenceResult:
    """One original-SGLang FFN result returned on CPU."""

    case_id: str
    output: torch.Tensor


def run_ffn_reference(
    *,
    model_path: Path,
    tensor_parallel_size: int,
    cases: Sequence[SglangFfnReferenceCase],
    workdir: Path,
    timeout_seconds: float,
) -> tuple[SglangFfnReferenceResult, ...]:
    """Load one raw model, evaluate every case, and retire its isolated child.

    Inputs are readonly CPU values; synchronous file serialization snapshots them
    before child startup. Results preserve case order. The private work directory
    must be new, and child failure or invalid completion raises RuntimeError.
    """

    if timeout_seconds <= 0:
        raise ValueError("FFN reference timeout_seconds must be positive")
    deadline = time.monotonic() + timeout_seconds
    if not model_path.is_dir():
        raise ValueError(f"FFN reference model_path is not a directory: {model_path}")
    visible_device_count = torch.cuda.device_count()
    if tensor_parallel_size <= 0 or tensor_parallel_size > visible_device_count:
        raise ValueError(
            "FFN reference tensor_parallel_size must be positive and no greater than "
            f"the visible device count ({visible_device_count})"
        )
    normalized_cases = tuple(cases)
    if not normalized_cases:
        raise ValueError("FFN reference cases must be nonempty")
    case_ids = tuple(case.case_id for case in normalized_cases)
    if any(not case_id for case_id in case_ids):
        raise ValueError("FFN reference case_id must be nonempty")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("FFN reference case_id values must be unique")
    for case in normalized_cases:
        if case.layer_id < 0:
            raise ValueError("FFN reference layer_id must be nonnegative")
        hidden = case.hidden_states
        if hidden.ndim != 2 or hidden.device.type != "cpu" or not hidden.is_floating_point():
            raise ValueError("FFN reference hidden_states must be a rank-two CPU floating tensor")
        if case.routing is not None and case.routing.topk_ids.shape[0] != hidden.shape[0]:
            raise ValueError("FFN reference routing rows must match hidden states")

    workdir.mkdir(parents=True, exist_ok=False)
    case_specs: list[ffn_protocol.FfnReferenceCaseSpec] = []
    for index, case in enumerate(normalized_cases):
        case_dir = workdir / "cases" / f"case-{index:04d}"
        case_dir.mkdir(parents=True)
        input_path = case_dir / "input.safetensors"
        tensors = {"hidden_states": case.hidden_states.detach().contiguous()}
        if case.routing is not None:
            tensors["topk_ids"] = case.routing.topk_ids.detach()
            tensors["topk_weights"] = case.routing.topk_weights.detach()
        with input_path.open("xb") as output:
            output.write(safetensors.torch.save(tensors))
        case_specs.append(
            ffn_protocol.FfnReferenceCaseSpec(
                case_id=case.case_id,
                layer_id=case.layer_id,
                input_path=input_path,
                output_path=case_dir / "output.safetensors",
            )
        )
    process = PythonChildProcess(
        "sglang-ffn-reference",
        ffn_child.run_ffn_reference_child,
        ffn_protocol.FfnReferenceJob(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            workdir=workdir,
            cases=tuple(case_specs),
        ),
        log_path=workdir / "child.log",
    )
    try:
        process.start()
        completed = process.receive(
            ffn_protocol.FfnReferenceCompleted,
            timeout_seconds=remaining_seconds(deadline, "completion"),
        )
        if completed.case_count != len(case_specs):
            raise RuntimeError(f"FFN reference completed {completed.case_count} cases, expected {len(case_specs)}")
        process.wait(timeout_seconds=remaining_seconds(deadline, "child exit"))
        return tuple(read_reference_result(spec) for spec in case_specs)
    finally:
        if process.process.is_alive():
            PythonChildProcess.terminate_all((process,))
        process.close()


def read_reference_result(spec: ffn_protocol.FfnReferenceCaseSpec) -> SglangFfnReferenceResult:
    """Read one child-owned CPU output under its requested case identity."""

    try:
        tensors = safetensors.torch.load_file(spec.output_path, device="cpu")
    except Exception as error:
        raise RuntimeError(f"FFN reference could not read {spec.output_path}: {error}") from error
    if set(tensors) != {"hidden_states"}:
        raise RuntimeError(f"FFN reference output has invalid tensor keys: {sorted(tensors)}")
    return SglangFfnReferenceResult(case_id=spec.case_id, output=tensors["hidden_states"])
