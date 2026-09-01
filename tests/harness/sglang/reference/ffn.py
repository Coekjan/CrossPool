"""Parent-owned interface for isolated original-SGLang FFN evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import safetensors.torch
import torch

from tests.harness.runner.child import PythonChildProcess
from tests.harness.sglang.reference import ffn_child, ffn_protocol
from tests.harness.support.wait import remaining_seconds


@dataclass(frozen=True, slots=True)
class SglangFfnReferenceCase:
    """One hidden-state input for one original-model FFN layer."""

    case_id: str
    layer_id: int
    hidden_states: torch.Tensor


@dataclass(frozen=True, slots=True)
class SglangFfnRoutingEvidence:
    """Canonical routed and always-selected expert choices."""

    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class SglangFfnReferenceResult:
    """One original-SGLang FFN result returned on CPU."""

    case_id: str
    layer_id: int
    output: torch.Tensor
    routing: SglangFfnRoutingEvidence | None


class SglangFfnReferenceRunner:
    """Own one single-use original-SGLang FFN child process."""

    def __init__(self, *, workdir: Path, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("FFN reference timeout_seconds must be positive")
        self.workdir = workdir
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        *,
        model_path: Path,
        tensor_parallel_size: int,
        cases: Sequence[SglangFfnReferenceCase],
    ) -> tuple[SglangFfnReferenceResult, ...]:
        """Load one raw model, evaluate every case, and release the child.

        Args:
            model_path: Existing raw model checkpoint directory.
            tensor_parallel_size: Number of SGLang tensor-parallel ranks.
            cases: Nonempty ordered FFN evaluation batch.

        Returns:
            CPU results in the same order as ``cases``.

        Raises:
            ValueError: If the model, TP placement, or a case is invalid.
            FileExistsError: If this runner's work directory already exists.
            RuntimeError: If the child fails, times out, or returns invalid files.
        """

        deadline = time.monotonic() + self.timeout_seconds
        if not model_path.is_dir():
            raise ValueError(f"FFN reference model_path is not a directory: {model_path}")
        visible_device_count = torch.cuda.device_count()
        if tensor_parallel_size <= 0 or tensor_parallel_size > visible_device_count:
            raise ValueError(
                "FFN reference tensor_parallel_size must be positive and no greater than "
                f"the visible CUDA device count ({visible_device_count})"
            )
        normalized_cases = tuple(cases)
        if not normalized_cases:
            raise ValueError("FFN reference cases must be nonempty")
        case_ids = tuple(case.case_id for case in normalized_cases)
        if any(not case_id for case_id in case_ids):
            raise ValueError("FFN reference case_id must be nonempty")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("FFN reference case_id values must be unique")

        normalized_hidden_states: list[torch.Tensor] = []
        for case in normalized_cases:
            if case.layer_id < 0:
                raise ValueError("FFN reference layer_id must be nonnegative")
            hidden_states = case.hidden_states
            if hidden_states.ndim != 2 or hidden_states.device.type != "cpu" or not hidden_states.is_floating_point():
                raise ValueError("FFN reference hidden_states must be a rank-two CPU floating tensor")
            normalized_hidden_states.append(hidden_states.detach().clone().contiguous())

        self.workdir.mkdir(parents=True, exist_ok=False)
        case_specs: list[ffn_protocol.FfnReferenceCaseSpec] = []
        for index, (case, hidden_states) in enumerate(zip(normalized_cases, normalized_hidden_states, strict=True)):
            case_dir = self.workdir / "cases" / f"case-{index:04d}"
            case_dir.mkdir(parents=True)
            input_path = case_dir / "input.safetensors"
            with input_path.open("xb") as input_file:
                input_file.write(safetensors.torch.save({"hidden_states": hidden_states}))
            case_specs.append(
                ffn_protocol.FfnReferenceCaseSpec(
                    case_id=case.case_id,
                    layer_id=case.layer_id,
                    input_path=input_path,
                    output_path=case_dir / "output.safetensors",
                )
            )

        process = PythonChildProcess.start(
            "sglang-ffn-reference",
            ffn_child.run_ffn_reference_child,
            ffn_protocol.FfnReferenceJob(
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                workdir=self.workdir,
                cases=tuple(case_specs),
            ),
            log_path=self.workdir / "child.log",
        )
        try:
            completed = process.receive(
                ffn_protocol.FfnReferenceCompleted,
                timeout_seconds=remaining_seconds(deadline, "completion"),
            )
            if completed.case_count != len(case_specs):
                raise RuntimeError(f"FFN reference completed {completed.case_count} cases, expected {len(case_specs)}")
            process.wait(timeout_seconds=remaining_seconds(deadline, "child exit"))
            return tuple(
                read_reference_result(case, spec) for case, spec in zip(normalized_cases, case_specs, strict=True)
            )
        finally:
            if process.process.is_alive():
                PythonChildProcess.terminate_all((process,))
            process.close()


def read_reference_result(
    case: SglangFfnReferenceCase,
    spec: ffn_protocol.FfnReferenceCaseSpec,
) -> SglangFfnReferenceResult:
    """Project one validated private output file into the public result."""

    try:
        tensors = safetensors.torch.load_file(spec.output_path, device="cpu")
    except Exception as error:
        raise RuntimeError(f"FFN reference could not read {spec.output_path}: {error}") from error
    keys = set(tensors)
    if keys == {"hidden_states"}:
        routing = None
    elif keys == {"hidden_states", "topk_ids", "topk_weights"}:
        routing = SglangFfnRoutingEvidence(
            topk_ids=tensors["topk_ids"],
            topk_weights=tensors["topk_weights"],
        )
    else:
        raise RuntimeError(f"FFN reference output has invalid tensor keys: {sorted(keys)}")
    return SglangFfnReferenceResult(
        case_id=case.case_id,
        layer_id=case.layer_id,
        output=tensors["hidden_states"],
        routing=routing,
    )
