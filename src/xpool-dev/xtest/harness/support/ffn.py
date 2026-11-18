"""Framework-independent CPU comparison for FFN numerical evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Self, cast

import safetensors
import torch
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from xpool.fabric import FabricGenerationId
from xpool.ffn import MoeFfnSpec, RouterAlgorithm, TopKWeightRounding
from xpool.model import ModelId
from xpool.utils.dtype import TorchDtype

NORMALIZED_RMSE_DENOMINATOR_FLOOR = 1e-12
FFN_NUMERICAL_ARTIFACT_FILENAME = "ffn-numerical.json"
FFN_TOPOLOGY_ARTIFACT_FILENAME = "ffn-topology.json"
FFN_OUTPUT_ATOL = 0.02
FFN_OUTPUT_RTOL = 0.02
FFN_OUTPUT_NORMALIZED_RMSE_LIMIT = 0.01
FFN_OUTPUT_MAXIMUM_SCALED_ERROR_LIMIT = 0.10
FFN_ROUTING_ATOL = 0.0001
FFN_ROUTING_RTOL = 0.001


def validate_qualification_dtype(dtype: torch.dtype) -> torch.dtype:
    """Restrict numerical qualification to the admitted BF16 and FP16 payloads."""

    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("qualification payload dtype must be BF16 or FP16")
    return dtype


type QualificationDtype = Annotated[TorchDtype, AfterValidator(validate_qualification_dtype)]


def generate_ffn_hidden_states(
    *,
    hidden_size: int,
    seed: int,
    row_counts: tuple[int, ...],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Generate deterministic admitted-dtype prefixes from one CPU normal sample."""

    if hidden_size <= 0 or seed < 0 or not row_counts or any(row_count <= 0 for row_count in row_counts):
        raise ValueError("FFN input geometry and seed must be positive, nonempty, and nonnegative")
    if tuple(sorted(set(row_counts))) != row_counts:
        raise ValueError("FFN input row counts must be strictly increasing")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"FFN input dtype must be BF16 or FP16, received {dtype}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.randn((row_counts[-1], hidden_size), generator=generator).to(dtype).contiguous()
    if not torch.isfinite(full).all():
        raise RuntimeError("FFN input generator produced non-finite values")
    return tuple(full[:row_count].clone() for row_count in row_counts)


def read_ffn_routing_records(
    directory: Path,
    generation: FabricGenerationId,
) -> dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Read complete, non-dropped Routing Observer records for one generation."""

    records_by_key: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    paths = tuple(sorted(directory.glob(f"xpool.ffn-routing-observer.{generation.format()}.*.safetensors")))
    if not paths:
        raise AssertionError("FFN Routing Observer produced no generation artifacts")
    for path in paths:
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as snapshot:
            metadata = snapshot.metadata()
            if metadata is None or set(metadata) != {"generation", "pe", "sequence", "dropped", "records"}:
                raise AssertionError(f"FFN Routing Observer metadata is invalid in {path}")
            if metadata["generation"] != generation.format():
                raise AssertionError(f"FFN Routing Observer generation is invalid in {path}")
            try:
                sequence = int(metadata["sequence"])
                dropped = int(metadata["dropped"])
                pe = int(metadata["pe"])
                record_metadata = json.loads(metadata["records"])
            except (ValueError, json.JSONDecodeError) as error:
                raise AssertionError(f"FFN Routing Observer metadata is malformed in {path}") from error
            if min(sequence, dropped, pe) < 0 or dropped != 0:
                raise AssertionError(f"FFN Routing Observer dropped or invalid records in {path}")
            if not isinstance(record_metadata, list) or len(record_metadata) != sequence:
                raise AssertionError(f"FFN Routing Observer record count is invalid in {path}")
            expected_tensor_keys: set[str] = set()
            for ordinal, raw_record in enumerate(record_metadata):
                if (
                    not isinstance(raw_record, dict)
                    or tuple(raw_record)
                    != (
                        "instance_index",
                        "invocation_sequence",
                        "layer_ordinal",
                    )
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_record.values())
                ):
                    raise AssertionError(f"FFN Routing Observer record metadata is invalid in {path}")
                typed_record = cast(dict[str, int], raw_record)
                key = (
                    typed_record["instance_index"],
                    typed_record["invocation_sequence"],
                    typed_record["layer_ordinal"],
                )
                if key in records_by_key:
                    raise AssertionError(f"FFN Routing Observer duplicated business key {key}")
                ids_key = f"records.{ordinal}.topk_ids"
                weights_key = f"records.{ordinal}.topk_weights"
                expected_tensor_keys.update((ids_key, weights_key))
                ids = snapshot.get_tensor(ids_key)
                weights = snapshot.get_tensor(weights_key)
                records_by_key[key] = (ids, weights)
            if set(snapshot.keys()) != expected_tensor_keys:
                raise AssertionError(f"FFN Routing Observer tensor keys are invalid in {path}")
    return records_by_key


@dataclass(frozen=True, slots=True)
class FfnOutputComparison:
    """Complete final-output error metrics and their verdict."""

    passed: bool
    maximum_absolute_error: float
    maximum_error_flat_index: int
    reference_at_maximum_error: float
    actual_at_maximum_error: float
    mean_absolute_error: float
    normalized_rmse: float
    maximum_scaled_error: float
    outlier_count: int
    outlier_fraction: float
    element_count: int
    atol: float
    rtol: float
    normalized_rmse_limit: float
    maximum_scaled_error_limit: float


@dataclass(frozen=True, slots=True)
class FfnRouterComparison:
    """Stage-local Router accuracy and exact installed-execution agreement."""

    passed: bool
    execution_capacity: int
    projection_outlier_count: int
    projection_maximum_absolute_error: float
    projection_atol: float
    projection_rtol: float
    selection_violation_count: int
    boundary_ambiguity_count: int
    coefficient_outlier_count: int
    coefficient_maximum_absolute_error: float
    coefficient_atol: float
    coefficient_rtol: float
    installed_ids_equal: bool
    installed_weights_equal: bool
    eager_graph_equal: bool


class FfnNumericalSampleEvidence(BaseModel):
    """One model-layer-row numerical comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_id: int = Field(ge=0)
    row_count: int = Field(gt=0)
    output: FfnOutputComparison
    router: FfnRouterComparison | None


class FfnNumericalEvidence(BaseModel):
    """Versionless runner-owned numerical qualification evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: ModelId
    model_spec_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: FabricGenerationId
    payload_dtype: QualificationDtype
    samples: tuple[FfnNumericalSampleEvidence, ...]

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        """Require nonempty unique Layer-by-Rows coordinates."""

        coordinates = tuple((sample.layer_id, sample.row_count) for sample in self.samples)
        if not coordinates or len(coordinates) != len(set(coordinates)):
            raise ValueError("FFN numerical samples must contain unique coordinates")
        return self

    def write(self, path: Path) -> None:
        """Exclusively create one runner-owned evidence artifact."""

        with path.open("xb") as output:
            output.write(self.model_dump_json(indent=2).encode())


class FfnTopologyProcessEvidence(BaseModel):
    """One observed CUDA client and physical device identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    process_id: int = Field(gt=0)
    device: int = Field(ge=0)
    device_uuid: str = Field(min_length=1)


class FfnTopologyMpsServerEvidence(BaseModel):
    """One observed MPS server and its supervised CUDA clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    process_id: int = Field(gt=0)
    client_process_names: tuple[str, ...]
    active_thread_percentage: float = Field(ge=1, le=100)

    @model_validator(mode="after")
    def validate_clients(self) -> Self:
        """Require a nonempty unique client-name set."""

        if not self.client_process_names or len(self.client_process_names) != len(set(self.client_process_names)):
            raise ValueError("FFN topology MPS clients must be nonempty and unique")
        return self


class FfnTopologyOutputEvidence(BaseModel):
    """One rank set compared with a complete FFN Reference output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_ordinal: int = Field(ge=0)
    output_ranks: tuple[int, ...]
    comparison: FfnOutputComparison

    @model_validator(mode="after")
    def validate_ranks(self) -> Self:
        """Require a nonempty strictly increasing Output-rank set."""

        if not self.output_ranks or tuple(sorted(set(self.output_ranks))) != self.output_ranks:
            raise ValueError("FFN topology Output ranks must be nonempty and strictly increasing")
        return self


class FfnTopologyEvidence(BaseModel):
    """Versionless runner-owned topology qualification evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    generation: FabricGenerationId
    processes: tuple[FfnTopologyProcessEvidence, ...]
    mps_servers: tuple[FfnTopologyMpsServerEvidence, ...]
    outputs: tuple[FfnTopologyOutputEvidence, ...]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Require complete unique process, MPS, and Output coordinates."""

        process_names = tuple(process.name for process in self.processes)
        process_ids = tuple(process.process_id for process in self.processes)
        if not process_names or len(process_names) != len(set(process_names)):
            raise ValueError("FFN topology process names must be nonempty and unique")
        if len(process_ids) != len(set(process_ids)):
            raise ValueError("FFN topology process IDs must be unique")
        if not self.mps_servers or len({server.process_id for server in self.mps_servers}) != len(self.mps_servers):
            raise ValueError("FFN topology MPS servers must be nonempty and unique")
        clients = tuple(name for server in self.mps_servers for name in server.client_process_names)
        if len(clients) != len(set(clients)) or not set(clients) <= set(process_names):
            raise ValueError("FFN topology MPS membership must reference unique observed clients")
        output_coordinates = tuple((output.request_ordinal, output.output_ranks) for output in self.outputs)
        if not output_coordinates or len(output_coordinates) != len(set(output_coordinates)):
            raise ValueError("FFN topology Outputs must be nonempty and unique")
        return self

    def write(self, path: Path) -> None:
        """Exclusively create one runner-owned evidence artifact."""

        with path.open("xb") as output:
            output.write(self.model_dump_json(indent=2).encode())


def compare_ffn_outputs(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    normalized_rmse_limit: float,
    maximum_scaled_error_limit: float,
) -> FfnOutputComparison:
    """Measure one admitted-dtype production output against its reference."""

    if reference.shape != actual.shape:
        raise AssertionError(f"FFN output shape mismatch: reference={reference.shape}, actual={actual.shape}")
    if reference.dtype != actual.dtype or reference.dtype not in (torch.bfloat16, torch.float16):
        raise AssertionError(
            f"FFN output dtypes must match and be BF16 or FP16: reference={reference.dtype}, actual={actual.dtype}"
        )
    if reference.numel() == 0:
        raise AssertionError("FFN output tensors must be nonempty")
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise AssertionError("FFN output tensors must contain only finite values")
    if min(atol, rtol, normalized_rmse_limit, maximum_scaled_error_limit) < 0:
        raise ValueError("FFN output tolerances must be nonnegative")

    reference_fp32 = reference.float()
    actual_fp32 = actual.float()
    absolute_error = (actual_fp32 - reference_fp32).abs()
    maximum_error_flat_index = int(absolute_error.argmax().item())
    reference_flat = reference_fp32.flatten()
    actual_flat = actual_fp32.flatten()
    mixed_bound = atol + rtol * reference_fp32.abs()
    outlier_count = int((absolute_error > mixed_bound).sum().item())
    reference_scale = reference_fp32.square().mean().sqrt().clamp_min(NORMALIZED_RMSE_DENOMINATOR_FLOOR)
    normalized_rmse = float(((actual_fp32 - reference_fp32).square().mean().sqrt() / reference_scale).item())
    maximum_scaled_error = float((absolute_error / torch.maximum(reference_fp32.abs(), reference_scale)).max().item())
    element_count = reference.numel()
    return FfnOutputComparison(
        passed=(normalized_rmse <= normalized_rmse_limit and maximum_scaled_error <= maximum_scaled_error_limit),
        maximum_absolute_error=float(absolute_error.max().item()),
        maximum_error_flat_index=maximum_error_flat_index,
        reference_at_maximum_error=float(reference_flat[maximum_error_flat_index].item()),
        actual_at_maximum_error=float(actual_flat[maximum_error_flat_index].item()),
        mean_absolute_error=float(absolute_error.mean().item()),
        normalized_rmse=normalized_rmse,
        maximum_scaled_error=maximum_scaled_error,
        outlier_count=outlier_count,
        outlier_fraction=outlier_count / element_count,
        element_count=element_count,
        atol=atol,
        rtol=rtol,
        normalized_rmse_limit=normalized_rmse_limit,
        maximum_scaled_error_limit=maximum_scaled_error_limit,
    )


def router_selection_bounds(
    logits: torch.Tensor,
    *,
    algorithm: RouterAlgorithm,
    correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP64 score bounds and FP32 exponential arguments for selection.

    This engineering policy covers uncorrected Softmax and stored-FP32 Sigmoid.
    Its approximate-operation allowance is not a portable instruction proof.
    """

    # Include FP32 RN and the inspected NVIDIA exp2/divide approximation errors.
    unit_roundoff = 2.0**-24
    relative_error = 5 * unit_roundoff / (1 - 4 * unit_roundoff)
    absolute_error = 2.5 * 2.0**-149 / (1 - 4 * unit_roundoff)
    values = logits.float()
    log2_e = torch.tensor(math.log2(math.e), dtype=torch.float32)
    if algorithm is RouterAlgorithm.SOFTMAX:
        if correction_bias is not None:
            raise ValueError("Router selection policy does not cover corrected Softmax")
        argument = (values - values.max(dim=-1, keepdim=True).values) * log2_e
        # Softmax's common denominator cancels in ranking; work in log2 space.
        lower = argument.double() + 2 * math.log2(1 - relative_error)
        upper = argument.double() + 2 * math.log2(1 + relative_error)
    else:
        argument = -values * log2_e
        exponential = torch.exp2(argument.double())
        exponential_lower = (exponential * (1 - relative_error) - absolute_error).clamp_min(0)
        exponential_upper = exponential * (1 + relative_error) + absolute_error
        lower = ((1 - relative_error) / ((1 + exponential_upper) * (1 + unit_roundoff)) - absolute_error).clamp_min(0)
        upper = (1 + relative_error) / ((1 + exponential_lower) * (1 - unit_roundoff)) + absolute_error
        lower = torch.where(lower < 2.0**-126, 0.0, lower)
        upper = torch.where(upper < 2.0**-126, 0.0, upper)
        if correction_bias is not None:
            lower = lower + correction_bias.double()
            upper = upper + correction_bias.double()
        lower, upper = lower.float().double(), upper.float().double()
    if not torch.isfinite(argument).all():
        raise ValueError("Router selection policy requires finite FP32 exponential arguments")
    return lower, upper, argument


def router_coefficient_outliers(
    actual: torch.Tensor, reference: torch.Tensor, *, rounding_dtype: torch.dtype
) -> torch.Tensor:
    """Apply the FP32 coefficient allowance before the declared final store."""

    error = FFN_ROUTING_ATOL + FFN_ROUTING_RTOL * reference.abs()
    lower = (reference - error).to(rounding_dtype).float()
    upper = (reference + error).to(rounding_dtype).float()
    return (actual != actual.to(rounding_dtype).float()) | (actual < lower) | (actual > upper) | ~torch.isfinite(actual)


def compare_ffn_router(
    *,
    layer: MoeFfnSpec,
    execution_capacity: int,
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    installed_ids_equal: bool,
    installed_weights_equal: bool,
    eager_graph_equal: bool,
) -> FfnRouterComparison:
    """Check projection, selection and coefficients at their own stored boundaries.

    Inputs are CPU tensors from one prepared Router. The caller checks carrier
    geometry and separately establishes exact production and Graph agreement.
    """

    projected_reference = reference_logits.to(logits.dtype).double()
    projection_atol, projection_rtol = (0.02, 0.02) if logits.dtype != torch.float32 else (1e-4, 1e-3)
    projection_error = (logits.double() - projected_reference).abs()
    projection_outliers = projection_error > projection_atol + projection_rtol * projected_reference.abs()

    lower, upper, argument = router_selection_bounds(
        logits, algorithm=layer.router_algorithm, correction_bias=correction_bias
    )
    if layer.router_algorithm is RouterAlgorithm.SOFTMAX:
        unit_roundoff = 2.0**-24
        relative_error = 5 * unit_roundoff / (1 - 4 * unit_roundoff)
        sum_error = (layer.routed_expert_count - 1) * unit_roundoff
        sum_bound = layer.routed_expert_count * (1 + relative_error) / (1 - sum_error)
        cutoff = lower.topk(layer.routed_topk, dim=-1).values[:, -1] - math.log2(sum_bound)
        if not (cutoff > -126).all():
            raise ValueError("Router selection policy does not cover a subnormal Softmax cutoff")

    sorted_ids = topk_ids.sort(dim=-1).values
    invalid_rows = ((topk_ids < 0) | (topk_ids >= layer.routed_expert_count)).any(dim=-1)
    invalid_rows |= (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(dim=-1)
    indices = topk_ids.long().clamp(0, layer.routed_expert_count - 1)
    selected = torch.zeros_like(lower, dtype=torch.bool).scatter_(1, indices, True)
    same_source = argument.gather(1, indices).unsqueeze(-1) == argument.unsqueeze(1)
    if correction_bias is not None:
        same_source &= correction_bias[indices].unsqueeze(-1) == correction_bias.reshape(1, 1, -1)
    other_ids = torch.arange(layer.routed_expert_count).reshape(1, 1, -1)
    preferred = (
        other_ids < indices.unsqueeze(-1)
        if layer.router_algorithm is RouterAlgorithm.SOFTMAX
        else other_ids > indices.unsqueeze(-1)
    )
    selected_upper = upper.gather(1, indices).unsqueeze(-1)
    forced_tie = lower.unsqueeze(1) == selected_upper
    wrong_tie = ((same_source | forced_tie) & preferred & ~selected.unsqueeze(1)).any(dim=(1, 2))
    excluded_lower = lower.masked_fill(selected, -math.inf).max(dim=-1).values
    violations = invalid_rows | wrong_tie | (excluded_lower > selected_upper.squeeze(-1).min(dim=-1).values)
    selected_lower = lower.gather(1, indices).unsqueeze(-1)
    could_reverse = (upper.unsqueeze(1) > selected_lower) | ((upper.unsqueeze(1) == selected_lower) & preferred)
    ambiguous = (could_reverse & ~selected.unsqueeze(1) & ~same_source).any(dim=(1, 2))

    scores = (
        torch.softmax(logits.float(), dim=-1)
        if layer.router_algorithm is RouterAlgorithm.SOFTMAX
        else torch.sigmoid(logits.float())
    )
    coefficients = scores.gather(1, indices)
    if layer.renormalize:
        normalizer = coefficients.sum(dim=-1, keepdim=True)
        coefficients = coefficients / torch.where(normalizer > 0, normalizer, 1)
    rounding_dtype = logits.dtype if layer.topk_weight_rounding is TopKWeightRounding.LOGITS_DTYPE else torch.float32
    coefficient_outliers = router_coefficient_outliers(topk_weights, coefficients, rounding_dtype=rounding_dtype)
    finite = bool(torch.isfinite(logits).all() and torch.isfinite(reference_logits).all())
    return FfnRouterComparison(
        passed=(
            finite
            and not bool(projection_outliers.any() or violations.any() or coefficient_outliers.any())
            and installed_ids_equal
            and installed_weights_equal
            and eager_graph_equal
        ),
        execution_capacity=execution_capacity,
        projection_outlier_count=int(projection_outliers.sum()),
        projection_maximum_absolute_error=float(projection_error.max()),
        projection_atol=projection_atol,
        projection_rtol=projection_rtol,
        selection_violation_count=int(violations.sum()),
        boundary_ambiguity_count=int(ambiguous.sum()),
        coefficient_outlier_count=int(coefficient_outliers.sum()),
        coefficient_maximum_absolute_error=float((topk_weights - coefficients).abs().max()),
        coefficient_atol=FFN_ROUTING_ATOL,
        coefficient_rtol=FFN_ROUTING_RTOL,
        installed_ids_equal=installed_ids_equal,
        installed_weights_equal=installed_weights_equal,
        eager_graph_equal=eager_graph_equal,
    )
