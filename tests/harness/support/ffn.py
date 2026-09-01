"""Framework-independent CPU comparison for FFN numerical evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import safetensors
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from xpool.fabric import FabricGenerationId

NORMALIZED_RMSE_DENOMINATOR_FLOOR = 1e-12
FFN_NUMERICAL_ARTIFACT_FILENAME = "ffn-numerical.json"
FFN_TOPOLOGY_ARTIFACT_FILENAME = "ffn-topology.json"
FFN_OUTPUT_ATOL = 0.02
FFN_OUTPUT_RTOL = 0.02
FFN_OUTPUT_NORMALIZED_RMSE_LIMIT = 0.01
FFN_OUTPUT_MAXIMUM_SCALED_ERROR_LIMIT = 0.10
FFN_ROUTING_ATOL = 0.0001
FFN_ROUTING_RTOL = 0.001


def generate_ffn_hidden_states(
    *,
    hidden_size: int,
    seed: int,
    row_counts: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    """Generate deterministic BF16 prefix inputs from one CPU normal sample."""

    if hidden_size <= 0 or seed < 0 or not row_counts or any(row_count <= 0 for row_count in row_counts):
        raise ValueError("FFN input geometry and seed must be positive, nonempty, and nonnegative")
    if tuple(sorted(set(row_counts))) != row_counts:
        raise ValueError("FFN input row counts must be strictly increasing")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.randn((row_counts[-1], hidden_size), generator=generator).to(torch.bfloat16).contiguous()
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
class FfnRoutingComparison:
    """Canonical expert-route comparison and first mismatch evidence."""

    passed: bool
    first_mismatch_token: int | None
    first_mismatch_pair: int | None
    reference_expert_id: int | None
    actual_expert_id: int | None
    reference_weight: float | None
    actual_weight: float | None
    maximum_absolute_weight_error: float
    atol: float
    rtol: float


class FfnNumericalSampleEvidence(BaseModel):
    """One model-layer-row numerical comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_id: int = Field(ge=0)
    row_count: int = Field(gt=0)
    output: FfnOutputComparison
    routing: FfnRoutingComparison | None


class FfnNumericalEvidence(BaseModel):
    """Versionless runner-owned numerical qualification evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_spec_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: FabricGenerationId
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
    cuda_device: int = Field(ge=0)
    gpu_uuid: str = Field(min_length=1)


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
        if len(clients) != len(set(clients)) or set(clients) != set(process_names):
            raise ValueError("FFN topology MPS membership must cover every CUDA client exactly once")
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
    """Measure one BF16 production output against its reference."""

    if reference.shape != actual.shape:
        raise AssertionError(f"FFN output shape mismatch: reference={reference.shape}, actual={actual.shape}")
    if reference.dtype != torch.bfloat16 or actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"FFN output dtype must be torch.bfloat16: reference={reference.dtype}, actual={actual.dtype}"
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


def compare_ffn_routing(
    reference_ids: torch.Tensor,
    reference_weights: torch.Tensor,
    actual_ids: torch.Tensor,
    actual_weights: torch.Tensor,
    *,
    expert_count: int,
    atol: float,
    rtol: float,
) -> FfnRoutingComparison:
    """Compare semantic routes after sorting each token's pairs by expert ID."""

    if expert_count <= 0:
        raise ValueError("FFN routing expert_count must be positive")
    if atol < 0 or rtol < 0:
        raise ValueError("FFN routing tolerances must be nonnegative")
    for name, ids, weights in (
        ("reference", reference_ids, reference_weights),
        ("actual", actual_ids, actual_weights),
    ):
        if ids.ndim != 2 or ids.shape != weights.shape or ids.numel() == 0:
            raise AssertionError(f"FFN {name} routing IDs and weights must be nonempty matching rank-two tensors")
        if ids.dtype != torch.int32 or weights.dtype != torch.float32:
            raise AssertionError(
                f"FFN {name} routing dtypes must be int32/float32, received {ids.dtype}/{weights.dtype}"
            )
        if ids.device.type != "cpu" or weights.device.type != "cpu":
            raise AssertionError(f"FFN {name} routing evidence must be on CPU")
        if not torch.isfinite(weights).all():
            raise AssertionError(f"FFN {name} routing weights must be finite")
        if (ids < 0).any() or (ids >= expert_count).any():
            raise AssertionError(f"FFN {name} routing contains an illegal expert ID")
        sorted_ids = ids.sort(dim=1).values
        if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
            raise AssertionError(f"FFN {name} routing contains duplicate expert IDs for one token")
    if reference_ids.shape != actual_ids.shape:
        raise AssertionError(f"FFN routing shape mismatch: reference={reference_ids.shape}, actual={actual_ids.shape}")

    reference_order = reference_ids.argsort(dim=1)
    actual_order = actual_ids.argsort(dim=1)
    reference_ids = reference_ids.gather(1, reference_order)
    actual_ids = actual_ids.gather(1, actual_order)
    reference_weights = reference_weights.gather(1, reference_order)
    actual_weights = actual_weights.gather(1, actual_order)
    absolute_weight_error = (actual_weights - reference_weights).abs()
    weight_outliers = absolute_weight_error > atol + rtol * reference_weights.abs()
    mismatches = (actual_ids != reference_ids) | weight_outliers
    mismatch_indices = mismatches.nonzero()
    if mismatch_indices.numel() == 0:
        first_token = first_pair = None
    else:
        first_token, first_pair = (int(value) for value in mismatch_indices[0].tolist())

    return FfnRoutingComparison(
        passed=first_token is None,
        first_mismatch_token=first_token,
        first_mismatch_pair=first_pair,
        reference_expert_id=None if first_token is None else int(reference_ids[first_token, first_pair].item()),
        actual_expert_id=None if first_token is None else int(actual_ids[first_token, first_pair].item()),
        reference_weight=None if first_token is None else float(reference_weights[first_token, first_pair].item()),
        actual_weight=None if first_token is None else float(actual_weights[first_token, first_pair].item()),
        maximum_absolute_weight_error=float(absolute_weight_error.max().item()),
        atol=atol,
        rtol=rtol,
    )
