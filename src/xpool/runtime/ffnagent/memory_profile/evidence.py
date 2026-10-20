"""Attributed calibration measurements and candidate publication."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from xpool.config import get_global_config
from xpool.memory import FfnMemoryCalibrationCoefficients, MemoryCalibrationDevice, XpoolMemoryCalibrationProfile
from xpool.runtime.ffnagent.device_memory import DeviceMemoryPoint
from xpool.runtime.ffnagent.memory_profile.experiment import CalibrationExperiment
from xpool.utils.files import atomic_write_text


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    """One exact feature row and its observed baseline-relative device peak."""

    point: DeviceMemoryPoint
    observed_device_bytes: int


@dataclass(frozen=True, slots=True)
class MemoryProfileSoftwareEnvironment:
    """Software compatibility facts observed by every FfnAgent child."""

    native_abi_version: int
    cuda_driver_version: int
    cuda_runtime_version: int
    torch_version: str
    triton_version: str
    sglang_version: str
    sglang_kernel_version: str
    nvshmem_version: str


@dataclass(frozen=True, slots=True)
class MemoryProfileParticipantEvidence:
    """One FfnAgent child's complete ephemeral evidence."""

    ffnagent_index: int
    device: MemoryCalibrationDevice
    environment: MemoryProfileSoftwareEnvironment
    observations: tuple[MemoryObservation, ...]


@dataclass(frozen=True, slots=True)
class MemoryProfileWorld:
    """One experiment repetition with complete ordered participant measurements."""

    experiment: CalibrationExperiment
    repetition: int
    participants: tuple[MemoryProfileParticipantEvidence, ...]

    def __post_init__(self) -> None:
        if not self.participants or tuple(row.ffnagent_index for row in self.participants) != tuple(
            range(len(self.participants))
        ):
            raise ValueError("memory-profile evidence must contain ordered complete FfnAgent indices")
        if any(row.environment != self.environment for row in self.participants[1:]):
            raise RuntimeError("memory-profile participants disagree on the software environment")

    @property
    def devices(self) -> tuple[MemoryCalibrationDevice, ...]:
        return tuple(row.device for row in self.participants)

    @property
    def environment(self) -> MemoryProfileSoftwareEnvironment:
        return self.participants[0].environment

    @property
    def observations(self) -> tuple[MemoryObservation, ...]:
        return tuple(observation for row in self.participants for observation in row.observations)

    def to_json_mapping(self, coefficients: FfnMemoryCalibrationCoefficients) -> dict[str, object]:
        """Keep the definition once and project predictions for each rank and phase."""

        participants = []
        for participant in self.participants:
            observations = []
            for observation in participant.observations:
                point = observation.point
                analytic = point.exact_resource_ledger_bytes + point.allocator_allowance_bytes
                calibrated = analytic + coefficients.evaluate(**asdict(point.features))
                observations.append(
                    {
                        "point": asdict(point),
                        "observed_device_bytes": observation.observed_device_bytes,
                        "analytic_prediction_bytes": analytic,
                        "calibrated_prediction_bytes": calibrated,
                    }
                )
            participants.append({"ffnagent_index": participant.ffnagent_index, "observations": observations})
        return {
            "experiment": self.experiment.to_json_mapping(),
            "repetition": self.repetition,
            "participants": participants,
        }


@dataclass(frozen=True, slots=True)
class MemoryProfileEvidence:
    """One held-out-qualified candidate and its uncollapsed measurements."""

    profile: XpoolMemoryCalibrationProfile
    fit_worlds: tuple[MemoryProfileWorld, ...]
    held_out_worlds: tuple[MemoryProfileWorld, ...]

    def to_json_mapping(self) -> dict[str, object]:
        """Retain fitting role and the uncollapsed world, rank and phase evidence."""

        return {
            "profile": self.profile.model_dump(mode="json"),
            "fit_worlds": [world.to_json_mapping(self.profile.ffn.coefficients) for world in self.fit_worlds],
            "held_out_worlds": [world.to_json_mapping(self.profile.ffn.coefficients) for world in self.held_out_worlds],
        }


def write_memory_profile_evidence(evidence: MemoryProfileEvidence) -> Path:
    """Atomically publish the latest evidence beside the configured Profile.

    Returns the evidence path. A publication failure leaves the previous file
    untouched and raises RuntimeError; it does not publish the runtime Profile.
    """

    profile_path = get_global_config().ffn.device_memory_calibration
    if profile_path is None:
        raise RuntimeError("ffn.device_memory_calibration is required to write calibration evidence")
    path = profile_path.with_name(f"{profile_path.stem}.evidence.json")
    try:
        atomic_write_text(path, json.dumps(evidence.to_json_mapping(), indent=2, allow_nan=False) + "\n")
    except OSError as error:
        raise RuntimeError(f"failed to write memory calibration evidence {path}: {error}") from error
    return path
