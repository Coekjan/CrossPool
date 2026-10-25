"""Pure behavior tests for offline FFN memory profiling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from xpool import ffn
from xpool.config import XpoolConfig
from xpool.fabric import FABRIC_UID_HEX_LENGTH, FabricRole
from xpool.memory import (
    FfnMemoryCalibration,
    FfnMemoryCalibrationCoefficients,
    MemoryCalibrationDevice,
    MemoryCalibrationEnvironment,
    XpoolMemoryCalibrationProfile,
    load_memory_calibration_profile,
)
from xpool.native import ABI_VERSION
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent.device_memory import DeviceMemoryFeatures, DeviceMemoryPoint
from xpool.runtime.ffnagent.memory_profile import corpus, evidence, fitting, runner
from xpool.runtime.ffnagent.memory_profile.experiment import CalibrationExperiment
from xtest.harness.support.config import install_test_config, reset_global_config, synthetic_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def software_environment() -> evidence.MemoryProfileSoftwareEnvironment:
    """Return one self-consistent synthetic software environment."""

    return evidence.MemoryProfileSoftwareEnvironment(
        native_abi_version=ABI_VERSION,
        cuda_driver_version=1,
        cuda_runtime_version=1,
        torch_version="1",
        triton_version="1",
        sglang_version="1",
        sglang_kernel_version="1",
        nvshmem_version="1",
    )


def device(index: int) -> MemoryCalibrationDevice:
    """Return one ordered synthetic device record."""

    return MemoryCalibrationDevice(
        name="device",
        compute_capability=(8, 0),
        total_memory_bytes=100,
        uuid=f"device-{index}",
    )


def feature_row(index: int) -> DeviceMemoryFeatures:
    """Exercise one nonconstant coefficient feature per row."""

    return DeviceMemoryFeatures(
        tensor_storage_bytes=(1 << 20) if index == 1 else 0,
        tensor_storage_allocation_count=int(index == 2),
        dense_graph_capture_count=int(index == 3),
        moe_graph_capture_count=int(index == 4),
        executor_lane_count=int(index == 5),
        compute_branch_count=int(index == 6),
        dense_implementation_present=index == 7,
        moe_implementation_present=index == 8,
        joined_ffnagent=index == 9,
    )


def world(
    experiment: CalibrationExperiment,
    observations: tuple[evidence.MemoryObservation, ...],
    *,
    repetition: int = 1,
    participant_count: int = 1,
) -> evidence.MemoryProfileWorld:
    return evidence.MemoryProfileWorld(
        experiment=experiment,
        repetition=repetition,
        participants=tuple(
            evidence.MemoryProfileParticipantEvidence(
                ffnagent_index=index,
                device=device(index),
                environment=software_environment(),
                observations=observations,
            )
            for index in range(participant_count)
        ),
    )


def test_solve_coefficients_preserves_canonical_tie_break() -> None:
    coefficient_count = len(fitting.COEFFICIENT_NAMES)
    coefficients = fitting.solve_coefficients(
        {(1,) * coefficient_count: 14},
        observation_quantum_bytes=2,
    )

    assert coefficients.values() == (0,) * (coefficient_count - 1) + (14,)


@pytest.mark.parametrize("underpredict", (False, True))
def test_fit_worlds_covers_all_features_and_held_out_evidence(underpredict: bool) -> None:
    experiments, held_out_experiment = corpus.calibration_experiments(synthetic_config())
    observations = tuple(
        evidence.MemoryObservation(
            point=DeviceMemoryPoint(
                point=str(index),
                exact_resource_ledger_bytes=0,
                allocator_allowance_bytes=2,
                features=feature_row(index),
            ),
            observed_device_bytes=2 * (index + 1),
        )
        for index in range(10)
    )
    fit = world(experiments[0], observations)
    held_out = world(
        held_out_experiment,
        (
            evidence.MemoryObservation(
                point=observations[0].point,
                observed_device_bytes=5 if underpredict else 2,
            ),
        ),
    )

    if underpredict:
        with pytest.raises(RuntimeError, match="underpredicts held-out experiment by 1 bytes"):
            fitting.fit_worlds((fit,), (held_out,))
        return

    coefficients, headroom = fitting.fit_worlds((fit,), (held_out,))

    assert all(value >= 0 for value in coefficients.values())
    assert all(value % 2 == 0 for value in coefficients.values())
    assert coefficients.base_bytes == 2
    assert headroom == 2


@pytest.mark.parametrize(
    ("ffn_devices", "packed_tp_sizes"),
    (
        ((1,), (1,)),
        ((1, 2), (2,)),
        ((1, 2, 3), (2, 3)),
        ((1, 2, 3, 4), (2, 4)),
    ),
)
def test_profile_ffn_memory_runs_fixed_complete_fleet_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ffn_devices: tuple[int, ...],
    packed_tp_sizes: tuple[int, ...],
) -> None:
    config = synthetic_config(ffn_devices=ffn_devices)
    config = config.model_copy(
        update={"ffn": config.ffn.model_copy(update={"device_memory_calibration": tmp_path / "profile.json"})}
    )
    install_test_config(config)
    experiments, held_out_experiment = corpus.calibration_experiments(config)
    measured: list[tuple[CalibrationExperiment, int]] = []
    fitted: list[evidence.MemoryProfileWorld] = []
    validated: list[evidence.MemoryProfileWorld] = []

    def measure(
        experiment: CalibrationExperiment, source: XpoolConfig, *, repetition: int
    ) -> evidence.MemoryProfileWorld:
        measured.append((experiment, repetition))
        return world(experiment, (), repetition=repetition, participant_count=len(source.ffn.devices))

    coefficients = FfnMemoryCalibrationCoefficients(*([1] * 10))

    def fit(
        fit_worlds: tuple[evidence.MemoryProfileWorld, ...],
        held_out_worlds: tuple[evidence.MemoryProfileWorld, ...],
    ) -> tuple[FfnMemoryCalibrationCoefficients, int]:
        fitted.extend(fit_worlds)
        validated.extend(held_out_worlds)
        return coefficients, 7

    monkeypatch.setattr(runner, "refuse_live_daemon", lambda config: None)
    monkeypatch.setattr(runner, "run_world", measure)
    monkeypatch.setattr(runner, "fit_worlds", fit)

    result = runner.profile_ffn_memory()
    profile = result.profile

    packed_spec = held_out_experiment.models[-1].spec
    assert (
        tuple(
            model.ffn_tp_size for experiment in experiments for model in experiment.models if model.spec == packed_spec
        )
        == packed_tp_sizes
    )
    assert all(1 <= model.ffn_tp_size <= len(ffn_devices) for exp in experiments for model in exp.models)
    assert measured == [
        (experiment, repetition) for experiment in (*experiments, held_out_experiment) for repetition in range(1, 4)
    ]
    assert profile.environment.ffnagent_devices == tuple(device(index) for index in range(len(ffn_devices)))
    assert profile.ffn.minimum_held_out_headroom_bytes == 7
    assert profile.ffn.coefficients == coefficients
    assert tuple(fitted) == result.fit_worlds
    assert tuple(validated) == result.held_out_worlds
    assert [(row.experiment, row.repetition) for row in result.fit_worlds] == [
        (experiment, repetition) for experiment in experiments for repetition in range(1, 4)
    ]
    assert [(row.experiment, row.repetition) for row in result.held_out_worlds] == [
        (held_out_experiment, repetition) for repetition in range(1, 4)
    ]


@pytest.mark.parametrize("ffn_devices", ((2, 3), (2, 3, 4, 5)))
def test_experiment_projections_preserve_models_placement_and_complete_fleet(
    tmp_path: Path,
    ffn_devices: tuple[int, ...],
) -> None:
    source = synthetic_config(atn_devices=(0, 1), ffn_devices=ffn_devices)
    fit, held_out = corpus.calibration_experiments(source)

    for index, experiment in enumerate((*fit, held_out)):
        path = tmp_path / f"world-{index}.toml"
        environment = runner.write_world_config(path, source, experiment)
        config = XpoolConfig.from_file(path, env=environment)
        plan = experiment.fabric_plan("a" * FABRIC_UID_HEX_LENGTH, config=config)
        definition = experiment.to_json_mapping()

        assert config.scheduler.slo == source.scheduler.slo
        assert config.atn.devices == source.atn.devices
        assert config.ffn.devices == source.ffn.devices
        assert tuple(model.id for model in config.models) == tuple(spec.model_id for spec in experiment.model_specs)
        assert tuple(model.ffn_tp_size for model in config.models) == tuple(
            model.ffn_tp_size for model in experiment.models
        )
        assert all(config.atn_tp_size_of(model.id) == 2 for model in config.models)
        assert tuple(placement.device for placement in plan.pe_placements) == source.devices
        assert tuple(placement.role for placement in plan.pe_placements) == (
            FabricRole.ATNAGENT,
            FabricRole.ATNAGENT,
        ) + (FabricRole.FFNAGENT,) * len(ffn_devices)
        assert definition == {
            "models": [
                {
                    "spec": model.spec.model_dump(mode="json"),
                    "ffn_tp_size": model.ffn_tp_size,
                    "instance_plan": instance.model_dump(mode="json"),
                }
                for model, instance in zip(experiment.models, plan.instance_plans, strict=True)
            ]
        }
        for model, model_plan, instance in zip(experiment.models, plan.model_plans, plan.instance_plans, strict=True):
            assert model_plan.model_spec_digest == model.spec.digest()
            assert all(layer.ffnagent_indices == tuple(range(model.ffn_tp_size)) for layer in model_plan.layers)
            assert instance.instance_rank_topology == model.topology
            assert instance.ffn_profile.payload_dtype is torch.bfloat16
            assert instance.ffn_profile.decode_payload_row_capacity == 2048
            assert instance.ffn_profile.prefill_payload_row_capacity == 4096
            assert instance.ffn_profile.group_sum_complete_admitted == model.group_sum_complete_admitted
    assert all(model.topology.atnagent_indices == (0,) for experiment in fit for model in experiment.models)
    assert all(model.topology.atnagent_indices == (0, 1) for model in held_out.models)


def test_experiment_snapshot_keeps_resolved_debug_and_omits_calibration_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = XpoolConfig.from_mapping(
        synthetic_config().to_config_mapping(),
        env={"XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1", "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(tmp_path)},
    )
    monkeypatch.setenv("XPOOL_FFN_DEVICE_MEMORY_CALIBRATION", str(tmp_path / "unreadable.json"))
    monkeypatch.setenv("XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE", "0")
    path = tmp_path / "world.toml"
    experiments, _ = corpus.calibration_experiments(source)
    environment = runner.write_world_config(path, source, experiments[0])
    config = runner.init_global_config(config_path=path, env=environment)

    assert config.ffn.device_memory_calibration is None
    assert load_memory_calibration_profile() is None
    assert config.debug.graph_observer == source.debug.graph_observer


def candidate(
    fit_worlds: tuple[evidence.MemoryProfileWorld, ...],
    held_out_worlds: tuple[evidence.MemoryProfileWorld, ...] = (),
) -> evidence.MemoryProfileEvidence:
    return evidence.MemoryProfileEvidence(
        profile=XpoolMemoryCalibrationProfile(
            environment=MemoryCalibrationEnvironment(
                ffnagent_devices=fit_worlds[0].devices, **asdict(fit_worlds[0].environment)
            ),
            ffn=FfnMemoryCalibration(
                atnagent_count=1,
                executor_lane_count=1,
                minimum_held_out_headroom_bytes=0,
                coefficients=FfnMemoryCalibrationCoefficients(*([1] * 10)),
            ),
        ),
        fit_worlds=fit_worlds,
        held_out_worlds=held_out_worlds,
    )


def test_evidence_preserves_world_rank_phase_and_before_after_predictions() -> None:
    experiments, held_out_experiment = corpus.calibration_experiments(synthetic_config(ffn_devices=(1, 2)))
    observations = tuple(
        evidence.MemoryObservation(point=DeviceMemoryPoint(phase, 10, 7, feature_row(0)), observed_device_bytes=18)
        for phase in ("weight_materialization", "retained")
    )
    result = candidate(
        (world(experiments[0], observations, participant_count=2),),
        (world(held_out_experiment, observations, repetition=2, participant_count=2),),
    )
    payload = result.to_json_mapping()
    assert payload["profile"] == result.profile.model_dump(mode="json")
    retained = json.loads(json.dumps(payload))
    for role, worlds in (("fit_worlds", result.fit_worlds), ("held_out_worlds", result.held_out_worlds)):
        assert len(retained[role]) == len(worlds)
        for definition, measured in zip(retained[role], worlds, strict=True):
            assert definition["experiment"] == measured.experiment.to_json_mapping()
            assert definition["repetition"] == measured.repetition
            assert [row["ffnagent_index"] for row in definition["participants"]] == [0, 1]
            for participant in definition["participants"]:
                rows = participant["observations"]
                assert [row["point"] for row in rows] == [asdict(observation.point) for observation in observations]
                assert all(row["analytic_prediction_bytes"] == 17 for row in rows)
                assert all(row["calibrated_prediction_bytes"] == row["observed_device_bytes"] == 18 for row in rows)


@pytest.mark.parametrize("fail_replace", (False, True))
def test_evidence_publication_keeps_the_runtime_profile_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_replace: bool
) -> None:
    config = synthetic_config()
    profile_path = tmp_path / "memory.json"
    evidence_path = tmp_path / "memory.evidence.json"
    config = config.model_copy(
        update={"ffn": config.ffn.model_copy(update={"device_memory_calibration": profile_path})}
    )
    install_test_config(config)
    profile_path.write_bytes(b"previous profile")
    evidence_path.write_bytes(b"previous evidence")
    experiments, _ = corpus.calibration_experiments(config)
    result = candidate((world(experiments[0], ()),))

    if fail_replace:

        def reject_publication(path: Path, text: str) -> None:
            raise OSError("publication denied")

        monkeypatch.setattr(evidence, "atomic_write_text", reject_publication)
        with pytest.raises(RuntimeError, match="publication denied"):
            evidence.write_memory_profile_evidence(result)
        assert evidence_path.read_bytes() == b"previous evidence"
    else:
        assert evidence.write_memory_profile_evidence(result) == evidence_path
        assert json.loads(evidence_path.read_text()) == result.to_json_mapping()
    assert profile_path.read_bytes() == b"previous profile"


def test_calibration_corpus_uses_one_copy_of_each_signature_shape() -> None:
    fit, held_out = corpus.calibration_experiments(synthetic_config(ffn_devices=(1, 2)))
    members = tuple(experiment.models[0].spec for experiment in fit[:4])
    assert tuple(model.ffn_tp_size for experiment in fit for model in experiment.models) == (1, 1, 2, 2, 2, 2)
    assert fit[4].model_specs == (members[0],)
    assert held_out.model_specs == (members[3], members[0], fit[-1].models[0].spec)
    packed_layer = fit[-1].models[0].spec.layers[0]
    assert isinstance(packed_layer, ffn.MoeFfnSpec)
    assert packed_layer.topk_weight_rounding is ffn.TopKWeightRounding.LOGITS_DTYPE
    assert tuple(model.ffn_tp_size for model in held_out.models) == (2, 2, 2)
    assert tuple(model.group_sum_complete_admitted for model in held_out.models) == (True, False, True)

    assert tuple(tuple(layer.kind for layer in member.layers) for member in members) == (
        (LayerKind.DENSE,),
        (LayerKind.DENSE, LayerKind.MOE),
        (LayerKind.DENSE, LayerKind.MOE),
        (LayerKind.MOE,),
    )
    dense = members[0].layers[0]
    softmax_dense = members[1].layers[0]
    softmax_moe = members[1].layers[1]
    corrected_dense = members[2].layers[0]
    corrected_moe = members[2].layers[1]
    moe128 = members[3].layers[0]
    assert isinstance(dense, ffn.DenseFfnSpec)
    assert isinstance(softmax_dense, ffn.DenseFfnSpec)
    assert isinstance(softmax_moe, ffn.MoeFfnSpec)
    assert isinstance(corrected_dense, ffn.DenseFfnSpec)
    assert isinstance(corrected_moe, ffn.MoeFfnSpec)
    assert isinstance(moe128, ffn.MoeFfnSpec)
    assert (members[0].hidden_size, dense.intermediate_size) == (5120, 17408)
    assert (
        (softmax_moe.router_algorithm, softmax_moe.router_projection_precision),
        (corrected_moe.router_algorithm, corrected_moe.router_projection_precision),
        (moe128.router_algorithm, moe128.router_projection_precision),
    ) == (
        (ffn.RouterAlgorithm.SOFTMAX, ffn.RouterProjectionPrecision.FP32_LOGITS),
        (ffn.RouterAlgorithm.SIGMOID, ffn.RouterProjectionPrecision.FP32),
        (ffn.RouterAlgorithm.SOFTMAX, ffn.RouterProjectionPrecision.PAYLOAD),
    )
    assert (
        members[1].hidden_size,
        softmax_dense.intermediate_size,
        softmax_moe.expert_intermediate_size,
        softmax_moe.routed_expert_count,
        softmax_moe.shared_expert_count,
        softmax_moe.routed_topk,
        softmax_moe.router_correction_bias_key is not None,
        softmax_moe.renormalize,
        softmax_moe.routed_scaling_factor,
    ) == (2048, 10944, 1408, 64, 2, 6, False, False, 1.0)
    assert (
        members[2].hidden_size,
        corrected_dense.intermediate_size,
        corrected_moe.expert_intermediate_size,
        corrected_moe.routed_expert_count,
        corrected_moe.shared_expert_count,
        corrected_moe.routed_topk,
        corrected_moe.router_correction_bias_key is not None,
        corrected_moe.renormalize,
        corrected_moe.routed_scaling_factor,
    ) == (2048, 10240, 1536, 64, 1, 4, True, True, 1.8)
    assert (
        members[3].hidden_size,
        moe128.expert_intermediate_size,
        moe128.routed_expert_count,
        moe128.shared_expert_count,
        moe128.routed_topk,
        moe128.router_correction_bias_key is not None,
        moe128.renormalize,
        moe128.routed_scaling_factor,
    ) == (2048, 768, 128, 0, 8, False, True, 1.0)
