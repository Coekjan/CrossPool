"""CUDA behavior tests for canonical FFN true-TP checkpoint packing."""

from __future__ import annotations

import json
import math
import queue
import threading
from pathlib import Path
from typing import Literal, Protocol, cast

import pytest
import safetensors.torch
import torch
from xpool.native.ffn import LayerKind

import xtest
from xpool.config import XpoolConfig
from xpool.ffn import (
    ActivationKind,
    DenseFfnSpec,
    GatedFfnCheckpointKeys,
    MoeFfnCheckpointKeys,
    MoeFfnSpec,
    Mxfp4GatedFfnCheckpoint,
)
from xpool.quantization import MXFP4_DECODE_BLOCK_LIMIT, Mxfp4TensorSource
from xpool.runtime.ffnagent import architecture, execution, loader, operators, weights
from xtest.harness.support.config import TEST_MODEL_ID, install_test_config, reset_global_config

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def install_loader_config() -> None:
    """Install a two-reader test config without changing production defaults."""

    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1], "loader": {"parallelism": 2}},
                "models": [{"id": str(TEST_MODEL_ID), "path": "/models/model"}],
            }
        )
    )


def write_checkpoint(path: Path, shards: tuple[dict[str, torch.Tensor], ...]) -> None:
    """Write one indexed multi-file checkpoint for a packing test."""

    path.mkdir()
    weight_map: dict[str, str] = {}
    for index, tensors in enumerate(shards, start=1):
        shard_name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        safetensors.torch.save_file(tensors, path / shard_name)
        weight_map.update(dict.fromkeys(tensors, shard_name))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def dense_spec() -> DenseFfnSpec:
    """Return one small canonical Dense layer."""

    return DenseFfnSpec(
        kind=LayerKind.DENSE,
        layer_id=0,
        intermediate_size=6,
        checkpoint=architecture.gated_checkpoint_keys("model.layers.0.mlp"),
    )


def moe_spec(router_correction_bias_key: str) -> MoeFfnSpec:
    """Return one small corrected-routing MoE layer with two shared Experts."""

    prefix = "model.layers.1.mlp"
    return MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=1,
        expert_intermediate_size=4,
        shared_expert_count=2,
        routed_topk=2,
        renormalize=True,
        routed_scaling_factor=1.8,
        checkpoint=MoeFfnCheckpointKeys(
            router_weight_key=f"{prefix}.gate.weight",
            router_correction_bias_key=router_correction_bias_key,
            routed_experts=tuple(
                architecture.gated_checkpoint_keys(f"{prefix}.experts.{expert_id}") for expert_id in range(2)
            ),
            shared_expert=architecture.gated_checkpoint_keys(f"{prefix}.shared_experts"),
        ),
    )


def tensor_values(shape: tuple[int, ...], start: int) -> torch.Tensor:
    """Return recognizable BF16 source values."""

    return torch.arange(start, start + torch.Size(shape).numel(), dtype=torch.float32).reshape(shape).to(torch.bfloat16)


@xtest.requirements(device_count=1)
def test_materialize_packs_dense_and_moe_in_stable_request_order(tmp_path: Path) -> None:
    install_loader_config()
    hidden_size = 4
    dense = dense_spec()
    assert isinstance(dense.checkpoint, GatedFfnCheckpointKeys)
    correction_bias_key = "model.layers.1.mlp.gate.e_score_correction_bias"
    moe = moe_spec(correction_bias_key)
    dense_tensors = {
        dense.checkpoint.gate_weight_key: tensor_values((6, hidden_size), 0),
        dense.checkpoint.up_weight_key: tensor_values((6, hidden_size), 100),
        dense.checkpoint.down_weight_key: tensor_values((hidden_size, 6), 200),
    }
    moe_tensors: dict[str, torch.Tensor] = {
        moe.checkpoint.router_weight_key: torch.linspace(0.101, 0.909, 2 * hidden_size).reshape(2, hidden_size),
        correction_bias_key: torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    for expert_id, expert in enumerate(moe.checkpoint.routed_experts):
        assert isinstance(expert, GatedFfnCheckpointKeys)
        moe_tensors[expert.gate_weight_key] = tensor_values((4, hidden_size), 400 + expert_id * 100)
        moe_tensors[expert.up_weight_key] = tensor_values((4, hidden_size), 500 + expert_id * 100)
        moe_tensors[expert.down_weight_key] = tensor_values((hidden_size, 4), 600 + expert_id * 100)
    shared = moe.checkpoint.shared_expert
    assert isinstance(shared, GatedFfnCheckpointKeys)
    moe_tensors[shared.gate_weight_key] = tensor_values((8, hidden_size), 800)
    moe_tensors[shared.up_weight_key] = tensor_values((8, hidden_size), 900)
    moe_tensors[shared.down_weight_key] = tensor_values((hidden_size, 8), 1000)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (dense_tensors, moe_tensors))

    materialized = loader.materialize_local_layer_weights(
        requests=(
            loader.LocalLayerWeightRequest(
                model_path=model_path,
                hidden_size=hidden_size,
                payload_dtype=torch.float16,
                router_weight_dtype=None,
                layer=dense,
                tp_rank=1,
                tp_size=2,
            ),
            loader.LocalLayerWeightRequest(
                model_path=model_path,
                hidden_size=hidden_size,
                payload_dtype=torch.bfloat16,
                router_weight_dtype=torch.float32,
                layer=moe,
                tp_rank=0,
                tp_size=2,
            ),
        )
    )

    dense_weights, moe_weights = materialized
    assert isinstance(dense_weights, weights.DenseFfnWeights)
    assert isinstance(moe_weights, weights.MoeFfnWeights)
    assert dense_weights.gate_up_weight.device.index == torch.cuda.current_device()
    assert dense_weights.gate_up_weight.dtype is torch.float16
    torch.testing.assert_close(
        dense_weights.gate_up_weight.cpu(),
        torch.cat(
            (dense_tensors[dense.checkpoint.gate_weight_key][3:], dense_tensors[dense.checkpoint.up_weight_key][3:])
        ).to(torch.float16),
    )
    torch.testing.assert_close(
        dense_weights.down_weight.cpu(),
        dense_tensors[dense.checkpoint.down_weight_key][:, 3:].to(torch.float16),
    )
    assert moe_weights.router is not None
    torch.testing.assert_close(moe_weights.router.weight.cpu(), moe_tensors[moe.checkpoint.router_weight_key])
    assert moe_weights.router.correction_bias is not None
    torch.testing.assert_close(
        moe_weights.router.correction_bias.cpu(),
        moe_tensors[correction_bias_key],
    )
    expected_shared_gate = moe_tensors[shared.gate_weight_key].reshape(2, 4, hidden_size)[:, :2]
    torch.testing.assert_close(moe_weights.expert_gate_up_weight.cpu()[2:, :2], expected_shared_gate)
    expected_shared_down = moe_tensors[shared.down_weight_key].reshape(hidden_size, 2, 4)[:, :, :2].permute(1, 0, 2)
    torch.testing.assert_close(moe_weights.expert_down_weight.cpu()[2:], expected_shared_down)


@xtest.requirements(device_count=1)
def test_materialize_non_router_rank_and_shape_failure(tmp_path: Path) -> None:
    install_loader_config()
    correction_bias_key = "model.layers.1.mlp.gate.e_score_correction_bias"
    layer = moe_spec(correction_bias_key)
    hidden_size = 4
    tensors: dict[str, torch.Tensor] = {}
    for expert_id, expert in enumerate(layer.checkpoint.routed_experts):
        assert isinstance(expert, GatedFfnCheckpointKeys)
        tensors[expert.gate_weight_key] = tensor_values((4, hidden_size), expert_id * 100)
        tensors[expert.up_weight_key] = tensor_values((4, hidden_size), 200 + expert_id * 100)
        tensors[expert.down_weight_key] = tensor_values((hidden_size, 4), 400 + expert_id * 100)
    shared = layer.checkpoint.shared_expert
    assert isinstance(shared, GatedFfnCheckpointKeys)
    tensors[shared.gate_weight_key] = tensor_values((8, hidden_size), 600)
    tensors[shared.up_weight_key] = tensor_values((8, hidden_size), 700)
    tensors[shared.down_weight_key] = tensor_values((hidden_size, 8), 800)
    tensors[layer.checkpoint.router_weight_key] = tensor_values((2, hidden_size), 900)
    tensors[correction_bias_key] = torch.ones(3, dtype=torch.float32)
    model_path = tmp_path / "model"
    write_checkpoint(model_path, (tensors,))
    request = loader.LocalLayerWeightRequest(
        model_path=model_path,
        hidden_size=hidden_size,
        payload_dtype=torch.bfloat16,
        router_weight_dtype=torch.bfloat16,
        layer=layer,
        tp_rank=1,
        tp_size=2,
    )

    (materialized,) = loader.materialize_local_layer_weights(requests=(request,))
    assert isinstance(materialized, weights.MoeFfnWeights)
    assert materialized.router is None

    bad_request = loader.LocalLayerWeightRequest(
        model_path=model_path,
        hidden_size=hidden_size,
        payload_dtype=torch.bfloat16,
        router_weight_dtype=torch.bfloat16,
        layer=layer,
        tp_rank=0,
        tp_size=2,
    )
    with pytest.raises(RuntimeError, match=r"expected floating-point \(2,\), found torch.float32 \(3,\)"):
        loader.materialize_local_layer_weights(requests=(bad_request,))


def encoded_sources(
    prefix: str,
    *,
    experts: int | None = None,
    intermediate: int = 96,
    hidden: int = 64,
) -> tuple[tuple[Mxfp4GatedFfnCheckpoint, ...], dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    expert_prefix = () if experts is None else (experts,)
    gate_shape = (2 * intermediate, hidden) if experts is None else (experts, 2 * intermediate, hidden)
    down_shape = (hidden, intermediate) if experts is None else (experts, hidden, intermediate)
    for projection, shape in (
        ("gate_up", (*expert_prefix, 2 * intermediate, hidden)),
        ("down", (*expert_prefix, hidden, intermediate)),
    ):
        blocks_shape = (*shape[:-1], shape[-1] // 32, 16)
        scales_shape = blocks_shape[:-1]
        tensors[f"{prefix}.{projection}.blocks"] = (
            torch.arange(math.prod(blocks_shape), dtype=torch.int64).remainder(256).to(torch.uint8).view(blocks_shape)
        )
        tensors[f"{prefix}.{projection}.scales"] = (
            torch.arange(math.prod(scales_shape), dtype=torch.int64)
            .remainder(3)
            .add(123)
            .to(torch.uint8)
            .view(scales_shape)
        )
    triplets = []
    for expert_index in (None,) if experts is None else range(experts):
        triplets.append(
            Mxfp4GatedFfnCheckpoint(
                kind="mxfp4",
                gate=Mxfp4TensorSource(
                    blocks_key=f"{prefix}.gate_up.blocks",
                    scales_key=f"{prefix}.gate_up.scales",
                    decoded_shape=gate_shape,
                    expert_index=expert_index,
                    row_part="even",
                ),
                up=Mxfp4TensorSource(
                    blocks_key=f"{prefix}.gate_up.blocks",
                    scales_key=f"{prefix}.gate_up.scales",
                    decoded_shape=gate_shape,
                    expert_index=expert_index,
                    row_part="odd",
                ),
                down=Mxfp4TensorSource(
                    blocks_key=f"{prefix}.down.blocks",
                    scales_key=f"{prefix}.down.scales",
                    decoded_shape=down_shape,
                    expert_index=expert_index,
                    row_part="all",
                ),
            )
        )
    return tuple(triplets), tensors


def independently_decode(
    source: Mxfp4TensorSource, tensors: dict[str, torch.Tensor], dtype: torch.dtype
) -> torch.Tensor:
    magnitudes = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    blocks = tensors[source.blocks_key].reshape(-1, 16).tolist()
    scales = tensors[source.scales_key].reshape(-1).tolist()
    values = []
    for block, scale in zip(blocks, scales, strict=True):
        for byte in block:
            for code in (byte % 16, byte // 16):
                values.append(math.ldexp((-1.0 if code >= 8 else 1.0) * magnitudes[code % 8], scale - 127))
    decoded = torch.tensor(values, dtype=torch.float64).to(dtype).view(source.decoded_shape)
    if source.expert_index is not None:
        decoded = decoded[source.expert_index]
    if source.row_part != "all":
        decoded = decoded[int(source.row_part == "odd") :: 2]
    return decoded


def encoded_request(
    path: Path,
    triplets: tuple[Mxfp4GatedFfnCheckpoint, ...],
    *,
    tp: int = 2,
    rank: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> loader.LocalLayerWeightRequest:
    layer = (
        DenseFfnSpec(kind=LayerKind.DENSE, layer_id=0, intermediate_size=96, checkpoint=triplets[0])
        if len(triplets) == 1
        else MoeFfnSpec(
            kind=LayerKind.MOE,
            layer_id=0,
            expert_intermediate_size=96,
            shared_expert_count=0,
            routed_topk=2,
            renormalize=True,
            routed_scaling_factor=1,
            checkpoint=MoeFfnCheckpointKeys(
                router_weight_key="model.layers.0.mlp.router",
                router_correction_bias_key=None,
                routed_experts=triplets,
                shared_expert=None,
            ),
        )
    )
    return loader.LocalLayerWeightRequest(
        model_path=path,
        hidden_size=64,
        payload_dtype=dtype,
        router_weight_dtype=torch.float32 if len(triplets) > 1 else None,
        layer=layer,
        tp_size=tp,
        tp_rank=rank,
    )


@pytest.mark.parametrize("kind", ("dense", "moe", "shared"))
@pytest.mark.parametrize(
    ("tp", "rank", "dtype", "split"),
    (
        (1, 0, torch.bfloat16, False),
        (2, 0, torch.float16, True),
        (2, 1, torch.bfloat16, True),
        (3, 1, torch.float16, False),
    ),
)
@xtest.requirements(device_count=1)
def test_encoded_materialization_exact_shards_and_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    tp: int,
    rank: int,
    dtype: torch.dtype,
    split: bool,
) -> None:
    install_loader_config()
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts", experts=None if kind == "dense" else 2)
    request = encoded_request(tmp_path / "encoded", triplets, tp=tp, rank=rank, dtype=dtype)
    shared_triplet = None
    if kind != "dense":
        tensors["model.layers.0.mlp.router"] = torch.randn((2, 64), dtype=torch.float32)
    if kind == "shared":
        shared, shared_tensors = encoded_sources("model.layers.0.mlp.shared", intermediate=192)
        shared_triplet = shared[0]
        tensors.update(shared_tensors)
        assert isinstance(request.layer, MoeFfnSpec)
        request = loader.LocalLayerWeightRequest(
            model_path=request.model_path,
            hidden_size=64,
            payload_dtype=dtype,
            router_weight_dtype=torch.float32,
            layer=MoeFfnSpec(
                kind=LayerKind.MOE,
                layer_id=0,
                expert_intermediate_size=96,
                shared_expert_count=2,
                routed_topk=2,
                renormalize=True,
                routed_scaling_factor=1,
                checkpoint=MoeFfnCheckpointKeys(
                    router_weight_key="model.layers.0.mlp.router",
                    router_correction_bias_key=None,
                    routed_experts=triplets,
                    shared_expert=shared_triplet,
                ),
            ),
            tp_size=tp,
            tp_rank=rank,
        )
    shards = (
        (tensors,)
        if not split
        else (
            {key: value for key, value in tensors.items() if key.endswith("blocks")},
            {key: value for key, value in tensors.items() if not key.endswith("blocks")},
        )
    )
    write_checkpoint(request.model_path, shards)
    cuda_allocations = []
    original_empty = torch.empty

    def recorded_empty(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        if isinstance(device, torch.device) and device.type == "cuda":
            cuda_allocations.append(size)
        return original_empty(size, dtype=dtype, device=device, pin_memory=pin_memory)

    monkeypatch.setattr(torch, "empty", recorded_empty)
    (result,) = loader.materialize_local_layer_weights(requests=(request,))
    assert len(cuda_allocations) == 2 + int(kind != "dense" and rank == 0)
    local = 96 // tp
    start = rank * local
    expected_gate_up, expected_down = [], []
    for source in triplets:
        expected_gate_up.append(
            torch.cat(
                (
                    independently_decode(source.gate, tensors, dtype)[start : start + local],
                    independently_decode(source.up, tensors, dtype)[start : start + local],
                )
            )
        )
        expected_down.append(independently_decode(source.down, tensors, dtype)[:, start : start + local])
    if shared_triplet is not None:
        for ordinal in range(2):
            offset = ordinal * 96 + start
            expected_gate_up.append(
                torch.cat(
                    (
                        independently_decode(shared_triplet.gate, tensors, dtype)[offset : offset + local],
                        independently_decode(shared_triplet.up, tensors, dtype)[offset : offset + local],
                    )
                )
            )
            expected_down.append(independently_decode(shared_triplet.down, tensors, dtype)[:, offset : offset + local])
    if isinstance(result, weights.DenseFfnWeights):
        gate_up, down = result.gate_up_weight, result.down_weight
        expected_gate, expected_d = expected_gate_up[0], expected_down[0]
    else:
        gate_up, down = result.expert_gate_up_weight, result.expert_down_weight
        expected_gate, expected_d = torch.stack(expected_gate_up), torch.stack(expected_down)
        assert (result.router is not None) == (rank == 0)
        if result.router is not None:
            assert result.router.weight.dtype is torch.float32
            torch.testing.assert_close(result.router.weight.cpu(), tensors["model.layers.0.mlp.router"], rtol=0, atol=0)
    torch.testing.assert_close(gate_up.cpu(), expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(down.cpu(), expected_d, rtol=0, atol=0)
    for tensor in (gate_up, down):
        assert tensor.dtype is dtype and tensor.device.type == "cuda" and tensor.is_contiguous()
        assert tensor.storage_offset() == 0 and tensor.nbytes == tensor.untyped_storage().nbytes()
    assert gate_up.nbytes + down.nbytes == len(expected_gate_up) * 3 * 64 * local * dtype.itemsize


@pytest.mark.parametrize("failure", ("missing", "dtype", "shape", "logical", "tp", "alias"))
def test_encoded_preflight_precedes_all_final_cuda_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts", experts=2 if failure == "alias" else None)
    request = encoded_request(tmp_path / "encoded", triplets)
    if failure == "missing":
        del tensors[triplets[0].gate.scales_key]
    elif failure == "dtype":
        tensors[triplets[0].gate.blocks_key] = tensors[triplets[0].gate.blocks_key].to(torch.int16)
    elif failure == "shape":
        tensors[triplets[0].gate.blocks_key] = tensors[triplets[0].gate.blocks_key][:-1].contiguous()
    elif failure == "logical":
        request = loader.LocalLayerWeightRequest(
            model_path=request.model_path,
            hidden_size=32,
            payload_dtype=request.payload_dtype,
            router_weight_dtype=None,
            layer=request.layer,
            tp_rank=1,
            tp_size=2,
        )
    elif failure == "tp":
        request = encoded_request(request.model_path, triplets, tp=5, rank=1)
    elif failure == "alias":
        request = encoded_request(request.model_path, (triplets[0], triplets[0]))
    write_checkpoint(request.model_path, (tensors,))

    def unexpected_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA allocation phase reached before preflight rejection")

    monkeypatch.setattr(loader, "prepare_layer_weight_request", unexpected_allocation)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected_allocation)
    with pytest.raises(RuntimeError) as error:
        loader.materialize_local_layer_weights(requests=(request,))
    assert isinstance(error.value.__cause__, ValueError)


def test_preflight_reads_only_metadata_and_includes_earlier_floating_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts")
    request = encoded_request(tmp_path / "encoded", triplets)
    write_checkpoint(request.model_path, (tensors,))
    original_open = loader.safe_open
    headers = set()

    class HeaderOnlyFile:
        def __init__(self, filename: str, *, framework: str, device: str) -> None:
            self.handle = original_open(filename, framework=framework, device=device)

        def __enter__(self) -> HeaderOnlyFile:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.__exit__(*args)

        def get_slice(self, key: str) -> object:
            headers.add(key)
            return self.handle.get_slice(key)

        def get_tensor(self, key: str) -> torch.Tensor:
            raise AssertionError(f"preflight materialized tensor {key}")

    monkeypatch.setattr(loader, "safe_open", HeaderOnlyFile)
    loader.preflight_local_sources(
        (request,),
        key_views={request.model_path: loader.checkpoint.read_checkpoint_key_view(request.model_path)},
    )
    assert headers == set(tensors)

    floating = loader.LocalLayerWeightRequest(
        model_path=request.model_path,
        hidden_size=4,
        payload_dtype=torch.bfloat16,
        router_weight_dtype=None,
        layer=DenseFfnSpec(
            kind=LayerKind.DENSE,
            layer_id=1,
            intermediate_size=6,
            checkpoint=architecture.gated_checkpoint_keys("model.layers.1.mlp"),
        ),
        tp_rank=0,
        tp_size=1,
    )
    corrupted = {key: value.clone() for key, value in tensors.items()}
    corrupted[triplets[0].gate.scales_key] = corrupted[triplets[0].gate.scales_key].float()
    safetensors.torch.save_file(corrupted, request.model_path / "model-00001-of-00001.safetensors")

    def unexpected_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("allocation reached before batch preflight completed")

    monkeypatch.setattr(loader, "prepare_layer_weight_request", unexpected_allocation)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected_allocation)
    with pytest.raises(RuntimeError, match="expected U8"):
        loader.materialize_local_layer_weights(requests=(floating, request))


@pytest.mark.parametrize("projection", ("gate", "up", "down"))
@pytest.mark.parametrize("stacked", (False, True))
def test_rectangular_reader_bounds_physical_tiles_and_handles_block_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    projection: str,
    stacked: bool,
) -> None:
    triplets, tensors = encoded_sources(
        "model.layers.0.mlp.experts", experts=2 if stacked else None, intermediate=96, hidden=8192
    )
    path = tmp_path / "encoded"
    write_checkpoint(path, (tensors,))
    shard = path / "model-00001-of-00001.safetensors"
    source = {"gate": triplets[-1].gate, "up": triplets[-1].up, "down": triplets[-1].down}[projection]
    destination = torch.empty((8192, 48) if projection == "down" else (48, 8192), dtype=torch.float16)
    descriptor = loader.Mxfp4TensorReadDescriptor(
        source=source,
        blocks_shard_path=shard,
        scales_shard_path=shard,
        narrow_dimension=int(projection == "down"),
        narrow_start=48,
        narrow_length=48,
        staged_shape=tuple(destination.shape),
        destination=destination,
        tp_rank=1,
    )
    observed = []
    physical_rows = []
    original = loader.decode_mxfp4_blocks
    original_open = loader.safe_open

    # Safetensors does not export a typed declaration for its opaque slice object.
    class SliceReader(Protocol):
        def __getitem__(self, index: tuple[int | slice, ...]) -> torch.Tensor: ...

    class SliceSpy:
        def __init__(self, value: SliceReader) -> None:
            self.value = value

        def __getitem__(self, index: tuple[int | slice, ...]) -> torch.Tensor:
            row_slice = index[int(stacked)]
            assert isinstance(row_slice, slice)
            physical_rows.append(row_slice.stop - row_slice.start)
            return self.value[index]

    class FileSpy:
        def __init__(self, filename: str, *, framework: str, device: str) -> None:
            self.handle = original_open(filename, framework=framework, device=device)

        def __enter__(self) -> FileSpy:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.__exit__(*args)

        def get_slice(self, key: str) -> SliceSpy:
            return SliceSpy(cast(SliceReader, self.handle.get_slice(key)))

        def get_tensor(self, key: str) -> torch.Tensor:
            raise AssertionError(f"encoded read requested complete tensor {key}")

    def bounded_decode(blocks: torch.Tensor, scales: torch.Tensor, *, out: torch.Tensor) -> None:
        observed.append(blocks.shape[0])
        assert blocks.shape[0] <= MXFP4_DECODE_BLOCK_LIMIT
        original(blocks, scales, out=out)

    monkeypatch.setattr(loader, "decode_mxfp4_blocks", bounded_decode)
    monkeypatch.setattr(loader, "safe_open", FileSpy)
    with loader.safe_open(str(shard), framework="pt", device="cpu") as handle:
        loader.stage_mxfp4_tensor(descriptor, handles={shard: handle}, out=destination, cancelled=threading.Event())
    assert len(observed) < destination.shape[0]
    assert max(observed) * (1 if projection == "down" else 2) <= MXFP4_DECODE_BLOCK_LIMIT
    assert min(physical_rows) > 1
    reference = independently_decode(source, tensors, destination.dtype)
    expected = reference[:, 48:] if projection == "down" else reference[48:]
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ("second_file", "read", "decode", "copy"))
def test_source_groups_restore_slots_and_join_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts")
    path = tmp_path / "encoded"
    write_checkpoint(
        path,
        (
            {key: value for key, value in tensors.items() if key.endswith("blocks")},
            {key: value for key, value in tensors.items() if key.endswith("scales")},
        ),
    )
    paths = tuple(sorted(path.glob("*.safetensors")))
    if failure == "second_file":
        paths[1].write_bytes(b"invalid header")
    elif failure == "decode":
        broken = {key: value.clone() for key, value in tensors.items() if key.endswith("scales")}
        broken[triplets[0].gate.scales_key].fill_(255)
        safetensors.torch.save_file(broken, paths[1])
    source = triplets[0].gate
    if failure == "read":
        source = source.model_copy(update={"blocks_key": "missing"})
    destination = torch.empty((96, 64), dtype=torch.float16)
    descriptor = loader.Mxfp4TensorReadDescriptor(
        source=source,
        blocks_shard_path=paths[0],
        scales_shard_path=paths[1],
        narrow_dimension=0,
        narrow_start=0,
        narrow_length=96,
        staged_shape=tuple(destination.shape),
        destination=destination,
        tp_rank=0,
    )
    if failure == "copy":
        original = torch.Tensor.copy_

        def failing_copy(self: torch.Tensor, other: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
            if self is destination:
                raise RuntimeError("injected destination copy failure")
            return original(self, other, non_blocking=non_blocking)

        monkeypatch.setattr(torch.Tensor, "copy_", failing_copy)
    slots: queue.Queue[torch.Tensor] = queue.Queue()
    slots.put(torch.empty(descriptor.staged_bytes, dtype=torch.uint8))
    threads_before = set(threading.enumerate())
    with pytest.raises(Exception) as error:
        loader.copy_staged_tensors(descriptor_groups={paths: [descriptor, descriptor]}, slots=slots, parallelism=2)
    if failure == "decode":
        assert isinstance(error.value, ValueError) and "reserved" in str(error.value)
    elif failure == "copy":
        assert isinstance(error.value, RuntimeError) and "injected" in str(error.value)
        notes = "; ".join(getattr(error.value, "__notes__", ()))
        assert "destination copy" in notes and source.blocks_key in notes
        assert "TP rank=0" in notes and str(paths[0]) in notes
    else:
        assert "SafetensorError" == type(error.value).__name__
    assert slots.qsize() == 1
    assert not (set(threading.enumerate()) - threads_before)


@pytest.mark.parametrize("row_part", ("all", "odd"))
def test_reader_tiles_group_axis_when_one_row_exceeds_block_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_part: Literal["all", "odd"],
) -> None:
    source = Mxfp4TensorSource(
        blocks_key="projection.blocks",
        scales_key="projection.scales",
        decoded_shape=(4, 32 * (MXFP4_DECODE_BLOCK_LIMIT + 1)),
        expert_index=None,
        row_part=row_part,
    )
    tensors = {
        source.blocks_key: torch.arange(math.prod(source.blocks_shape), dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .view(source.blocks_shape),
        source.scales_key: torch.full(source.scales_shape, 127, dtype=torch.uint8),
    }
    shard = tmp_path / "wide.safetensors"
    safetensors.torch.save_file(tensors, shard)
    destination = torch.empty(source.logical_shape, dtype=torch.bfloat16)
    descriptor = loader.Mxfp4TensorReadDescriptor(
        source=source,
        blocks_shard_path=shard,
        scales_shard_path=shard,
        narrow_dimension=None,
        narrow_start=0,
        narrow_length=0,
        staged_shape=tuple(destination.shape),
        destination=destination,
        tp_rank=0,
    )
    observed: list[int] = []
    original = loader.decode_mxfp4_blocks

    def bounded_decode(blocks: torch.Tensor, scales: torch.Tensor, *, out: torch.Tensor) -> None:
        observed.append(blocks.shape[0])
        assert blocks.shape[0] * (1 if row_part == "all" else 2) <= MXFP4_DECODE_BLOCK_LIMIT
        original(blocks, scales, out=out)

    monkeypatch.setattr(loader, "decode_mxfp4_blocks", bounded_decode)
    with loader.safe_open(str(shard), framework="pt", device="cpu") as handle:
        loader.stage_mxfp4_tensor(descriptor, handles={shard: handle}, out=destination, cancelled=threading.Event())
    assert len(observed) > 1
    assert sum(observed) == math.prod(source.logical_shape) // 32
    torch.testing.assert_close(destination, independently_decode(source, tensors, destination.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("phase", ("waiting", "decoding"))
def test_reader_cancellation_releases_occupied_slots_without_publishing_partial_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts", hidden=8192)
    path = tmp_path / "encoded"
    write_checkpoint(path, (tensors,))
    shard = next(path.glob("*.safetensors"))
    destination = torch.full((48, 8192), float("nan"), dtype=torch.float16)
    descriptor = loader.Mxfp4TensorReadDescriptor(
        source=triplets[0].gate,
        blocks_shard_path=shard,
        scales_shard_path=shard,
        narrow_dimension=0,
        narrow_start=48,
        narrow_length=48,
        staged_shape=tuple(destination.shape),
        destination=destination,
        tp_rank=1,
    )
    slot = torch.full((descriptor.staged_bytes,), 255, dtype=torch.uint8)
    slots: queue.Queue[torch.Tensor] = queue.Queue()
    slots.put(slot)
    messages: queue.Queue[loader.ReaderMessage] = queue.Queue()
    cancelled, entered, released = threading.Event(), threading.Event(), threading.Event()
    original_decode = loader.decode_mxfp4_blocks
    original_acquire = loader.acquire_slot

    def paused_decode(blocks: torch.Tensor, scales: torch.Tensor, *, out: torch.Tensor) -> None:
        original_decode(blocks, scales, out=out)
        assert slots.empty()
        entered.set()
        if not released.wait(timeout=5):
            raise TimeoutError("test did not release the occupied decoding slot")

    def observed_acquire(slots: queue.Queue[torch.Tensor], cancelled: threading.Event) -> torch.Tensor | None:
        entered.set()
        return original_acquire(slots, cancelled)

    if phase == "waiting":
        assert slots.get_nowait() is slot
        monkeypatch.setattr(loader, "acquire_slot", observed_acquire)
    else:
        monkeypatch.setattr(loader, "decode_mxfp4_blocks", paused_decode)
    reader = threading.Thread(
        target=loader.stage_source_group,
        kwargs={
            "shard_paths": (shard,),
            "descriptors": (descriptor, descriptor),
            "slots": slots,
            "messages": messages,
            "cancelled": cancelled,
        },
    )
    reader.start()
    try:
        assert entered.wait(timeout=5)
        cancelled.set()
    finally:
        cancelled.set()
        released.set()
        reader.join(timeout=5)
        if phase == "waiting":
            slots.put(slot)
    assert not reader.is_alive()
    assert slots.qsize() == 1 and slots.get_nowait() is slot
    assert isinstance(messages.get_nowait(), loader.ReaderFinished)
    assert messages.empty()
    assert bool(torch.isnan(destination).all())
    staged = loader.slot_view(slot, descriptor)
    assert bool(torch.isnan(staged).any())
    if phase == "decoding":
        assert bool(torch.isfinite(staged).any())


@xtest.requirements(device_count=1)
def test_materialization_copy_failure_preserves_cause_and_retires_all_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_loader_config()
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts")
    request = encoded_request(tmp_path / "encoded", triplets)
    write_checkpoint(request.model_path, (tensors,))
    injected = RuntimeError("injected CUDA destination copy failure")
    original_copy = torch.Tensor.copy_
    original_readers = loader.copy_staged_tensors
    copies = 0
    joined = False

    def failing_copy(self: torch.Tensor, other: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        nonlocal copies
        if self.device.type == "cuda":
            assert other.device.type == "cpu" and other.is_pinned()
            assert not non_blocking
            copies += 1
            if copies == 2:
                raise injected
        return original_copy(self, other, non_blocking=non_blocking)

    def verified_readers(
        *,
        descriptor_groups: dict[tuple[Path, ...], list[loader.WeightReadDescriptor]],
        slots: queue.Queue[torch.Tensor],
        parallelism: int,
    ) -> None:
        nonlocal joined
        initial_slots = slots.qsize()
        threads_before = set(threading.enumerate())
        try:
            original_readers(descriptor_groups=descriptor_groups, slots=slots, parallelism=parallelism)
        finally:
            assert slots.qsize() == initial_slots
            assert not (set(threading.enumerate()) - threads_before)
            joined = True

    monkeypatch.setattr(torch.Tensor, "copy_", failing_copy)
    monkeypatch.setattr(loader, "copy_staged_tensors", verified_readers)
    with pytest.raises(RuntimeError, match="injected CUDA destination copy failure") as error:
        loader.materialize_local_layer_weights(requests=(request,))
    assert error.value.__cause__ is injected
    assert copies == 2 and joined
    assert triplets[0].up.blocks_key in str(error.value)
    assert "TP rank=1" in str(error.value) and str(request.model_path) in str(error.value)


def test_second_file_open_failure_publishes_one_terminal_marker(tmp_path: Path) -> None:
    first = tmp_path / "a.safetensors"
    safetensors.torch.save_file({"x": torch.zeros(1)}, first)
    second = tmp_path / "b.safetensors"
    second.write_bytes(b"invalid header")
    messages: queue.Queue[loader.ReaderMessage] = queue.Queue()
    cancelled = threading.Event()
    loader.stage_source_group(
        shard_paths=(first, second),
        descriptors=(),
        slots=queue.Queue(),
        messages=messages,
        cancelled=cancelled,
    )
    assert cancelled.is_set()
    assert isinstance(messages.get_nowait(), loader.ReaderFailure)
    assert isinstance(messages.get_nowait(), loader.ReaderFinished)
    assert messages.empty()


@pytest.mark.parametrize("kind", ("dense", "moe"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@xtest.requirements(device_count=1)
def test_loaded_encoded_weights_use_existing_eager_and_captured_operators(
    tmp_path: Path,
    kind: str,
    dtype: torch.dtype,
) -> None:
    install_loader_config()
    triplets, tensors = encoded_sources("model.layers.0.mlp.experts", experts=2 if kind == "moe" else None)
    if kind == "moe":
        tensors["model.layers.0.mlp.router"] = torch.zeros((2, 64), dtype=torch.float32)
    request = encoded_request(tmp_path / "encoded", triplets, dtype=dtype)
    write_checkpoint(request.model_path, (tensors,))
    (loaded,) = loader.materialize_local_layer_weights(requests=(request,))
    hidden = torch.full((4, 64), 0.01, dtype=dtype, device="cuda")
    output = torch.empty_like(hidden)
    reference_gate_up = [
        torch.cat(
            (independently_decode(t.gate, tensors, dtype)[48:], independently_decode(t.up, tensors, dtype)[48:])
        ).cuda()
        for t in triplets
    ]
    reference_down = [independently_decode(t.down, tensors, dtype)[:, 48:].cuda() for t in triplets]
    if isinstance(loaded, weights.DenseFfnWeights):
        workspace = torch.empty(
            execution.dense_workspace_bytes(
                payload_dtype=dtype,
                row_capacity=4,
                local_intermediate_size=48,
            ),
            dtype=torch.uint8,
            device="cuda",
        )

        def launch() -> None:
            operators.compute_dense_partial(
                hidden_states=hidden,
                layer_weights=loaded,
                workspace=workspace,
                output=output,
                activation=ActivationKind.SILU,
            )

        route_weights = (1.0,)
    else:
        w13, w2 = operators.select_moe_kernel_configs(layer_weights=loaded, row_capacity=4, effective_topk=2)
        padded, expert_blocks, cumsum = execution.moe_alignment_workspace_shapes(
            row_capacity=4,
            effective_topk=2,
            expert_count=2,
            block_size_m=w13["BLOCK_SIZE_M"],
        )
        ids = torch.tensor(((0, 1),) * 4, dtype=torch.int32, device="cuda")
        routing_weights = torch.tensor(((0.25, 0.75),) * 4, dtype=torch.float32, device="cuda")
        sorted_ids = torch.empty(padded, dtype=torch.int32, device="cuda")
        expert_ids = torch.empty(expert_blocks, dtype=torch.int32, device="cuda")
        post_padded = torch.empty(1, dtype=torch.int32, device="cuda")
        cumsum_buffer = torch.empty(cumsum, dtype=torch.int32, device="cuda")
        gate_up = torch.empty((8, 96), dtype=dtype, device="cuda")
        activated = torch.empty((8, 48), dtype=dtype, device="cuda")
        route_outputs = torch.empty((4, 2, 64), dtype=dtype, device="cuda")

        def launch() -> None:
            operators.compute_moe_partial(
                hidden_states=hidden,
                layer_weights=loaded,
                topk_ids=ids,
                topk_weights=routing_weights,
                sorted_token_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=post_padded,
                cumsum_buffer=cumsum_buffer,
                gate_up=gate_up,
                activated=activated,
                route_outputs=route_outputs,
                output=output,
                w13_config=w13,
                w2_config=w2,
                activation=ActivationKind.SILU,
                routed_scaling_factor=1,
            )

        route_weights = (0.25, 0.75)

    def check_reference() -> None:
        routes = []
        for gate_up_ref, down_ref, probability in zip(reference_gate_up, reference_down, route_weights, strict=True):
            gate, up = torch.mm(hidden, gate_up_ref.t()).chunk(2, dim=1)
            activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)
            routes.append((torch.mm(activation.float(), down_ref.float().t()) * probability).to(dtype))
        expected = torch.stack(routes).float().sum(0).to(dtype)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)

    launch()
    check_reference()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    hidden.fill_(-0.02)
    graph.replay()
    check_reference()
