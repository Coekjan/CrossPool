"""Independent GPT-OSS mask/sink evidence on the installed hybrid pool storage."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from pydantic import JsonValue
from safetensors import safe_open
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KvBufferDesc, KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.models.gpt_oss import get_attention_sliding_window_size
from transformers import GptOssConfig

from xkit.config import assemble_config
from xkit.results import write_json
from xpool import ecs
from xpool.integrations.sglang.ecs.pool import ElasticMHATokenToKVPool, prepared_backing_scope
from xpool.integrations.sglang.ecs.vmm import KvVmmBacking
from xpool.model import ModelId
from xpool.runtime.ffnagent.checkpoint import read_checkpoint_key_view
from xtest.harness.runner.requirements import ResolvedConfig


def run_attention_numerical(model_id: ModelId, base: ResolvedConfig, artifact_dir: Path) -> None:
    """Compare upstream paged attention with FP32 masked softmax and real sinks.

    Use the checkpoint's heads, window and layer mapping, with one original
    Full and one SWA layer. Synthetic Q/K/V isolate attention from FFN errors;
    nonidentity page tables and sparse compound views exercise actual storage.
    """

    config = assemble_config((model_id,), runtime_config=base.path)
    model_path = config.model_path_of(model_id)
    source = GptOssConfig.from_pretrained(model_path, local_files_only=True)
    if source.layer_types is None or source.sliding_window is None:
        raise ValueError("hybrid attention qualification requires layer types and a sliding window")
    checkpoint = read_checkpoint_key_view(model_path)
    full_layers = [index for index, kind in enumerate(source.layer_types) if kind == "full_attention"]
    swa_layers = [index for index, kind in enumerate(source.layer_types) if kind == "sliding_attention"]
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    page_size = 16
    kv_length = source.sliding_window + 2 * page_size + 3
    page_count = math.ceil(kv_length / page_size)
    layouts = tuple(
        KvVmmBacking.describe(
            device=device,
            storage_dtype=dtype,
            token_page_size=page_size,
            descriptors=tuple(
                KvBufferDesc(
                    f"buffer{index}",
                    (size + page_size, source.num_key_value_heads, source.head_dim),
                    row_bytes=source.num_key_value_heads * source.head_dim * dtype.itemsize,
                    tokens_per_row=1,
                )
                for index in range(2 * len(layers))
            ),
            buffers_per_layer=2,
        )
        for size, layers in ((8192, full_layers), (4096, swa_layers))
    )
    quota = ecs.MemoryQuota(sum(layout.profile.bytes_for(layout.profile.bundle_capacity) for layout in layouts))
    backings: list[KvVmmBacking] = []
    samples: list[JsonValue] = []
    try:
        for layout in layouts:
            backings.append(KvVmmBacking(layout, quota))
        with prepared_backing_scope(tuple(backings)):
            pool = SWAKVPool(
                size=8192,
                size_swa=4096,
                page_size=page_size,
                dtype=dtype,
                head_num=source.num_key_value_heads,
                head_dim=source.head_dim,
                swa_attention_layer_ids=swa_layers,
                full_attention_layer_ids=full_layers,
                device=str(device),
                full_kv_pool_class=ElasticMHATokenToKVPool,
                swa_kv_pool_class=ElasticMHATokenToKVPool,
            )
        full_pages = torch.arange(130, 130 + 2 * page_count, 2, device=device)
        swa_pages = torch.arange(130 + page_count, 130, -1, device=device)
        offsets = torch.arange(page_size, device=device)
        full_locations = (full_pages[:, None] * page_size + offsets).flatten()[:kv_length]
        swa_locations = (swa_pages[:, None] * page_size + offsets).flatten()[:kv_length]
        mapping = torch.full((pool.size + page_size + 1,), -1, dtype=torch.int64, device=device)
        mapping[full_locations] = swa_locations
        pool.register_mapping(mapping)
        for backing, pages in zip(backings, (full_pages, swa_pages), strict=True):
            starts, ends = backing.bundle_spans(pages)
            bundle_ids = tuple(
                sorted({index for start, end in zip(starts.tolist(), ends.tolist()) for index in range(start, end)})
            )
            assert backing.try_map(bundle_ids)

        generator = torch.Generator().manual_seed(59)
        keys = torch.randn(kv_length, source.num_key_value_heads, source.head_dim, generator=generator).to(dtype)
        values = torch.randn(keys.shape, generator=generator).to(dtype)
        query = torch.randn(kv_length, source.num_attention_heads, source.head_dim, generator=generator).to(dtype)
        groups = source.num_attention_heads // source.num_key_value_heads
        reference_keys = keys.float().repeat_interleave(groups, dim=1)
        reference_values = values.float().repeat_interleave(groups, dim=1)
        positions = torch.arange(kv_length)
        for layer_id, sliding in ((full_layers[0], False), (swa_layers[0], True)):
            sink_key = f"model.layers.{layer_id}.self_attn.sinks"
            with safe_open(checkpoint[sink_key], framework="pt", device="cpu") as shard:
                sinks = shard.get_tensor(sink_key).to(dtype)
            window = get_attention_sliding_window_size(source) if sliding else -1
            layer = RadixAttention(
                num_heads=source.num_attention_heads,
                num_kv_heads=source.num_key_value_heads,
                head_dim=source.head_dim,
                scaling=source.head_dim**-0.5,
                layer_id=layer_id,
                sliding_window_size=window,
            )
            pool.set_kv_buffer(
                layer,
                KVWriteLoc(full_locations, swa_loc=pool.translate_loc_from_full_to_swa(full_locations)),
                keys.to(device),
                values.to(device),
            )
            key_cache, value_cache = pool.get_kv_buffer(layer_id)
            assert not key_cache.is_contiguous()
            page_table = (swa_pages if sliding else full_pages).to(torch.int32).unsqueeze(0)
            for query_length in (1, 17, kv_length):
                q = query[-query_length:]
                scores = torch.einsum("qhd,khd->hqk", q.float(), reference_keys) * source.head_dim**-0.5
                query_positions = positions[-query_length:, None]
                mask = positions <= query_positions
                if sliding:
                    mask &= positions > query_positions - source.sliding_window
                scores.masked_fill_(~mask.unsqueeze(0), -torch.inf)
                sink_scores = sinks.float()[:, None, None].expand(-1, query_length, 1)
                probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[..., :-1]
                expected = torch.einsum("hqk,khd->qhd", probabilities, reference_values)
                actual = (
                    flash_attn_with_kvcache(
                        q=q.to(device),
                        k_cache=key_cache.view(-1, page_size, source.num_key_value_heads, source.head_dim),
                        v_cache=value_cache.view(-1, page_size, source.num_key_value_heads, source.head_dim),
                        page_table=page_table,
                        cache_seqlens=torch.tensor([kv_length], dtype=torch.int32, device=device),
                        cu_seqlens_q=torch.tensor([0, query_length], dtype=torch.int32, device=device),
                        max_seqlen_q=query_length,
                        softmax_scale=layer.scaling,
                        causal=True,
                        window_size=(window, 0) if sliding else (-1, -1),
                        sinks=sinks.to(device),
                        ver=3,
                    )
                    .float()
                    .cpu()
                )
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                samples.append(
                    {
                        "layer_id": layer_id,
                        "sliding_window": source.sliding_window if sliding else None,
                        "query_length": query_length,
                        "kv_length": kv_length,
                        "maximum_absolute_error": (actual - expected).abs().max().item(),
                        "key_stride": list(key_cache.stride()),
                    }
                )
        write_json(artifact_dir / "attention-numerical.json", {"model_id": str(model_id), "samples": samples})
    finally:
        torch.cuda.synchronize()
        for backing in backings:
            backing.close()
