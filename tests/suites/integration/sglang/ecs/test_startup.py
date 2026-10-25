from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor.model_runner_components.layer_setup import ModelLayerInfo
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import SpecAuxHiddenStateConfig
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import DeepseekV2Config, GptOssConfig, PretrainedConfig, Qwen3Config

import xtest
from xpool.integrations.sglang.ecs.state import SglangElasticContextState
from xpool.integrations.sglang.hooks.ecs import ElasticContextStateHookSet
from xtest.harness.native.case import run_native_case
from xtest.harness.support.sglang.fakes import server_args
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


@xtest.requirements(device_count=1)
@pytest.mark.parametrize(
    "hf_config",
    (
        Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            num_hidden_layers=3,
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=64,
            max_position_embeddings=2048,
        ),
        DeepseekV2Config(
            architectures=["DeepseekV2ForCausalLM"],
            num_hidden_layers=3,
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=8,
            kv_lora_rank=64,
            qk_rope_head_dim=32,
            qk_nope_head_dim=32,
            v_head_dim=64,
            max_position_embeddings=2048,
        ),
        GptOssConfig(
            architectures=["GptOssForCausalLM"],
            num_hidden_layers=3,
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=64,
            max_position_embeddings=2048,
            layer_types=["full_attention", "full_attention", "sliding_attention"],
            sliding_window=8,
        ),
    ),
    ids=("mha", "mla", "full-swa"),
)
def test_original_factory_consumes_exact_prepared_geometry_and_keeps_reference_limits_stable(
    hf_config: PretrainedConfig,
    tmp_path: Path,
) -> None:
    run_native_case(run_original_factory_case, hf_config, tmp_path / "model", workdir=tmp_path / "case")


def run_original_factory_case(hf_config: PretrainedConfig, tmp_path: Path) -> None:
    for hook in ElasticContextStateHookSet().hooks():
        HookRegistry.register(hook.target, hook.handler, hook.kind)
    HookRegistry.apply_hooks()
    hf_config.save_pretrained(tmp_path)
    model = ModelConfig(str(tmp_path), dtype="bfloat16")
    device = torch.device("cuda", torch.cuda.current_device())
    configurator = KVCacheConfigurator(
        device=str(device),
        gpu_id=device.index,
        ps=ParallelState.trivial(),
        pp_group=None,
        model=torch.nn.Module(),
        model_config=model,
        server_args=server_args(),
        kv_cache_dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        page_size=4,
        sliding_window_size=8 if model.is_hybrid_swa else None,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        is_draft_worker=False,
        post_capture_kv_active=True,
        spec_aux_config=SpecAuxHiddenStateConfig(),
        is_hybrid_swa=model.is_hybrid_swa,
        is_hybrid_swa_compress=model.is_hybrid_swa_compress,
        use_mla_backend=model.attention_arch == AttentionArch.MLA,
        layer_info=ModelLayerInfo(start_layer=0, end_layer=3, num_effective_layers=3),
        forward_stream=torch.cuda.current_stream(),
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        memory_pool_config=None,
    )
    with get_context().override_server_args(
        page_size=4, swa_full_tokens_ratio=0.5, cuda_graph_config=server_args().cuda_graph_config
    ):
        config = MemoryPoolConfig(
            max_total_num_tokens=8192,
            max_running_requests=2,
            full_max_total_num_tokens=8192 if model.is_hybrid_swa else None,
            swa_max_total_num_tokens=4096 if model.is_hybrid_swa else None,
        )
        pools = configurator._init_pools(
            sizes=configurator._derive_pool_sizes(config=config),
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
        )
        state = SglangElasticContextState.create(
            pools.token_to_kv_pool,
            pools.token_to_kv_pool_allocator,
            pools.req_to_token_pool,
            configurator=configurator,
        )
        try:
            assert len(state.components) == (2 if model.is_hybrid_swa else 1)
            assert state.quota.charged_bytes == state.quota.limit_bytes == state.profile.floor_bytes
            before = state.logical_free_pages()
            ceiling = state.profile.floor_bytes + max(profile.bundle_bytes for profile in state.profile.pools)
            reference = state.reference_config(ceiling)
            cost = state.reference_bytes(reference)
            assert cost is not None and cost <= ceiling
            assert reference.max_running_requests == 2
            assert 0 < reference.max_total_num_tokens <= 8192
            state.quota.set_limit(ceiling)
            assert state.reference_config(ceiling) == reference
            assert state.logical_free_pages() == before
            assert state.quota.charged_bytes == state.profile.floor_bytes
        finally:
            torch.cuda.synchronize()
            state.close()
