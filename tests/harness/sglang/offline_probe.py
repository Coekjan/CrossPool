from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from xpool.config import XpoolConfig
from xpool.integrations.sglang.topology import derive_parallel_policy, load_model_spec

MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
NEW_TOKENS = 8


@dataclass(frozen=True, slots=True)
class SglangGraphSettings:
    cuda_graph: bool
    piecewise_cuda_graph: bool


GRAPH_SETTINGS: tuple[SglangGraphSettings, ...] = (
    SglangGraphSettings(cuda_graph=False, piecewise_cuda_graph=False),
    SglangGraphSettings(cuda_graph=False, piecewise_cuda_graph=True),
    SglangGraphSettings(cuda_graph=True, piecewise_cuda_graph=False),
    SglangGraphSettings(cuda_graph=True, piecewise_cuda_graph=True),
)


class ProbeResult(TypedDict):
    cuda_graph: bool
    piecewise_cuda_graph: bool
    base_gpu_id: int
    output_ids: list[int]
    model_path: str


def main() -> None:
    args = parse_args()
    result = run_probe(
        cuda_graph=not args.disable_cuda_graph,
        piecewise_cuda_graph=not args.disable_piecewise_cuda_graph,
        base_gpu_id=args.base_gpu_id,
        config_path=args.config_path,
    )
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


def run_probe(*, cuda_graph: bool, piecewise_cuda_graph: bool, base_gpu_id: int, config_path: Path) -> ProbeResult:
    config = XpoolConfig.from_file(config_path, env=os.environ)
    model_path = config.model_path_of(MODEL_ID).resolve()
    spec = load_model_spec(model_path, model_id=MODEL_ID)
    policy = derive_parallel_policy(
        spec,
        atn_device_count=len(config.devices.atn_cuda_devices),
        ffn_tp_size=len(config.devices.ffn_cuda_devices),
    )

    from sglang import Engine

    engine = None
    try:
        engine = Engine(
            model_path=str(model_path),
            trust_remote_code=True,
            tp_size=policy.sglang_tp_size,
            dp_size=policy.sglang_dp_size,
            enable_dp_attention=policy.enable_dp_atn,
            random_seed=0,
            disable_cuda_graph=not cuda_graph,
            disable_piecewise_cuda_graph=not piecewise_cuda_graph,
            base_gpu_id=base_gpu_id,
            log_level="error",
            log_level_http="error",
        )
        output = engine.generate(
            prompt=PROMPT,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": NEW_TOKENS,
                "min_new_tokens": NEW_TOKENS,
                "ignore_eos": True,
            },
        )
        if not isinstance(output, dict):
            raise TypeError(f"expected SGLang Engine.generate to return dict, got {type(output).__name__}")
        output_ids = output.get("output_ids")
        if not isinstance(output_ids, list) or not all(isinstance(token_id, int) for token_id in output_ids):
            raise TypeError(f"expected SGLang output_ids to be list[int], got {output_ids!r}")
        if len(output_ids) != NEW_TOKENS:
            raise ValueError(f"expected {NEW_TOKENS} output tokens, got {len(output_ids)}")
        return {
            "cuda_graph": cuda_graph,
            "piecewise_cuda_graph": piecewise_cuda_graph,
            "base_gpu_id": base_gpu_id,
            "output_ids": output_ids,
            "model_path": str(model_path),
        }
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
