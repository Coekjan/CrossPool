"""SGLang Devkit observer for request-prefill next-token logits."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Concatenate
from urllib.parse import quote

import torch
from safetensors.torch import save_file
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput

from xpool.config import get_global_config
from xpool.native import RuntimeRole

__all__ = ["install"]

runtime_roles = frozenset({RuntimeRole.INSTANCE})
logger = logging.getLogger(__name__)
installed = False


def install() -> None:
    """Install request-prefill logit capture on SGLang's model runner."""

    config = get_global_config()
    outdir = config.debug.prefill_logit_observer.outdir
    if outdir is None:
        raise RuntimeError("xpool prefill-logit observer requires debug.prefill_logit_observer.outdir")
    outdir.mkdir(parents=True, exist_ok=True)

    global installed
    if installed:
        return
    setattr(ModelRunner, "forward", wrap_forward(ModelRunner.forward))
    installed = True


def wrap_forward[**P](
    original: Callable[Concatenate[ModelRunner, ForwardBatch, P], ModelRunnerOutput],
) -> Callable[Concatenate[ModelRunner, ForwardBatch, P], ModelRunnerOutput]:
    """Return a model-forward wrapper that records qualifying Prefill outputs."""

    @functools.wraps(original)
    def wrapped(
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> ModelRunnerOutput:
        result = original(model_runner, forward_batch, *args, **kwargs)
        capture_prefill_logits(model_runner, forward_batch, result)
        return result

    return wrapped


def capture_prefill_logits(
    model_runner: ModelRunner,
    forward_batch: ForwardBatch,
    result: ModelRunnerOutput,
) -> None:
    """Retain each request's latest rank-zero, single-request EXTEND logits.

    Replacing the same request snapshot across chunks leaves its final Prefill
    distribution available when the HTTP response completes.
    """

    if model_runner.ps.tp_rank != 0 or forward_batch.forward_mode is not ForwardMode.EXTEND:
        return
    rids = forward_batch.rids
    if not rids:
        return
    try:
        if len(rids) != 1:
            raise ValueError(f"prefill-logit observer expected one request id, received {len(rids)}")
        request_id = rids[0]
        if not request_id.startswith(("xpool-serving-graph-", "xpool-elastic-kv-")):
            return
        logits_output = result.logits_output
        if not isinstance(logits_output, LogitsProcessorOutput) or logits_output.next_token_logits is None:
            raise ValueError("prefill-logit observer received no next-token logits")
        logits = logits_output.next_token_logits
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"prefill-logit observer expected [1, V] logits, received {tuple(logits.shape)}")
        if logits.dtype is not torch.float32:
            raise ValueError(f"prefill-logit observer expected FP32 logits, received {logits.dtype}")
        outdir = get_global_config().debug.prefill_logit_observer.outdir
        if outdir is None:
            raise RuntimeError("xpool prefill-logit observer output directory became unavailable")
        save_file(
            {"next_token_logits": logits.detach().to(device="cpu").contiguous()},
            outdir / f"xpool.prefill-logits.{os.getpid()}.{quote(request_id, safe='')}.safetensors",
            metadata={"rid": request_id},
        )
    except Exception:
        logger.warning("failed to record xpool prefill logits", exc_info=True)
