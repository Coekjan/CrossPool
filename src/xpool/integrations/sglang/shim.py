"""SGLang-side FFN shim modules."""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum

import torch
from sglang.srt.layers.dp_attention import DpPaddingMode as SglangDpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch import nn

import xpool.ops
from xpool.abi import DpPaddingMode, FfnCollectivePolicy, FfnRequestMetadata, XPoolForwardMode


class ShimUnavailableError(RuntimeError):
    """Raised when the xpool shim is enabled but native runtime is not ready."""


class FfnLayerKind(StrEnum):
    """FFN layer category represented by a shim module.

    Attributes:
        DENSE: Dense MLP/FFN layer.
        SPARSE: Sparse routed-expert MoE layer.
    """

    DENSE = "dense"
    SPARSE = "sparse"


class FfnShimModule(nn.Module):
    """Parameter-free decoder FFN replacement.

    Model-specific adapters may inherit from both this class and an original
    SGLang FFN class. This constructor intentionally initializes ``nn.Module``
    directly so the original FFN constructor does not allocate attention-side
    FFN weights.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        hidden_size: int,
        layer_kind: FfnLayerKind,
    ) -> None:
        """Initialize a parameter-free FFN shim.

        Args:
            layer_id: Decoder layer id routed to the native FFN executor.
            hidden_size: Expected hidden-state width for every shim call.
            layer_kind: Dense or sparse FFN category.

        Preconditions:
            The caller is a model-specific replacement class that intentionally
            avoids the original SGLang FFN constructor.

        Side Effects:
            Initializes ``nn.Module`` directly and leaves integer instance/model
            identity unbound until plugin post-load binding.
        """

        nn.Module.__init__(self)
        self.model_architecture = "unknown"
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.layer_kind = layer_kind
        # Integer instance identity is injected post-load (see inject_shim_identity);
        # -1 marks an unbound shim so a forward before binding fails closed loudly.
        self.instance_index = -1
        self.sglang_rank = -1
        self.atn_tp_rank = 0
        self.atn_tp_size = 1
        self.atn_dp_rank = 0
        self.atn_dp_size = 1

    def bind_identity(
        self,
        instance_index: int,
        sglang_rank: int,
        *,
        model_architecture: str,
        atn_tp_rank: int = 0,
        atn_tp_size: int = 1,
        atn_dp_rank: int = 0,
        atn_dp_size: int = 1,
    ) -> None:
        """Stamp diagnostic metadata and integer identity after load.

        Args:
            instance_index: Integer SGLang instance id from xpool config order.
            sglang_rank: SGLang tensor-parallel rank that owns this shim.
            model_architecture: SGLang/Hugging Face architecture string read
                from the loaded model runner config.
            atn_tp_rank: Attention tensor-parallel rank for this SGLang rank.
            atn_tp_size: Attention tensor-parallel world size for this SGLang rank.
            atn_dp_rank: Attention data-parallel rank for this SGLang rank.
            atn_dp_size: Attention data-parallel world size for this SGLang rank.

        Side Effects:
            Mutates the shim diagnostic and identity fields used by later
            ``forward`` calls.
        """

        self.instance_index = instance_index
        self.sglang_rank = sglang_rank
        self.model_architecture = model_architecture
        self.atn_tp_rank = atn_tp_rank
        self.atn_tp_size = atn_tp_size
        self.atn_dp_rank = atn_dp_rank
        self.atn_dp_size = atn_dp_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: object | None = None,
    ) -> torch.Tensor:
        """Forward hidden states through the configured xpool FFN shim op.

        Args:
            hidden_states: Tensor with shape ``[num_tokens, hidden_size]``. The
                selected Torch dispatcher op validates backend, contiguity, and
                dtype. Debug loopback supports CUDA tensors for execution and
                Meta tensors for dispatcher tracing.
            forward_batch: SGLang forward-batch metadata used to derive the
                exact xpool forward mode.
            should_allreduce_fusion: SGLang all-reduce fusion flag. Must be
                false because the current shim ABI has no fused all-reduce path.
            use_reduce_scatter: SGLang reduce-scatter output flag. Must be
                false until the transport path can return reduce-scattered FFN
                output.
            gemm_output_zero_allocator: Optional SGLang allocator hook. Must be
                absent because the shim owns native output placement.

        Returns:
            Output tensor returned by the selected Torch dispatcher op. The
            debug loopback op returns a tensor with the same shape, dtype, and
            device as ``hidden_states``; the production op fails closed until
            daemon-brokered runtime metadata is registered in the native shim
            registry.

        Raises:
            ShimUnavailableError: If the shim is unbound or receives unsupported
                SGLang runtime modes before dispatcher invocation.
            AttributeError: If the expected ``xpool.ops`` wrapper is not
                registered in the current process.
            RuntimeError: Propagated from the selected native dispatcher op,
                including the intentionally unimplemented production
                ``ffn_shim`` route and loopback contract violations.

        Side Effects:
            Dispatches through ``xpool.ops.instance.ffn_shim``. The xpool
            SGLang plugin loads and preflights the C extension and global
            config during startup. The wrapper fake implementation preserves
            symbolic token dimensions for SGLang piecewise CUDA graph warmup.
        """

        if should_allreduce_fusion:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang all-reduce fusion")
        if use_reduce_scatter:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang reduce-scatter output")
        if gemm_output_zero_allocator is not None:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang GEMM zero allocator output")
        if forward_batch is None:
            raise ShimUnavailableError("xpool FFN shim requires a ForwardBatch to determine the forward mode")
        if self.instance_index < 0 or self.sglang_rank < 0:
            raise ShimUnavailableError(
                f"xpool FFN shim for {self.model_architecture} layer {self.layer_id} has no bound "
                "xpool identity; the xpool plugin must inject instance and rank identity after load"
            )
        # Match by exact SGLang ForwardMode value, not by is_decode()/is_extend():
        # those predicates fold MIXED/SPLIT_PREFILL/DLLM_EXTEND/TARGET_VERIFY/
        # DRAFT_EXTEND into "extend", but the current xpool shim ABI only publishes
        # plain DECODE, EXTEND, and IDLE descriptors.
        match forward_batch.forward_mode:
            case mode if mode is ForwardMode.DECODE:
                forward_mode = XPoolForwardMode.DECODE
            case mode if mode is ForwardMode.EXTEND:
                forward_mode = XPoolForwardMode.EXTEND
            case mode if mode is ForwardMode.IDLE:
                forward_mode = XPoolForwardMode.IDLE
            case unsupported_mode:
                raise ShimUnavailableError(
                    f"xpool FFN shim does not support SGLang forward mode {unsupported_mode!r} "
                    "(only DECODE, EXTEND, and IDLE are supported)"
                )
        validate_hidden_states(hidden_states, expected_hidden_size=self.hidden_size)
        if self.atn_dp_size == 1:
            request_dp_padding_mode = DpPaddingMode.NONE
            request_global_dp_buffer_len = int(hidden_states.shape[0])
            request_global_num_tokens_gpu = None
        else:
            request_dp_padding_mode = dp_padding_mode(forward_batch)
            request_global_dp_buffer_len = global_dp_buffer_len(forward_batch, hidden_states)
            request_global_num_tokens_gpu = global_num_tokens_gpu(forward_batch)
        request_metadata = FfnRequestMetadata(
            instance_index=self.instance_index,
            layer_id=self.layer_id,
            forward_mode=int(forward_mode),
            collective_policy=int(FfnCollectivePolicy.FULL_REDUCED),
            dp_padding_mode=int(request_dp_padding_mode),
            global_dp_buffer_len=request_global_dp_buffer_len,
            atn_tp_rank=self.atn_tp_rank,
            atn_tp_size=self.atn_tp_size,
            atn_dp_rank=self.atn_dp_rank,
            atn_dp_size=self.atn_dp_size,
        )
        return xpool.ops.instance.ffn_shim(
            hidden_states,
            request_global_num_tokens_gpu,
            request_metadata,
            self.sglang_rank,
        )

    def extra_repr(self) -> str:
        """Return a concise module representation for SGLang model dumps.

        Returns:
            String containing architecture, layer id, hidden size, and layer kind.
        """

        return (
            f"model_architecture={self.model_architecture!r}, instance_index={self.instance_index}, "
            f"sglang_rank={self.sglang_rank}, layer_id={self.layer_id}, "
            f"hidden_size={self.hidden_size}, layer_kind={self.layer_kind.value}"
        )


def iter_ffn_shims(module: nn.Module) -> Iterator[FfnShimModule]:
    """Iterate xpool FFN shims contained in a module tree.

    Args:
        module: Root PyTorch module to scan.

    Yields:
        Every descendant module that is an ``FfnShimModule``.
    """

    for child in module.modules():
        if isinstance(child, FfnShimModule):
            yield child


def validate_hidden_states(hidden_states: torch.Tensor, *, expected_hidden_size: int) -> None:
    """Validate the tensor contract known only to the Python model adapter.

    Args:
        hidden_states: Candidate hidden-state tensor passed by SGLang.
        expected_hidden_size: Hidden dimension declared by the model adapter.

    Raises:
        ShimUnavailableError: If the tensor is not 2D or the hidden dimension
            does not match the model adapter.
    """

    if hidden_states.dim() != 2:
        raise ShimUnavailableError(f"xpool FFN shim expects a 2D hidden-state tensor, got {hidden_states.dim()}D")
    if hidden_states.shape[1] != expected_hidden_size:
        raise ShimUnavailableError(
            f"xpool FFN shim expected hidden size {expected_hidden_size}, got {hidden_states.shape[1]}"
        )


def dp_padding_mode(forward_batch: ForwardBatch) -> DpPaddingMode:
    """Return the xpool DP padding mode represented by a SGLang forward batch."""

    value = getattr(forward_batch, "dp_padding_mode", None)
    if value is None:
        return DpPaddingMode.NONE
    match value:
        case SglangDpPaddingMode.MAX_LEN:
            return DpPaddingMode.MAX_LEN
        case SglangDpPaddingMode.SUM_LEN:
            return DpPaddingMode.SUM_LEN
        case _:
            raise ShimUnavailableError(f"xpool FFN shim does not support SGLang DP padding mode {value!r}")


def global_dp_buffer_len(forward_batch: ForwardBatch, hidden_states: torch.Tensor) -> int:
    """Return the global DP buffer length for the current FFN request."""

    value = getattr(forward_batch, "global_dp_buffer_len", None)
    if value is None:
        return int(hidden_states.shape[0])
    if not isinstance(value, int) or value < 0:
        raise ShimUnavailableError(f"xpool FFN shim expected non-negative global_dp_buffer_len, got {value!r}")
    return value


def global_num_tokens_gpu(forward_batch: ForwardBatch) -> torch.Tensor | None:
    """Return optional per-DP-rank token counts for the current request."""

    value = getattr(forward_batch, "global_num_tokens_gpu", None)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ShimUnavailableError("xpool FFN shim expected global_num_tokens_gpu to be a tensor")
    return value
