"""SGLang-side FFN shim modules."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from sglang.srt.layers import dp_attention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch import nn

import xpool.ops
from xpool.abi import DpPaddingMode, FfnResultHandoff, XPoolForwardMode
from xpool.fabric import FfnLayerKind
from xpool.transport import FfnRequestMetadata


class ShimUnavailableError(RuntimeError):
    """Raised when the xpool shim is enabled but native runtime is not ready."""


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
            Initializes ``nn.Module`` directly and leaves runtime layer metadata
            unbound until plugin post-load binding.
        """

        if not isinstance(layer_id, int) or isinstance(layer_id, bool) or layer_id < 0:
            raise ValueError("xpool FFN shim layer_id must be a non-negative integer")
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError("xpool FFN shim hidden_size must be a positive integer")
        nn.Module.__init__(self)
        self.model_architecture = "unknown"
        self.layer_id = layer_id
        # -1 marks an unbound shim so forward fails before post-load binding.
        self.layer_ordinal = -1
        self.hidden_size = hidden_size
        self.layer_kind = layer_kind
        self.atn_dp_size = 1

    def bind_runtime(
        self,
        *,
        layer_ordinal: int,
        model_architecture: str,
        atn_dp_size: int = 1,
    ) -> None:
        """Bind request-time SGLang metadata after model load.

        Args:
            layer_ordinal: Zero-based position in the Fabric workload contract.
            model_architecture: SGLang/Hugging Face architecture string read
                from the loaded model runner config.
            atn_dp_size: Attention data-parallel world size for this SGLang rank.

        Side Effects:
            Mutates the runtime fields used by later ``forward`` calls.
        """

        self.layer_ordinal = layer_ordinal
        self.model_architecture = model_architecture
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
            hidden_states: Contiguous CUDA tensor with shape
                ``[num_tokens, hidden_size]`` and a supported floating dtype.
                FakeTensor or Meta dispatch is handled only by the registered
                dispatcher fake implementation during graph tracing.
            forward_batch: SGLang forward-batch metadata used to derive the
                exact xpool forward mode.
            should_allreduce_fusion: SGLang all-reduce fusion flag. Must be
                false because the current shim ABI has no fused all-reduce path.
            use_reduce_scatter: SGLang result-handoff flag. When true, the
                native request returns input for SGLang's existing
                reduce-scatter path; otherwise it returns replicated full
                hidden states.
            gemm_output_zero_allocator: Optional SGLang allocator hook. Must be
                absent because the shim owns native output placement.

        Returns:
            Output tensor returned by the sole ``xpool.ops.ffn_shim``
            dispatcher API. Current concrete execution succeeds only through
            the configured Instance, AtnAgent, or FfnAgent debug loopback; the
            native FFN execution boundary returns ``NotImplemented`` until
            Phase 8 provides a Dense or MoE body.

        Raises:
            ShimUnavailableError: If the shim is unbound or receives unsupported
                SGLang runtime modes before dispatcher invocation.
            RuntimeError: Propagated from the selected native dispatcher op,
                including synchronous Tensor, attachment, and launch
                precondition failures.

        Side Effects:
            Dispatches through ``xpool.ops.ffn_shim``. The SGLang plugin binds
            runtime layer metadata after model load and the Instance runtime
            attaches the daemon-brokered Transport arena before serving. The
            fake implementation preserves symbolic token dimensions for
            piecewise CUDA graph warmup. Device execution failures are sticky:
            the native path poisons output with NaNs and the Instance failure
            monitor terminates the serving process after observing the failure;
            they are not synchronously raised by this call.
        """

        if should_allreduce_fusion:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang all-reduce fusion")
        if gemm_output_zero_allocator is not None:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang GEMM zero allocator output")
        if forward_batch is None:
            raise ShimUnavailableError("xpool FFN shim requires a ForwardBatch to determine the forward mode")
        if self.layer_ordinal < 0:
            raise ShimUnavailableError(
                f"xpool FFN shim for {self.model_architecture} layer {self.layer_id} has no bound "
                "runtime metadata; the xpool plugin must bind the shim after load"
            )
        # Match by exact SGLang ForwardMode value, not by is_decode()/is_extend():
        # those predicates fold MIXED/SPLIT_PREFILL/DLLM_EXTEND/TARGET_VERIFY/
        # DRAFT_EXTEND into "extend", but the current xpool shim ABI only publishes
        # plain DECODE, EXTEND, and IDLE requests.
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
        self.validate_hidden_states(hidden_states)
        if self.atn_dp_size == 1:
            request_dp_padding_mode = DpPaddingMode.NONE
            request_global_num_tokens_gpu = None
        else:
            sglang_padding_mode = getattr(forward_batch, "dp_padding_mode", None)
            match sglang_padding_mode:
                case None:
                    raise ShimUnavailableError("xpool attention DP requires an explicit SGLang padding mode")
                case dp_attention.DpPaddingMode.MAX_LEN:
                    request_dp_padding_mode = DpPaddingMode.MAX_LEN
                case dp_attention.DpPaddingMode.SUM_LEN:
                    request_dp_padding_mode = DpPaddingMode.SUM_LEN
                case _:
                    raise ShimUnavailableError(
                        f"xpool FFN shim does not support SGLang DP padding mode {sglang_padding_mode!r}"
                    )
            token_counts = getattr(forward_batch, "global_num_tokens_gpu", None)
            if not isinstance(token_counts, torch.Tensor):
                raise ShimUnavailableError("xpool FFN shim expected global_num_tokens_gpu to be a tensor")
            request_global_num_tokens_gpu = token_counts
        request_metadata = FfnRequestMetadata(
            layer_ordinal=self.layer_ordinal,
            forward_mode=forward_mode,
            result_handoff=(
                FfnResultHandoff.REDUCE_SCATTER_INPUT if use_reduce_scatter else FfnResultHandoff.REPLICATED_FULL
            ),
            dp_padding_mode=request_dp_padding_mode,
        )
        return xpool.ops.ffn_shim(
            hidden_states,
            request_global_num_tokens_gpu,
            request_metadata,
        )

    def validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Validate hidden-state rank and width against this shim.

        Args:
            hidden_states: Candidate hidden-state tensor passed by SGLang.

        Raises:
            ShimUnavailableError: If rank or hidden width is invalid.
        """

        if hidden_states.dim() != 2:
            raise ShimUnavailableError(f"xpool FFN shim expects a 2D hidden-state tensor, got {hidden_states.dim()}D")
        if hidden_states.shape[1] != self.hidden_size:
            raise ShimUnavailableError(
                f"xpool FFN shim expected hidden size {self.hidden_size}, got {hidden_states.shape[1]}"
            )

    def extra_repr(self) -> str:
        """Return a concise module representation for SGLang model dumps.

        Returns:
            String containing architecture, layer id, hidden size, and layer kind.
        """

        return (
            f"model_architecture={self.model_architecture!r}, layer_id={self.layer_id}, "
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
