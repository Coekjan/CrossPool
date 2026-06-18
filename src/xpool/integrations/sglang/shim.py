"""SGLang-side FFN shim modules."""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode
from torch import nn

from xpool.abi import ForwardMode


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
        model_architecture: str,
        layer_id: int,
        hidden_size: int,
        layer_kind: FfnLayerKind,
    ) -> None:
        """Initialize a parameter-free FFN shim.

        Args:
            model_architecture: Stable SGLang architecture name used for diagnostics.
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
        self.model_architecture = model_architecture
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.layer_kind = layer_kind
        # Integer instance/model identity is injected post-load (see inject_shim_identity);
        # -1 marks an unbound shim so a forward before binding fails closed loudly.
        self.instance_index = -1
        self.model_index = -1

    def bind_identity(self, instance_index: int, model_index: int) -> None:
        """Stamp the integer identity used by the native FFN op after load.

        Args:
            instance_index: Integer SGLang instance id from xpool config order.
            model_index: Integer model id selecting the FFN executor weight set.

        Side Effects:
            Mutates the shim identity fields used by later ``forward`` calls.
        """

        self.instance_index = instance_index
        self.model_index = model_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: object | None = None,
    ) -> torch.Tensor:
        """Forward hidden states through the native xpool FFN shim op.

        Args:
            hidden_states: Contiguous CUDA tensor with shape
                ``[num_tokens, hidden_size]``.
            forward_batch: SGLang forward-batch metadata used to derive the
                exact xpool forward mode.
            should_allreduce_fusion: SGLang all-reduce fusion flag. Must be
                false because the current shim ABI has no fused all-reduce path.
            use_reduce_scatter: SGLang reduce-scatter output flag. Must be
                false because the current shim ABI returns a full hidden-state tensor.
            gemm_output_zero_allocator: Optional SGLang allocator hook. Must be
                absent because the shim owns native output placement.

        Returns:
            Output CUDA tensor with the same shape and dtype as ``hidden_states``.

        Raises:
            ShimUnavailableError: If the shim is unbound, receives unsupported
                SGLang runtime modes, or cannot call the native op.

        Side Effects:
            Dispatches to a separately registered ``torch.ops.xpool.ffn_shim``
            implementation when one is loaded. This Python skeleton does not
            implement descriptor publication itself; the eventual native op must
            own that work and remain CUDA graph safe.
        """

        if should_allreduce_fusion:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang all-reduce fusion")
        if use_reduce_scatter:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang reduce-scatter FFN output")
        if gemm_output_zero_allocator is not None:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang GEMM zero allocator output")
        validate_hidden_states(hidden_states, expected_hidden_size=self.hidden_size)
        if self.instance_index < 0 or self.model_index < 0:
            raise ShimUnavailableError(
                f"xpool FFN shim for {self.model_architecture} layer {self.layer_id} has no bound "
                "instance/model identity; the xpool plugin must inject it after load"
            )
        if forward_batch is None:
            raise ShimUnavailableError("xpool FFN shim requires a ForwardBatch to determine the forward mode")
        # Match by exact SGLang ForwardMode value, not by is_decode()/is_extend(): those
        # predicates fold MIXED/SPLIT_PREFILL/DLLM_EXTEND/TARGET_VERIFY/DRAFT_EXTEND into
        # "extend", but the xpool shim ABI only supports plain DECODE and EXTEND.
        sglang_mode = forward_batch.forward_mode
        if sglang_mode is SglangForwardMode.DECODE:
            forward_mode = ForwardMode.DECODE
        elif sglang_mode is SglangForwardMode.EXTEND:
            forward_mode = ForwardMode.EXTEND
        else:
            raise ShimUnavailableError(
                f"xpool FFN shim does not support SGLang forward mode {sglang_mode!r} "
                "(only DECODE and EXTEND are supported)"
            )
        return call_native_ffn_shim(self, hidden_states, forward_mode)

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


def validate_hidden_states(hidden_states: torch.Tensor, *, expected_hidden_size: int) -> None:
    """Validate the tensor contract required by the native FFN shim.

    Args:
        hidden_states: Candidate hidden-state tensor passed by SGLang.
        expected_hidden_size: Hidden dimension declared by the model adapter.

    Raises:
        ShimUnavailableError: If the tensor is not 2D, CUDA-backed, contiguous,
            supported dtype, or the hidden dimension does not match.
    """

    if hidden_states.dim() != 2:
        raise ShimUnavailableError(f"xpool FFN shim expects a 2D hidden-state tensor, got {hidden_states.dim()}D")
    if hidden_states.shape[1] != expected_hidden_size:
        raise ShimUnavailableError(
            f"xpool FFN shim expected hidden size {expected_hidden_size}, got {hidden_states.shape[1]}"
        )
    if not hidden_states.is_cuda:
        raise ShimUnavailableError("xpool FFN shim expects a CUDA tensor")
    if not hidden_states.is_contiguous():
        raise ShimUnavailableError("xpool FFN shim expects a contiguous hidden-state tensor")
    if hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ShimUnavailableError(f"xpool FFN shim does not support dtype {hidden_states.dtype}")


def call_native_ffn_shim(
    shim: FfnShimModule,
    hidden_states: torch.Tensor,
    forward_mode: ForwardMode,
) -> torch.Tensor:
    """Call the registered native FFN shim op.

    Args:
        shim: Bound FFN shim module carrying instance, model, and layer identity.
        hidden_states: Validated CUDA hidden-state tensor.
        forward_mode: xpool ABI forward mode derived from SGLang metadata.

    Returns:
        Native op output tensor.

    Raises:
        ShimUnavailableError: If ``torch.ops.xpool.ffn_shim`` is not registered.

    Side Effects:
        Enters the registered native extension op. In the completed runtime that
        op is responsible for publishing descriptors and waiting in eager or CUDA
        graph replay contexts; this Python layer only validates and forwards the call.
    """

    try:
        native_op = torch.ops.xpool.ffn_shim
    except (AttributeError, RuntimeError) as exc:
        raise ShimUnavailableError(
            f"xpool native FFN shim op is not loaded for {shim.model_architecture} layer {shim.layer_id}"
        ) from exc

    return native_op(
        hidden_states,
        int(shim.instance_index),
        int(shim.model_index),
        int(shim.layer_id),
        int(forward_mode),
    )
