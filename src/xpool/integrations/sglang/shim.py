"""SGLang-side FFN shim modules."""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode as SglangForwardMode
from torch import nn

from xpool.abi import ForwardMode
from xpool.config import get_global_config


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
        # Integer instance/model identity is injected post-load (see inject_shim_identity);
        # -1 marks an unbound shim so a forward before binding fails closed loudly.
        self.instance_index = -1
        self.model_index = -1

    def bind_identity(self, instance_index: int, model_index: int, *, model_architecture: str) -> None:
        """Stamp diagnostic metadata and integer identity after load.

        Args:
            instance_index: Integer SGLang instance id from xpool config order.
            model_index: Integer model id selecting the FFN executor weight set.
            model_architecture: SGLang/Hugging Face architecture string read
                from the loaded model runner config.

        Side Effects:
            Mutates the shim diagnostic and identity fields used by later
            ``forward`` calls.
        """

        self.instance_index = instance_index
        self.model_index = model_index
        self.model_architecture = model_architecture

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
                false because the current shim ABI returns a full hidden-state tensor.
            gemm_output_zero_allocator: Optional SGLang allocator hook. Must be
                absent because the shim owns native output placement.

        Returns:
            Output tensor returned by the selected Torch dispatcher op. The
            debug loopback op returns a tensor with the same shape, dtype, and
            device as ``hidden_states``; the production op currently fails
            closed because descriptor publication is not implemented yet.

        Raises:
            ShimUnavailableError: If the shim is unbound or receives unsupported
                SGLang runtime modes before dispatcher invocation.
            AttributeError: If the expected ``torch.ops.xpool`` operator is not
                registered in the current process.
            RuntimeError: Propagated from the selected native dispatcher op,
                including the intentionally unimplemented production
                ``ffn_shim`` route and loopback contract violations.

        Side Effects:
            Dispatches directly through ``torch.ops.xpool.ffn_shim`` by default
            or ``torch.ops.xpool.ffn_shim_loopback`` when the process-global
            xpool config has ``debug.enable_shim_loopback`` enabled. The xpool
            SGLang plugin loads and preflights the C extension and global config
            during startup. This Python layer intentionally avoids native-loader
            locks, package-resource lookup, and exception translation so SGLang
            piecewise CUDA graph can trace the shim as a custom Torch op.
        """

        if should_allreduce_fusion:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang all-reduce fusion")
        if use_reduce_scatter:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang reduce-scatter FFN output")
        if gemm_output_zero_allocator is not None:
            raise ShimUnavailableError("xpool FFN shim does not yet support SGLang GEMM zero allocator output")
        if forward_batch is None:
            raise ShimUnavailableError("xpool FFN shim requires a ForwardBatch to determine the forward mode")
        if self.instance_index < 0 or self.model_index < 0:
            raise ShimUnavailableError(
                f"xpool FFN shim for {self.model_architecture} layer {self.layer_id} has no bound "
                "instance/model identity; the xpool plugin must inject it after load"
            )
        # Match by exact SGLang ForwardMode value, not by is_decode()/is_extend():
        # those predicates fold MIXED/SPLIT_PREFILL/DLLM_EXTEND/TARGET_VERIFY/
        # DRAFT_EXTEND into "extend", but the current xpool shim ABI only publishes
        # plain DECODE and EXTEND descriptors.
        sglang_mode = forward_batch.forward_mode
        match sglang_mode:
            case _ if sglang_mode is SglangForwardMode.DECODE:
                forward_mode = ForwardMode.DECODE
            case _ if sglang_mode is SglangForwardMode.EXTEND:
                forward_mode = ForwardMode.EXTEND
            case _:
                raise ShimUnavailableError(
                    f"xpool FFN shim does not support SGLang forward mode {sglang_mode!r} "
                    "(only DECODE and EXTEND are supported)"
                )
        validate_hidden_states(hidden_states, expected_hidden_size=self.hidden_size)
        if get_global_config().debug.enable_shim_loopback:
            native_ffn_shim = torch.ops.xpool.ffn_shim_loopback
        else:
            native_ffn_shim = torch.ops.xpool.ffn_shim
        return native_ffn_shim(
            hidden_states,
            int(self.instance_index),
            int(self.model_index),
            int(self.layer_id),
            int(forward_mode),
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
