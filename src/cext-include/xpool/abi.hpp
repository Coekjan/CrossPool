#pragma once

/// \file xpool/abi.hpp
/// \brief Native descriptor and transport-observer ABI shared by xpool.
///
/// The descriptor fields are an inter-process ABI, so every field below
/// documents its producer, consumer, unit, and stability expectation.
/// Cross-process tensor references are arena offsets, not raw CUDA virtual
/// addresses, because CUDA IPC mappings may use different addresses in the
/// producer process and in the local devagent.

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include <xpool/utils/arith.hpp>

/// Native xpool ABI symbols shared by shim frontends and devagent runtime.
namespace xpool::abi {

/// Version stamped into every descriptor so incompatible producers fail closed.
inline constexpr std::uint32_t kAbiVersion = 22;

/// Process role selected during native runtime initialization.
struct RuntimeRole {
  /// On-wire runtime role integer values.
  enum Type : std::uint32_t {
    /// Instance process that attaches transport arenas and runs FFN shim calls.
    kInstance = 1,
    /// Devagent process that owns transport arenas and persistent kernels.
    kDevagent = 2,
  };

  /// Return whether an integer is a valid RuntimeRole value.
  /// \param role Integer value read from host-side op arguments.
  /// \return True for instance and devagent roles.
  static constexpr bool is_valid(std::int64_t role) {
    return role == static_cast<std::int64_t>(kInstance) ||
           role == static_cast<std::int64_t>(kDevagent);
  }

  /// Parse a host-side integer into a RuntimeRole value.
  /// \param role Integer value read from native init arguments.
  /// \return RuntimeRole value used to lock the native process role.
  /// \throws c10::Error if role is not instance or devagent.
  static Type parse(std::int64_t role) {
    TORCH_CHECK(is_valid(role), "xpool received an invalid runtime role");
    return static_cast<Type>(role);
  }

  /// Require one initialized native process role before running an op.
  /// \param actual Process role recorded by native init, or empty before init.
  /// \param expected Role required by the native op.
  /// \param op_name Torch operator name used in error messages.
  /// \throws c10::Error if init has not run or the process role differs.
  static void expect(const std::optional<Type> &actual, Type expected,
                     const char *op_name) {
    TORCH_CHECK(actual.has_value(), "xpool op ", op_name,
                " requires xpool.init(cuda_device, role, debug_options_mask) "
                "first");
    TORCH_CHECK(*actual == expected, "xpool op ", op_name,
                " requires runtime role ", name(expected),
                " but current process was initialized as ", name(*actual));
  }

private:
  static const char *name(Type role) {
    switch (role) {
    case kInstance:
      return "instance";
    case kDevagent:
      return "devagent";
    default:
      TORCH_CHECK(false, "xpool received an invalid runtime role");
    }
  }
};

/// Process-wide native debug options installed during runtime initialization.
struct DebugOption {
  /// On-wire debug option bit values.
  enum Bit : std::uint64_t {
    /// Route instance FFN shim calls through the direct debug loopback
    /// executor.
    kShimLoopback = 1ULL << 0,
    /// Route daemon-brokered transport requests through the debug persistent
    /// transport loopback executor.
    kTransportLoopback = 1ULL << 1,
    /// Record native transport device-phase timing records in arena storage.
    kTransportObserver = 1ULL << 2,
  };

  /// Known option bits accepted by this ABI version.
  static constexpr std::uint64_t kKnownMask =
      static_cast<std::uint64_t>(kShimLoopback) |
      static_cast<std::uint64_t>(kTransportLoopback) |
      static_cast<std::uint64_t>(kTransportObserver);
};

/// Process-wide native debug option set installed during runtime
/// initialization.
struct DebugOptions {
  /// Bitmask of DebugOption::Bit values.
  std::uint64_t mask;

  /// Parse and validate a host-side integer debug option bitmask.
  /// \param raw_mask Integer bitmask read from native init arguments.
  /// \return DebugOptions wrapper for host and device state.
  /// \throws c10::Error if raw_mask is negative, unknown bits are set, or
  /// mutually exclusive options are enabled together.
  static DebugOptions parse(std::int64_t raw_mask) {
    TORCH_CHECK(raw_mask >= 0,
                "xpool debug options require a non-negative option mask");
    DebugOptions options{static_cast<std::uint64_t>(raw_mask)};
    TORCH_CHECK((options.mask & ~DebugOption::kKnownMask) == 0U,
                "xpool received unknown debug option bits");
    TORCH_CHECK(!(options.enabled(DebugOption::kShimLoopback) &&
                  options.enabled(DebugOption::kTransportLoopback)),
                "xpool shim and transport loopback debug options are mutually "
                "exclusive");
    return options;
  }

  /// Return whether a debug option bit is enabled in a mask.
  /// \param option Option bit to test.
  /// \return True when option is set in mask.
#if defined(__CUDACC__)
  __host__ __device__
#endif
      constexpr bool
      enabled(DebugOption::Bit option) const {
    return (mask & static_cast<std::uint64_t>(option)) != 0U;
  }
};

static_assert(sizeof(DebugOptions) == sizeof(std::uint64_t));
static_assert(std::is_standard_layout_v<DebugOptions>);
static_assert(std::is_trivially_copyable_v<DebugOptions>);

/// Runtime forward mode values accepted by the first xpool FFN shim ABI.
struct XPoolForwardMode {
  /// On-wire forward-mode integer values.
  enum Type : std::uint32_t {
    /// Extend/prefill-mode FFN request for a contiguous prompt-token batch.
    kExtend = 1,
    /// Decode-mode FFN request for one scheduler decode step.
    kDecode = 2,
    /// Data-parallel idle-rank request with no live FFN work.
    kIdle = 4,
  };

  /// Return whether an integer is a valid on-wire XPoolForwardMode value.
  /// \param forward_mode Integer value read from host-side op arguments or
  /// descriptors.
  /// \return True for decode, extend, and idle forward modes.
  static constexpr bool is_valid(std::int64_t forward_mode) {
    return forward_mode == static_cast<std::int64_t>(kDecode) ||
           forward_mode == static_cast<std::int64_t>(kExtend) ||
           forward_mode == static_cast<std::int64_t>(kIdle);
  }
};

/// FFN output collective contract represented in request descriptors.
struct FfnCollectivePolicy {
  /// On-wire FFN collective-policy integer values.
  enum Type : std::uint32_t {
    /// FFN output is fully reduced and may be consumed directly.
    kFullReduced = 1,
    /// FFN output is an additive partial for this attention TP rank.
    kAtnTpPartial = 2,
  };

  /// Return whether an integer is a valid FfnCollectivePolicy value.
  /// \param collective_policy Integer value read from host-side op arguments or
  /// descriptors.
  /// \return True for fully reduced and attention-TP partial outputs.
  static constexpr bool is_valid(std::int64_t collective_policy) {
    return collective_policy == static_cast<std::int64_t>(kFullReduced) ||
           collective_policy == static_cast<std::int64_t>(kAtnTpPartial);
  }
};

/// Data-parallel padding mode represented in FFN request descriptors.
struct DpPaddingMode {
  /// On-wire data-parallel padding-mode integer values.
  enum Type : std::uint32_t {
    /// Request is not running through data-parallel synchronization.
    kNone = 0,
    /// Producer padded each DP rank to the maximum rank-local token count.
    kMaxLen = 1,
    /// Producer packed all DP rank token counts into one summed buffer.
    kSumLen = 2,
  };

  /// Return whether an integer is a valid DpPaddingMode value.
  /// \param dp_padding_mode Integer value read from host-side op arguments or
  /// descriptors.
  /// \return True for none, max-len, and sum-len padding modes.
  static constexpr bool is_valid(std::int64_t dp_padding_mode) {
    return dp_padding_mode == static_cast<std::int64_t>(kNone) ||
           dp_padding_mode == static_cast<std::int64_t>(kMaxLen) ||
           dp_padding_mode == static_cast<std::int64_t>(kSumLen);
  }
};

/// Element dtype of the hidden-state tensor referenced by an FFN request.
struct TensorDType {
  /// On-wire hidden-state dtype integer values.
  enum Type : std::uint32_t {
    /// bfloat16 hidden states.
    kBf16 = 1,
    /// IEEE float16 hidden states.
    kFp16 = 2,
    /// IEEE float32 hidden states.
    kFp32 = 3,
  };

  /// Wrapped on-wire TensorDType value.
  Type value;

  /// Construct a dtype wrapper from an on-wire TensorDType value.
  /// \param value On-wire TensorDType value.
  constexpr explicit TensorDType(Type value) : value(value) {}

  /// Parse a Torch scalar type into the stable xpool descriptor dtype.
  /// \param scalar_type Runtime hidden-state scalar type from a Torch tensor.
  /// \return Stable TensorDType wrapper for descriptor metadata.
  /// \throws c10::Error if scalar_type is not float32, float16, or bfloat16.
  static TensorDType parse(c10::ScalarType scalar_type) {
    switch (scalar_type) {
    case c10::ScalarType::BFloat16:
      return TensorDType{kBf16};
    case c10::ScalarType::Half:
      return TensorDType{kFp16};
    case c10::ScalarType::Float:
      return TensorDType{kFp32};
    default:
      TORCH_CHECK(false, "xpool FFN shim supports only float32, float16, and "
                         "bfloat16 tensors");
    }
  }

  /// Return whether two TensorDType wrappers hold the same on-wire value.
  /// \return True when both wrappers represent the same dtype.
  constexpr bool operator==(const TensorDType &) const = default;

  /// Return whether this wrapper holds a specific TensorDType value.
  /// \param dtype On-wire TensorDType value to compare with.
  /// \return True when this wrapper represents dtype.
  constexpr bool operator==(Type dtype) const { return value == dtype; }

  /// Return the hidden-state element byte width for this dtype.
  /// \return Element byte width in bytes.
  /// \throws c10::Error if value is not a supported TensorDType.
  std::int64_t elem_size() const {
    switch (value) {
    case kBf16:
    case kFp16:
      return 2;
    case kFp32:
      return 4;
    default:
      TORCH_CHECK(false, "xpool FFN shim received an invalid TensorDType");
    }
  }
};

static_assert(sizeof(TensorDType) == sizeof(std::uint32_t));

/// Descriptor lifecycle state shared between shim and devagent kernels.
struct DescriptorStatus {
  /// On-wire descriptor lifecycle integer values.
  enum Type : std::uint32_t {
    /// Descriptor slot is available for a new request.
    kEmpty = 0,
    /// Attention-side shim has published a request for devagent polling.
    kPublished = 1,
    /// Devagent has granted the communication slot for this request.
    kGranted = 2,
    /// FFN execution completed and the output descriptor is valid.
    kDone = 3,
    /// FFN execution failed; the paired result descriptor carries an error
    /// code.
    kFailed = 4,
  };
};

/// Result error codes written into failed FFN result descriptors.
struct FfnResultErrorCode {
  /// Device-visible FFN result error code values.
  enum Type : std::uint32_t {
    /// Result completed successfully.
    kOk = 0,
    /// Arena shutdown failed an in-flight request slot.
    kShutdown = 1,
    /// The requested executor path has not been implemented.
    kNotImplemented = 2,
  };
};

/// Device timestamps for one cross-process transport request.
/// Every timestamp field is a nanosecond value from the device-global timer;
/// zero denotes a phase that has not completed.
struct alignas(8) TransportTraceRecord {
  /// Monotonic request identity; zero denotes an unused record.
  std::uint64_t trace_id;
  /// Request slot used by this trace.
  std::uint64_t slot;
  /// Token rows carried by this request.
  std::uint64_t num_tokens;
  /// Instance request kernel entry timestamp in nanoseconds.
  std::uint64_t request_begin;
  /// Free-slot acquisition timestamp in nanoseconds.
  std::uint64_t slot_claimed;
  /// Input staging completion timestamp in nanoseconds.
  std::uint64_t input_staged;
  /// Used-queue publication timestamp in nanoseconds.
  std::uint64_t request_published;
  /// Devagent used-queue dequeue timestamp in nanoseconds.
  std::uint64_t devagent_dequeued;
  /// Descriptor grant timestamp in nanoseconds.
  std::uint64_t descriptor_granted;
  /// Executor entry timestamp in nanoseconds.
  std::uint64_t executor_begin;
  /// Executor completion timestamp in nanoseconds.
  std::uint64_t executor_end;
  /// Result publication timestamp in nanoseconds.
  std::uint64_t result_published;
  /// Instance result observation timestamp in nanoseconds.
  std::uint64_t result_observed;
  /// Output copy completion timestamp in nanoseconds.
  std::uint64_t output_copied;
  /// Slot recycle timestamp in nanoseconds and record completion marker.
  std::uint64_t slot_recycled;
};

/// Host snapshot copied from one transport observer ring at arena destruction.
struct TransportTraceSnapshot {
  /// Number of trace identities allocated since arena creation.
  std::uint64_t sequence;
  /// Number of trace rows overwritten after ring capacity was exhausted.
  std::uint64_t dropped;
  /// Trace rows in ring storage order; empty when observation was disabled.
  std::vector<TransportTraceRecord> records;
};

/// Header fields shared by all device-visible descriptors.
struct DescriptorHeader {
  /// ABI version; producer writes kAbiVersion and consumer rejects mismatches.
  std::uint32_t abi_version;
};

/// State fields shared by request and result descriptor state machines.
struct DescriptorState {
  /// On-wire DescriptorStatus value for this descriptor.
  std::uint32_t status;
  /// Communication-slot id owned by this descriptor state.
  std::uint32_t slot_id;
};

/// Request metadata stamped by the attention-side shim for FFN execution.
struct FfnRequestMetadata {
  /// Integer runtime instance index from xpool config declaration order.
  std::uint32_t instance_index;
  /// Decoder layer id whose FFN implementation should consume this request.
  std::uint32_t layer_id;
  /// On-wire XPoolForwardMode value for decode, extend, or idle requests.
  std::uint32_t forward_mode;
  /// On-wire FfnCollectivePolicy value selecting full or partial FFN output.
  std::uint32_t collective_policy;
  /// On-wire DpPaddingMode value already selected by the producer runtime.
  std::uint32_t dp_padding_mode;
  /// Attention tensor-parallel rank from producer runtime topology.
  std::uint32_t atn_tp_rank;
  /// Attention tensor-parallel size from producer runtime topology.
  std::uint32_t atn_tp_size;
  /// Attention data-parallel rank from producer runtime topology.
  std::uint32_t atn_dp_rank;
  /// Attention data-parallel size from producer runtime topology.
  std::uint32_t atn_dp_size;
  /// Token rows in the producer's global DP buffer after padding.
  std::uint32_t global_dp_buffer_len;

  /// Validate request-level FFN shim invariants.
  /// \param rank Rank-local process index within the instance.
  /// \param global_num_tokens_gpu Optional per-DP-rank token-count tensor
  /// supplied by the producer runtime.
  /// \throws c10::Error if the request cannot be executed by the selected shim
  /// path, if DP padding needs token counts and they are absent, or if present
  /// token counts cannot be staged into the transport arena.
  void validate(std::int64_t rank,
                const std::optional<at::Tensor> &global_num_tokens_gpu) const {
    TORCH_CHECK(rank >= 0, "xpool FFN shim requires a non-negative rank");
    TORCH_CHECK(XPoolForwardMode::is_valid(forward_mode),
                "xpool FFN shim only supports DECODE, EXTEND, and IDLE "
                "forward modes");
    TORCH_CHECK(DpPaddingMode::is_valid(dp_padding_mode),
                "xpool FFN shim received an unsupported DP padding mode");
    TORCH_CHECK(FfnCollectivePolicy::is_valid(collective_policy),
                "xpool FFN shim received an unsupported collective policy");
    TORCH_CHECK(atn_tp_rank < atn_tp_size,
                "xpool transport FFN shim attention TP rank is out of range");
    TORCH_CHECK(atn_dp_rank < atn_dp_size,
                "xpool transport FFN shim attention DP rank is out of range");
    const bool has_global_num_tokens = global_num_tokens_gpu.has_value() &&
                                       global_num_tokens_gpu->defined() &&
                                       global_num_tokens_gpu->numel() != 0;
    if (!has_global_num_tokens) {
      TORCH_CHECK(dp_padding_mode ==
                      static_cast<std::uint32_t>(DpPaddingMode::kNone),
                  "xpool FFN shim DP padding requires global_num_tokens_gpu");
      return;
    }
    const at::Tensor &token_counts = *global_num_tokens_gpu;
    TORCH_CHECK(token_counts.is_cuda(),
                "xpool FFN shim DP token counts must be a CUDA tensor");
    TORCH_CHECK(token_counts.is_contiguous(),
                "xpool FFN shim DP token counts must be contiguous");
    TORCH_CHECK(token_counts.dim() == 1,
                "xpool FFN shim DP token counts must be a 1D tensor");
    TORCH_CHECK(token_counts.scalar_type() == at::kInt ||
                    token_counts.scalar_type() == at::kLong,
                "xpool FFN shim DP token counts must be int32 or int64");
    TORCH_CHECK(token_counts.numel() == static_cast<std::int64_t>(atn_dp_size),
                "xpool FFN shim DP token counts must have one entry per "
                "attention DP rank");
  }
};

/// Arena offsets referenced by one FFN request descriptor.
struct FfnArenaOffsets {
  /// Byte offset of the input hidden-state tensor inside the shared arena.
  std::uint64_t input_offset;
  /// Byte offset of the output hidden-state tensor inside the shared arena.
  std::uint64_t output_offset;
  /// Byte offset of staged DP token counts, or zero when absent.
  std::uint64_t dp_token_counts_offset;
};

/// Tensor facts referenced by one FFN request descriptor.
struct FfnTensorMetadata {
  /// On-wire TensorDType value for input and output hidden-state tensors.
  TensorDType dtype;
  /// Number of token rows in the contiguous [num_tokens, hidden_size] tensor.
  std::uint32_t num_tokens;
  /// Hidden dimension columns in the FFN input/output tensor.
  std::uint32_t hidden_size;

  /// Derive descriptor tensor metadata from host-side tensors.
  /// \param hidden_states Hidden-state tensor shaped [tokens, hidden_size].
  /// \return Device-visible tensor metadata for one FFN request descriptor.
  /// \throws c10::Error if hidden_states is not a contiguous supported 2D
  /// tensor or if tensor sizes overflow descriptor fields.
  static FfnTensorMetadata parse(const at::Tensor &hidden_states) {
    TORCH_CHECK(hidden_states.dim() == 2, "xpool FFN shim expects a 2D tensor");
    TORCH_CHECK(hidden_states.is_contiguous(),
                "xpool FFN shim expects a contiguous tensor");
    return FfnTensorMetadata{
        TensorDType::parse(hidden_states.scalar_type()),
        XPOOL_CHECKED_U32(hidden_states.size(0)),
        XPOOL_CHECKED_U32(hidden_states.size(1)),
    };
  }
};

/// Request descriptor written by the attention-side shim and polled on device.
struct alignas(8) FfnRequestDescriptor {
  /// ABI header shared by all descriptors.
  DescriptorHeader header;
  /// Slot state and id for the request descriptor state machine.
  DescriptorState state;
  /// FFN request metadata consumed by the devagent executor.
  FfnRequestMetadata request_metadata;
  /// Tensor facts consumed by the devagent executor.
  FfnTensorMetadata tensor_metadata;
  /// Shared-arena offsets for staged request payloads.
  FfnArenaOffsets offsets;
  /// Monotonic observer trace id, or zero when transport observation is
  /// disabled.
  std::uint64_t trace_id;
};

/// Result descriptor written by the devagent after FFN execution advances.
struct alignas(8) FfnResultDescriptor {
  /// ABI header shared by all descriptors.
  DescriptorHeader header;
  /// Slot state and id for the result descriptor state machine.
  DescriptorState state;
  /// Native executor error code; zero means success.
  std::uint32_t error_code;
  /// Byte offset of the completed output tensor inside the shared arena.
  std::uint64_t output_offset;
};

static_assert(std::is_standard_layout_v<DescriptorHeader>);
static_assert(std::is_standard_layout_v<DescriptorState>);
static_assert(std::is_standard_layout_v<TensorDType>);
static_assert(std::is_standard_layout_v<FfnRequestMetadata>);
static_assert(std::is_standard_layout_v<FfnTensorMetadata>);
static_assert(std::is_standard_layout_v<FfnArenaOffsets>);
static_assert(std::is_standard_layout_v<FfnRequestDescriptor>);
static_assert(std::is_standard_layout_v<FfnResultDescriptor>);
static_assert(std::is_standard_layout_v<TransportTraceRecord>);
static_assert(std::is_trivially_copyable_v<TransportTraceRecord>);

/// Native byte size of FfnRequestDescriptor for ABI parity tests.
inline constexpr std::uint32_t kFfnRequestDescriptorBytes =
    sizeof(FfnRequestDescriptor);
/// Native byte size of FfnResultDescriptor for ABI parity tests.
inline constexpr std::uint32_t kFfnResultDescriptorBytes =
    sizeof(FfnResultDescriptor);

} // namespace xpool::abi
