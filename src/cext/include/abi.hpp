#pragma once

/// \file abi.hpp
/// \brief Native mirror of the Python FFN descriptor ABI in src/xpool/abi.py.
///
/// Keep this header and the Python struct packers in lockstep until xpool grows
/// generated ABI bindings. The descriptor fields are an inter-process ABI, so
/// every field below documents its producer, consumer, unit, and stability
/// expectation. Cross-process tensor references are arena offsets, not raw CUDA
/// virtual addresses, because CUDA IPC mappings may use different addresses in
/// the SGLang instance and in the local device agent.

#include <cstdint>

/// Native xpool ABI symbols shared by the SGLang shim and device-agent runtime.
namespace xpool {

/// Version stamped into every descriptor so incompatible producers fail closed.
inline constexpr std::uint32_t kAbiVersion = 2;

/// SGLang forward mode values accepted by the first xpool FFN shim ABI.
enum class ForwardMode : std::uint32_t {
  /// Extend/prefill-mode FFN request for a contiguous prompt-token batch.
  kExtend = 1,
  /// Decode-mode FFN request for one scheduler decode step.
  kDecode = 2,
};

/// Element dtype of the hidden-state tensor referenced by an FFN request.
enum class TensorDType : std::uint32_t {
  /// bfloat16 hidden states.
  kBf16 = 1,
  /// IEEE float16 hidden states.
  kFp16 = 2,
  /// IEEE float32 hidden states.
  kFp32 = 3,
};

/// Descriptor lifecycle state shared between shim and device-agent kernels.
enum class DescriptorStatus : std::uint32_t {
  /// Descriptor lane is available for a new request.
  kEmpty = 0,
  /// Attention-side shim has published a request for device-agent polling.
  kPublished = 1,
  /// Device agent has granted the communication slot for this request.
  kGranted = 2,
  /// FFN execution completed and the output descriptor is valid.
  kDone = 3,
  /// FFN execution failed; the paired result descriptor carries an error code.
  kFailed = 4,
};

/// Request descriptor written by the attention-side shim and polled on device.
struct alignas(8) FfnRequestDescriptor {
  /// ABI version; producer writes kAbiVersion and consumer rejects mismatches.
  std::uint32_t abi_version;
  /// Native byte size of this struct; guards Python/C++ layout drift.
  std::uint32_t descriptor_bytes;
  /// Monotonic lane sequence used to distinguish graph replays and stale slots.
  std::uint64_t sequence;
  /// Integer SGLang instance index from xpool config declaration order.
  std::uint32_t instance_id;
  /// Integer model index selecting the FFN weight set on the executor side.
  std::uint32_t model_id;
  /// Decoder layer id whose FFN implementation should consume this request.
  std::uint32_t layer_id;
  /// On-wire ForwardMode value; only plain decode and extend are supported.
  std::uint32_t forward_mode;
  /// On-wire TensorDType value for input and output hidden-state tensors.
  std::uint32_t dtype;
  /// On-wire DescriptorStatus value for the request lane state machine.
  std::uint32_t status;
  /// Byte offset of the input hidden-state tensor inside the shared arena.
  std::uint64_t input_offset;
  /// Byte offset of the output hidden-state tensor inside the shared arena.
  std::uint64_t output_offset;
  /// Byte offset of per-request scratch space, or zero when no scratch is used.
  std::uint64_t scratch_offset;
  /// Number of token rows in the contiguous [num_tokens, hidden_size] tensor.
  std::uint32_t num_tokens;
  /// Hidden dimension columns in the FFN input/output tensor.
  std::uint32_t hidden_size;
  /// Communication-slot id granted to this request and echoed in the result.
  std::uint32_t slot_id;
  /// Reserved for future ABI extension; producers must write zero.
  std::uint32_t reserved;
};

/// Result descriptor written by the device agent after FFN execution advances.
struct alignas(8) FfnResultDescriptor {
  /// ABI version; producer writes kAbiVersion and consumer rejects mismatches.
  std::uint32_t abi_version;
  /// Native byte size of this struct; guards Python/C++ layout drift.
  std::uint32_t descriptor_bytes;
  /// Request sequence completed by this result; consumers must match it.
  std::uint64_t sequence;
  /// On-wire DescriptorStatus value, normally kDone or kFailed.
  std::uint32_t status;
  /// Native executor error code; zero means success.
  std::uint32_t error_code;
  /// Communication-slot id released or failed by this result.
  std::uint32_t slot_id;
  /// Reserved for future ABI extension; producers must write zero.
  std::uint32_t reserved;
  /// Byte offset of the completed output tensor inside the shared arena.
  std::uint64_t output_offset;
};

/// Native byte size expected by Python's FfnRequestDescriptor packer.
inline constexpr std::uint32_t kFfnRequestDescriptorBytes =
    sizeof(FfnRequestDescriptor);
/// Native byte size expected by Python's FfnResultDescriptor packer.
inline constexpr std::uint32_t kFfnResultDescriptorBytes =
    sizeof(FfnResultDescriptor);

} // namespace xpool
