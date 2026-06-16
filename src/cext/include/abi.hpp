#pragma once

// Native mirror of src/xpool/abi.py.
//
// Keep this header and the Python struct packers in lockstep until xpool grows
// generated ABI bindings. Current tests cover Python packing and native
// byte-size parity; add explicit enum, offset, and alignment parity checks
// before treating those properties as stable ABI guarantees.

#include <cstdint>

namespace xpool {

inline constexpr std::uint32_t kAbiVersion = 1;

enum class ForwardMode : std::uint32_t {
  kDecode = 1,
  kExtend = 2,
};

enum class TensorDType : std::uint32_t {
  kBf16 = 1,
  kFp16 = 2,
  kFp32 = 3,
};

enum class DescriptorStatus : std::uint32_t {
  kEmpty = 0,
  kPublished = 1,
  kGranted = 2,
  kDone = 3,
  kFailed = 4,
};

struct alignas(8) FfnRequestDescriptor {
  std::uint32_t abi_version;
  std::uint32_t descriptor_bytes;
  std::uint64_t sequence;
  std::uint32_t instance_id;
  std::uint32_t model_id;
  std::uint32_t layer_id;
  std::uint32_t forward_mode;
  std::uint32_t dtype;
  std::uint32_t status;
  std::uint64_t input_ptr;
  std::uint64_t output_ptr;
  std::uint64_t scratch_ptr;
  std::uint32_t num_tokens;
  std::uint32_t hidden_size;
  std::uint32_t slot_id;
  std::uint32_t reserved;
};

struct alignas(8) FfnResultDescriptor {
  std::uint32_t abi_version;
  std::uint32_t descriptor_bytes;
  std::uint64_t sequence;
  std::uint32_t status;
  std::uint32_t error_code;
  std::uint32_t slot_id;
  std::uint32_t reserved;
  std::uint64_t output_ptr;
};

inline constexpr std::uint32_t kFfnRequestDescriptorBytes =
    sizeof(FfnRequestDescriptor);
inline constexpr std::uint32_t kFfnResultDescriptorBytes =
    sizeof(FfnResultDescriptor);

} // namespace xpool
