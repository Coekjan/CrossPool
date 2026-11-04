#pragma once

/// \file xpool/utils/cuda.hpp
/// \brief Host-side CUDA operations and stream ownership.

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace xpool::utils::cuda {

/// Return the current process-local CUDA device index.
/// \throws c10::AcceleratorError if the CUDA query fails.
[[nodiscard]] int current_device();

/// Query one attribute of an explicit process-local CUDA device.
/// \throws c10::AcceleratorError if the CUDA query fails.
[[nodiscard]] int device_attribute(cudaDeviceAttr attribute, int device);

/// Query the active blocks per multiprocessor for a kernel on the current device.
/// \throws c10::AcceleratorError if the occupancy query fails.
[[nodiscard]] int max_active_blocks_per_multiprocessor(const void *function, int block_size,
                                                       std::size_t dynamic_shared_memory_bytes);

/// Query the CUDA attributes of a borrowed pointer without changing its owner.
/// \throws c10::AcceleratorError if the CUDA query fails.
[[nodiscard]] cudaPointerAttributes pointer_attributes(const void *pointer);

/// Preserve a failing Driver result and its caller location in Torch's typed error.
/// \throws c10::AcceleratorError when result differs from CUDA_SUCCESS.
void check_driver_result(CUresult result, std::source_location location = std::source_location::current());

/// Move-only owner for a host-created CUDA stream.
///
/// The caller must install the intended device before creating, querying,
/// or destroying a live stream. Destruction performs best-effort cleanup and
/// never throws; call destroy() on the normal path when CUDA errors should be
/// surfaced.
class OwnedCudaStream {
public:
  /// Construct an empty CUDA stream owner.
  OwnedCudaStream() = default;

  /// Create an owned nonblocking CUDA stream on the current device.
  /// \throws c10::Error when CUDA cannot create the stream.
  static OwnedCudaStream create();

  /// Destroy the owned stream with best-effort cleanup.
  ~OwnedCudaStream();

  OwnedCudaStream(const OwnedCudaStream &) = delete;
  OwnedCudaStream &operator=(const OwnedCudaStream &) = delete;

  /// Move a stream owner, leaving the source empty.
  OwnedCudaStream(OwnedCudaStream &&other) noexcept;

  /// Replace this empty owner with another stream owner.
  /// \pre This owner is empty.
  OwnedCudaStream &operator=(OwnedCudaStream &&other);

  /// Return whether this owner holds a CUDA stream.
  explicit operator bool() const noexcept { return stream_ != nullptr; }

  /// Return the raw CUDA stream handle.
  cudaStream_t get() const { return stream_; }

  /// Enqueue one stream-ordered 32-bit write to a device address.
  /// \throws c10::Error if the owner is empty, address is null, or the CUDA
  /// Driver rejects the operation.
  void write_value(std::uint32_t *address, std::uint32_t value) const;

  /// Query whether all work submitted to the stream has completed.
  /// \throws c10::Error if CUDA reports an error other than pending work.
  [[nodiscard]] bool query() const;

  /// Confirm completion of all work on this stream.
  /// \throws c10::Error if the owner is empty or synchronization fails.
  void synchronize() const;

  /// Destroy the owned stream and surface CUDA errors.
  /// \throws c10::Error if cudaStreamDestroy fails.
  void destroy();

private:
  cudaStream_t stream_ = nullptr;
};

} // namespace xpool::utils::cuda
