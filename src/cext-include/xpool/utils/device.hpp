#pragma once

/// \file xpool/utils/device.hpp
/// \brief Host-side CUDA device utility helpers.

#include <c10/core/Device.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace xpool::utils::device {

/// Convert a public int64 CUDA device id to c10's compact device index type.
/// \param cuda_device Non-negative CUDA device id.
/// \return CUDA device id narrowed to c10::DeviceIndex.
/// \throws c10::Error if cuda_device is negative or outside DeviceIndex range.
inline c10::DeviceIndex cuda_device_index(std::int64_t cuda_device) {
  TORCH_CHECK(cuda_device >= 0, "xpool CUDA device id must be non-negative");
  TORCH_CHECK(cuda_device <= std::numeric_limits<c10::DeviceIndex>::max(),
              "xpool CUDA device id exceeds c10::DeviceIndex range");
  return static_cast<c10::DeviceIndex>(cuda_device);
}

/// RAII owner for a host-created CUDA stream.
///
/// The caller must install the intended CUDA device before constructing this
/// object. Destruction performs best-effort stream cleanup and never throws;
/// call reset() on the normal path when CUDA stream-destroy errors should be
/// surfaced.
class ScopedCudaStream {
public:
  /// Create a CUDA stream on the current CUDA device.
  /// \param flags Flags passed to cudaStreamCreateWithFlags.
  explicit ScopedCudaStream(unsigned int flags = cudaStreamNonBlocking) {
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
  }

  /// Destroy the owned stream with best-effort cleanup.
  ~ScopedCudaStream() { reset_noexcept(); }

  ScopedCudaStream(const ScopedCudaStream &) = delete;
  ScopedCudaStream &operator=(const ScopedCudaStream &) = delete;

  /// Move a stream owner, leaving the source empty.
  /// \param other Owner whose stream should be transferred here.
  ScopedCudaStream(ScopedCudaStream &&other) noexcept
      : stream_(std::exchange(other.stream_, nullptr)) {}

  /// Replace this owner with another stream owner.
  /// \param other Owner whose stream should be transferred here.
  /// \return This stream owner.
  ScopedCudaStream &operator=(ScopedCudaStream &&other) noexcept {
    if (this != &other) {
      reset_noexcept();
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  /// Return the raw CUDA stream handle.
  /// \return Owned stream, or nullptr after reset or move.
  cudaStream_t get() const { return stream_; }

  /// Release the owned stream without destroying it.
  /// \return Previously owned stream, or nullptr after reset or move.
  cudaStream_t release() { return std::exchange(stream_, nullptr); }

  /// Destroy the owned stream and surface CUDA errors.
  /// \throws c10::Error if cudaStreamDestroy fails.
  void reset() {
    if (stream_ == nullptr) {
      return;
    }
    cudaStream_t stream = std::exchange(stream_, nullptr);
    C10_CUDA_CHECK(cudaStreamDestroy(stream));
  }

private:
  void reset_noexcept() noexcept {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  cudaStream_t stream_ = nullptr;
};

} // namespace xpool::utils::device
