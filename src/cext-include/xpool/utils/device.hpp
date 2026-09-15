#pragma once

/// \file xpool/utils/device.hpp
/// \brief Host-side CUDA device utility helpers.

#include <cstdint>

#include <cuda_runtime_api.h>

namespace xpool::utils::device {

/// Move-only owner for a host-created CUDA stream.
///
/// The caller must install the intended CUDA device before creating, querying,
/// or destroying a live stream. Destruction performs best-effort cleanup and
/// never throws; call destroy() on the normal path when CUDA errors should be
/// surfaced.
class OwnedCudaStream {
public:
  /// Construct an empty CUDA stream owner.
  OwnedCudaStream() = default;

  /// Create an owned CUDA stream on the current CUDA device.
  /// \throws c10::Error when CUDA cannot create the stream.
  static OwnedCudaStream create(unsigned int flags = cudaStreamNonBlocking);

  /// Destroy the owned stream with best-effort cleanup.
  ~OwnedCudaStream();

  OwnedCudaStream(const OwnedCudaStream &) = delete;
  OwnedCudaStream &operator=(const OwnedCudaStream &) = delete;

  /// Move a stream owner, leaving the source empty.
  OwnedCudaStream(OwnedCudaStream &&other) noexcept;

  /// Replace this empty owner with another stream owner.
  /// \pre This owner is empty and has already been explicitly destroyed.
  OwnedCudaStream &operator=(OwnedCudaStream &&other);

  /// Return whether this owner holds a CUDA stream.
  explicit operator bool() const noexcept { return stream_ != nullptr; }

  /// Return the raw CUDA stream handle.
  cudaStream_t get() const { return stream_; }

  /// Enqueue one stream-ordered 32-bit write to a device address.
  /// \throws c10::Error if the owner is empty, address is null, or the CUDA
  /// Driver rejects the operation.
  void write_value(std::uint32_t *address, std::uint32_t value) const;

  /// Attempt one stream-ordered 32-bit device write during cleanup.
  [[nodiscard]] bool try_write_value(std::uint32_t *address, std::uint32_t value) const noexcept;

  /// Query whether all work submitted to the stream has completed.
  /// \throws c10::Error if CUDA reports an error other than pending work.
  [[nodiscard]] bool query() const;

  /// Destroy the owned stream and surface CUDA errors.
  /// \throws c10::Error if cudaStreamDestroy fails.
  void destroy();

private:
  cudaStream_t stream_ = nullptr;
};

} // namespace xpool::utils::device
