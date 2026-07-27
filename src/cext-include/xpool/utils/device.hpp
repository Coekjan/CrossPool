#pragma once

/// \file xpool/utils/device.hpp
/// \brief Host-side CUDA device utility helpers.

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>

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
  /// \param flags Flags passed to cudaStreamCreateWithFlags.
  /// \return Owner of the newly created stream.
  static OwnedCudaStream create(unsigned int flags = cudaStreamNonBlocking) {
    OwnedCudaStream stream;
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream.stream_, flags));
    return stream;
  }

  /// Destroy the owned stream with best-effort cleanup.
  ~OwnedCudaStream() {
    if (stream_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaStreamDestroy(stream_));
    }
  }

  OwnedCudaStream(const OwnedCudaStream &) = delete;
  OwnedCudaStream &operator=(const OwnedCudaStream &) = delete;

  /// Move a stream owner, leaving the source empty.
  /// \param other Owner whose stream should be transferred here.
  OwnedCudaStream(OwnedCudaStream &&other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

  /// Replace this empty owner with another stream owner.
  /// \param other Owner whose stream should be transferred here.
  /// \return This stream owner.
  /// \pre This owner is empty and has already been explicitly destroyed.
  OwnedCudaStream &operator=(OwnedCudaStream &&other) {
    if (this != &other) {
      TORCH_CHECK(stream_ == nullptr, "a live CUDA stream cannot be replaced by move");
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  /// Return whether this owner holds a CUDA stream.
  explicit operator bool() const noexcept { return stream_ != nullptr; }

  /// Return the raw CUDA stream handle.
  /// \return Owned stream, or nullptr when empty.
  cudaStream_t get() const { return stream_; }

  /// Enqueue one stream-ordered 32-bit write to a device address.
  /// \param address Device address receiving value.
  /// \param value Value published after preceding stream work.
  /// \throws c10::Error if the owner is empty, address is null, or the CUDA
  /// Driver rejects the operation.
  void write_value(std::uint32_t *address, std::uint32_t value) const {
    TORCH_CHECK(stream_ != nullptr, "xpool cannot write through an empty CUDA stream");
    TORCH_CHECK(address != nullptr, "xpool CUDA stream write requires a device address");
    const auto result = cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_),
                                             reinterpret_cast<CUdeviceptr>(address), value,
                                             CU_STREAM_WRITE_VALUE_DEFAULT);
    TORCH_CHECK(result == CUDA_SUCCESS, "xpool CUDA stream-ordered write failed: CUDA driver error ",
                static_cast<int>(result));
  }

  /// Attempt one stream-ordered 32-bit device write during cleanup.
  /// \param address Device address receiving value.
  /// \param value Value published after preceding stream work.
  /// \return True when the Driver accepted the write.
  [[nodiscard]] bool try_write_value(std::uint32_t *address, std::uint32_t value) const noexcept {
    if (stream_ == nullptr || address == nullptr) {
      return false;
    }
    return cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_),
                                reinterpret_cast<CUdeviceptr>(address), value,
                                CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS;
  }

  /// Query whether all work submitted to the stream has completed.
  /// \return True when the stream is empty or complete; false when work is
  /// still pending.
  /// \throws c10::Error if CUDA reports an error other than pending work.
  [[nodiscard]] bool query() const {
    if (stream_ == nullptr) {
      return true;
    }
    const auto status = cudaStreamQuery(stream_);
    if (status == cudaSuccess) {
      return true;
    }
    if (status == cudaErrorNotReady) {
      (void)cudaGetLastError();
      return false;
    }
    C10_CUDA_CHECK(status);
    return false;
  }

  /// Destroy the owned stream and surface CUDA errors.
  /// \throws c10::Error if cudaStreamDestroy fails.
  void destroy() {
    if (stream_ == nullptr) {
      return;
    }
    C10_CUDA_CHECK(cudaStreamDestroy(stream_));
    stream_ = nullptr;
  }

private:
  cudaStream_t stream_ = nullptr;
};

} // namespace xpool::utils::device
