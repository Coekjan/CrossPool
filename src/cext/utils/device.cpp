#include <xpool/utils/device.hpp>

#include <utility>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <cuda.h>

namespace xpool::utils::device {

OwnedCudaStream OwnedCudaStream::create(unsigned int flags) {
  auto stream = OwnedCudaStream{};
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream.stream_, flags));
  return stream;
}

OwnedCudaStream::~OwnedCudaStream() {
  if (stream_ != nullptr) {
    C10_CUDA_IGNORE_ERROR(cudaStreamDestroy(stream_));
  }
}

OwnedCudaStream::OwnedCudaStream(OwnedCudaStream &&other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}

OwnedCudaStream &OwnedCudaStream::operator=(OwnedCudaStream &&other) {
  if (this != &other) {
    TORCH_CHECK(stream_ == nullptr, "a live CUDA stream cannot be replaced by move");
    stream_ = std::exchange(other.stream_, nullptr);
  }
  return *this;
}

void OwnedCudaStream::write_value(std::uint32_t *address, std::uint32_t value) const {
  TORCH_CHECK(stream_ != nullptr, "xpool cannot write through an empty CUDA stream");
  TORCH_CHECK(address != nullptr, "xpool CUDA stream write requires a device address");
  C10_CUDA_DRIVER_CHECK(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_),
                                             reinterpret_cast<CUdeviceptr>(address), value,
                                             CU_STREAM_WRITE_VALUE_DEFAULT));
}

bool OwnedCudaStream::try_write_value(std::uint32_t *address, std::uint32_t value) const noexcept {
  if (stream_ == nullptr || address == nullptr) {
    return false;
  }
  return cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_), reinterpret_cast<CUdeviceptr>(address), value,
                              CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS;
}

bool OwnedCudaStream::query() const {
  if (stream_ == nullptr) {
    return true;
  }
  const auto status = cudaStreamQuery(stream_);
  if (status == cudaSuccess) {
    return true;
  }
  if (status == cudaErrorNotReady) {
    static_cast<void>(cudaGetLastError());
    return false;
  }
  C10_CUDA_CHECK(status);
  return false;
}

void OwnedCudaStream::destroy() {
  if (stream_ == nullptr) {
    return;
  }
  C10_CUDA_CHECK(cudaStreamDestroy(stream_));
  stream_ = nullptr;
}

} // namespace xpool::utils::device
