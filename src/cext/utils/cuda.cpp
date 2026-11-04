#include <xpool/utils/cuda.hpp>

#include <utility>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <cuda.h>

#include <xpool/utils/checked.hpp>

namespace xpool::utils::cuda {

CaptureAllocation::CaptureAllocation(std::string key, std::uintptr_t address, std::size_t bytes)
    : key(std::move(key)), address(address), bytes(bytes) {
  TORCH_CHECK(!this->key.empty() && address != 0 && bytes != 0,
              "xpool capture allocation must have a key and nonempty interval");
  static_cast<void>(xpool::utils::checked::sum(address, static_cast<std::uintptr_t>(bytes)));
}

int current_device() {
  auto device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

int device_attribute(cudaDeviceAttr attribute, int device) {
  auto value = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&value, attribute, device));
  return value;
}

int max_active_blocks_per_multiprocessor(const void *function, int block_size,
                                         std::size_t dynamic_shared_memory_bytes) {
  auto blocks = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, function, block_size, dynamic_shared_memory_bytes));
  return blocks;
}

cudaPointerAttributes pointer_attributes(const void *pointer) {
  auto attributes = cudaPointerAttributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
  return attributes;
}

void check_driver_result(CUresult result, std::source_location location) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *description = nullptr;
  static_cast<void>(cuGetErrorString(result, &description));
  throw c10::AcceleratorError{{location.function_name(), location.file_name(), location.line()},
                              static_cast<std::int32_t>(result),
                              c10::str("CUDA driver error: ", description == nullptr ? "unknown error" : description)};
}

OwnedCudaStream OwnedCudaStream::create() {
  auto stream = OwnedCudaStream{};
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream.stream_, cudaStreamNonBlocking));
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
  check_driver_result(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_), reinterpret_cast<CUdeviceptr>(address),
                                           value, CU_STREAM_WRITE_VALUE_DEFAULT));
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

void OwnedCudaStream::synchronize() const {
  TORCH_CHECK(stream_ != nullptr, "xpool capture stream is closed");
  C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void OwnedCudaStream::destroy() {
  if (stream_ == nullptr) {
    return;
  }
  C10_CUDA_CHECK(cudaStreamDestroy(stream_));
  stream_ = nullptr;
}

} // namespace xpool::utils::cuda
