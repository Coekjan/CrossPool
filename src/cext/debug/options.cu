#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <mutex>
#include <optional>

#include <xpool/debug/options.hpp>

namespace xpool::debug {

__device__ __constant__ DebugOptions options_d{};

namespace {

DebugOptions options_h{};
std::optional<c10::DeviceIndex> configured_cuda_device;
std::mutex options_mutex;

} // namespace

void configure(const DebugOptions &debug_options, c10::DeviceIndex cuda_device) {
  TORCH_CHECK(cuda_device >= 0, "xpool debug options require a non-negative CUDA device");
  std::lock_guard<std::mutex> lock(options_mutex);
  if (configured_cuda_device.has_value()) {
    TORCH_CHECK(*configured_cuda_device == cuda_device, "xpool debug CUDA device differs from the first configure "
                                                        "call");
    TORCH_CHECK(options_h == debug_options, "xpool debug options differ from the first configure call");
    return;
  }
  c10::cuda::CUDAGuard device_guard(cuda_device);
  C10_CUDA_CHECK(cudaMemcpyToSymbol(options_d, &debug_options, sizeof(debug_options)));
  options_h = debug_options;
  configured_cuda_device = cuda_device;
}

DebugOptions options() {
  std::lock_guard<std::mutex> lock(options_mutex);
  return options_h;
}

} // namespace xpool::debug
