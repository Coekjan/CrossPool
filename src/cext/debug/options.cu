#include <xpool/debug/options.hpp>

#include <mutex>
#include <optional>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <xpool/macros.hpp>

namespace xpool::debug {

XPOOL_DEVICE_CONST Options options_d{};
Options options_h{};

namespace {

std::mutex options_mutex;
std::optional<c10::DeviceIndex> configured_cuda_device;
bool configured = false;

} // namespace

void configure(const Options &debug_options, std::optional<c10::DeviceIndex> cuda_device) {
  std::lock_guard<std::mutex> lock(options_mutex);
  if (configured) {
    TORCH_CHECK(configured_cuda_device == cuda_device, "xpool debug CUDA device differs from the first configure call");
    TORCH_CHECK(options_h == debug_options, "xpool debug options differ from the first configure call");
    return;
  }
  if (cuda_device.has_value()) {
    TORCH_CHECK(*cuda_device >= 0, "xpool debug options require a non-negative CUDA device");
    c10::cuda::CUDAGuard device_guard(*cuda_device);
    C10_CUDA_CHECK(cudaMemcpyToSymbol(options_d, &debug_options, sizeof(debug_options)));
  }
  options_h = debug_options;
  configured_cuda_device = cuda_device;
  configured = true;
}

} // namespace xpool::debug
