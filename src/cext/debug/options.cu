#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <mutex>

#include <xpool/debug/options.cuh>
#include <xpool/debug/options.hpp>

namespace xpool::debug {

// Device-side debug options installed for transport kernels.
__device__ __constant__ xpool::abi::DebugOptions g_debug_options{0};

namespace {

xpool::abi::DebugOptions g_debug_options_host{0};
bool g_debug_options_initialized = false;
std::mutex g_debug_options_mutex;

} // namespace

void init(std::int64_t cuda_device, std::int64_t debug_options) {
  TORCH_CHECK(cuda_device >= 0,
              "xpool debug options require a non-negative CUDA device");
  xpool::abi::DebugOptions options =
      xpool::abi::DebugOptions::parse(debug_options);
  std::lock_guard<std::mutex> lock(g_debug_options_mutex);
  if (g_debug_options_initialized) {
    TORCH_CHECK(g_debug_options_host.raw == options.raw,
                "xpool init debug options differ from the process-wide options "
                "installed by the first init call");
  }
  c10::cuda::CUDAGuard device_guard(static_cast<int>(cuda_device));
  C10_CUDA_CHECK(
      cudaMemcpyToSymbol(g_debug_options, &options, sizeof(options)));
  g_debug_options_host = options;
  g_debug_options_initialized = true;
}

xpool::abi::DebugOptions options() {
  std::lock_guard<std::mutex> lock(g_debug_options_mutex);
  return g_debug_options_host;
}

} // namespace xpool::debug
