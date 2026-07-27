#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>
#include <nvshmemx.h>

#include <xpool/fabric/module.hpp>

namespace {

__global__ void fabric_module_anchor() {}

} // namespace

namespace xpool::fabric {

FabricModuleRegistration FabricModuleRegistration::create() {
  // NVSHMEM device entry points are linked into this CUDA module. Registration
  // must follow host-library initialization and outlive every launched Fabric
  // kernel that can call those entry points.
  auto function = cudaFunction_t{nullptr};
  C10_CUDA_CHECK(cudaGetFuncBySymbol(&function, reinterpret_cast<const void *>(fabric_module_anchor)));
  auto resolved_module = CUmodule{nullptr};
  const auto driver_status = cuFuncGetModule(&resolved_module, reinterpret_cast<CUfunction>(function));
  TORCH_CHECK(driver_status == CUDA_SUCCESS,
              "xpool failed to resolve the Fabric CUDA module: ", static_cast<int>(driver_status));
  const auto status = nvshmemx_cumodule_init(resolved_module);
  TORCH_CHECK(status == 0, "xpool failed to register the Fabric CUDA module: ", status);
  return FabricModuleRegistration{resolved_module};
}

FabricModuleRegistration::~FabricModuleRegistration() {
  // Rollback paths cannot report destructor failures. Normal shutdown uses
  // destroy() before finalizing the NVSHMEM host library and checks its status.
  if (module_ != nullptr) {
    static_cast<void>(nvshmemx_cumodule_finalize(module_));
  }
}

void FabricModuleRegistration::destroy() {
  if (module_ == nullptr) {
    return;
  }
  const auto status = nvshmemx_cumodule_finalize(module_);
  TORCH_CHECK(status == 0, "xpool failed to finalize the Fabric CUDA module: ", status);
  module_ = nullptr;
}

} // namespace xpool::fabric
