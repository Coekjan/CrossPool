#include <xpool/fabric/module.hpp>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <nvshmemx.h>

#include <xpool/macros.hpp>

namespace {

XPOOL_KERNEL_FN void fabric_module_anchor() {}

} // namespace

namespace xpool::fabric {

ModuleRegistration ModuleRegistration::create() {
  // NVSHMEM device entry points are linked into this CUDA module. Registration
  // must follow host-library initialization and outlive every launched Fabric
  // kernel that can call those entry points.
  auto function = cudaFunction_t{nullptr};
  C10_CUDA_CHECK(cudaGetFuncBySymbol(&function, reinterpret_cast<const void *>(fabric_module_anchor)));
  auto resolved_module = CUmodule{nullptr};
  C10_CUDA_DRIVER_CHECK(cuFuncGetModule(&resolved_module, reinterpret_cast<CUfunction>(function)));
  const auto status = nvshmemx_cumodule_init(resolved_module);
  TORCH_CHECK(status == 0, "xpool failed to register the Fabric CUDA module: ", status);
  return ModuleRegistration{resolved_module};
}

ModuleRegistration::~ModuleRegistration() {
  // Rollback paths cannot report destructor failures. Normal shutdown uses
  // destroy() before finalizing the NVSHMEM host library and checks its status.
  if (module_ != nullptr) {
    static_cast<void>(nvshmemx_cumodule_finalize(module_));
  }
}

void ModuleRegistration::destroy() {
  if (module_ == nullptr) {
    return;
  }
  const auto status = nvshmemx_cumodule_finalize(module_);
  TORCH_CHECK(status == 0, "xpool failed to finalize the Fabric CUDA module: ", status);
  module_ = nullptr;
}

} // namespace xpool::fabric
