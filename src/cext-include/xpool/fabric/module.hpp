#pragma once

/// \file xpool/fabric/module.hpp
/// \brief Explicit NVSHMEM CUDA module registration ownership.

#include <cuda.h>

#include <c10/util/Exception.h>

#include <utility>

namespace xpool::fabric {

/// Process-local registration of CrossPool Fabric CUDA device code with NVSHMEM.
class ModuleRegistration {
public:
  /// Construct an empty module registration.
  ModuleRegistration() = default;

  /// Best-effort fallback for an active module registration.
  /// \post Normal lifecycle code must call destroy() before destruction so
  /// finalization failures remain observable.
  ~ModuleRegistration();

  ModuleRegistration(const ModuleRegistration &) = delete;
  ModuleRegistration &operator=(const ModuleRegistration &) = delete;

  /// Move one module registration and leave its source empty.
  ModuleRegistration(ModuleRegistration &&other) noexcept
      : module_(std::exchange(other.module_, nullptr)) {}

  /// Replace this empty registration by moving another registration.
  ModuleRegistration &operator=(ModuleRegistration &&other) {
    if (this != &other) {
      TORCH_CHECK(module_ == nullptr, "a live Fabric module registration cannot be replaced");
      module_ = std::exchange(other.module_, nullptr);
    }
    return *this;
  }

  /// Resolve and register the Fabric-owned CUDA module with NVSHMEM.
  /// \throws c10::Error if CUDA module resolution or NVSHMEM registration
  /// fails.
  static ModuleRegistration create();

  /// Finalize this CUDA module registration if active.
  /// \throws c10::Error if NVSHMEM module finalization fails.
  void destroy();

private:
  /// Construct the sole owning state returned by create().
  explicit ModuleRegistration(CUmodule module) : module_(module) {}

  /// CUDA module registered with the active NVSHMEM runtime.
  CUmodule module_ = nullptr;
};

} // namespace xpool::fabric
