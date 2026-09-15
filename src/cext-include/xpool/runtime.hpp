#pragma once

/// \file xpool/runtime.hpp
/// \brief Process-wide native runtime identity and role enforcement.

#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string_view>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

namespace xpool {

/// Process role selected during native runtime initialization.
enum class RuntimeRole : std::uint32_t {
  Daemon = 1,
  Instance = 2,
  AtnAgent = 3,
  FfnAgent = 4,
};

/// Return whether a runtime role belongs to the closed native vocabulary.
constexpr bool is_valid(RuntimeRole value) {
  return value == RuntimeRole::Daemon || value == RuntimeRole::Instance || value == RuntimeRole::AtnAgent ||
         value == RuntimeRole::FfnAgent;
}

/// Return the stable lowercase diagnostic name of one runtime role.
/// \throws c10::Error when role is outside the closed vocabulary.
std::string_view name(RuntimeRole role);

/// Process-global native runtime identity shared by operator namespaces.
///
/// Function-local static destruction releases only host identity state; this
/// type never owns CUDA or NVSHMEM resources.
class RuntimeState {
public:
  /// Return the process-lifetime native runtime identity.
  static RuntimeState &singleton() {
    static RuntimeState runtime;
    return runtime;
  }

  /// Lock this process to one role and its optional CUDA device.
  /// The first successful call fixes both values for the process lifetime;
  /// identical later calls are idempotent.
  /// \throws c10::Error if the role/device pair is invalid or differs from a
  /// previous initialization.
  void initialize(RuntimeRole role, const std::optional<c10::DeviceIndex> &cuda_device);

  /// Return the process role fixed during initialization.
  /// \throws c10::Error if the process is uninitialized.
  RuntimeRole role() const;

  /// Require one exact runtime role for an operator.
  /// \throws c10::Error if the process is uninitialized or has another role.
  void require_role(RuntimeRole expected, std::string_view op_name) const { require_role({expected}, op_name); }

  /// Require any one of several runtime roles for an operator.
  /// \throws c10::Error if the process is uninitialized or has another role.
  void require_role(std::initializer_list<RuntimeRole> expected, std::string_view op_name) const;

  /// Return the CUDA device fixed during process initialization.
  /// \throws c10::Error if the process has no CUDA device.
  c10::DeviceIndex cuda_device(std::string_view op_name) const;

private:
  RuntimeState() = default;
  ~RuntimeState() = default;

  RuntimeState(const RuntimeState &) = delete;
  RuntimeState &operator=(const RuntimeState &) = delete;
  RuntimeState(RuntimeState &&) = delete;
  RuntimeState &operator=(RuntimeState &&) = delete;

  /// Guards role and CUDA-device initialization and reads.
  mutable std::mutex mutex_;
  /// Role fixed by the first successful initialization attempt.
  std::optional<RuntimeRole> role_;
  /// CUDA device fixed with role, or null for the daemon role.
  std::optional<c10::DeviceIndex> device_;
};

} // namespace xpool
