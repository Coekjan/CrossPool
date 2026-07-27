#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string_view>

#include <xpool/abort.hpp>
#include <xpool/utils/enum.hpp>

namespace xpool {

/// Process role selected during native runtime initialization.
struct RuntimeRole {
  /// Stable runtime role values accepted by xpool.init.
  enum Type : std::uint32_t {
    Daemon = 1,
    Instance = 2,
    AtnAgent = 3,
    FfnAgent = 4,
  };

  /// Construct a validated runtime role from its same-domain enum.
  /// \param value Stable runtime role selected by bootstrap.
  /// \pre is_valid(value) is true; violation fail-stops the process.
  constexpr RuntimeRole(Type value) : RuntimeRole(static_cast<std::uint32_t>(value)) {}

  /// Explicitly construct a validated runtime role from an integer.
  /// \tparam T Non-boolean integral input type.
  /// \param value Stable runtime role selected by bootstrap.
  /// \pre is_valid(value) is true; violation fail-stops the process.
  template <xpool::utils::EnumInput<Type> T>
  explicit constexpr RuntimeRole(T value) : value_(static_cast<Type>(value)) {
    if (!is_valid(value)) {
      xpool::abort();
    }
  }

  /// Return the wrapped runtime role.
  /// \return Stable role carried by this trusted wrapper.
  constexpr Type value() const { return value_; }

  /// Return whether an integer is a defined runtime role.
  /// \tparam T This type's Type or a non-boolean integral type.
  /// \param value Raw operator argument.
  /// \return True for every declared runtime role.
  template <xpool::utils::EnumInput<Type> T> static constexpr bool is_valid(T value) {
    return xpool::utils::enum_value_equal(value, Daemon) || xpool::utils::enum_value_equal(value, Instance) ||
           xpool::utils::enum_value_equal(value, AtnAgent) || xpool::utils::enum_value_equal(value, FfnAgent);
  }

  /// Parse one host operator argument into a runtime role.
  /// \param value Raw operator argument.
  /// \return Validated runtime role.
  /// \throws c10::Error if value is not a declared role.
  static RuntimeRole parse(std::int64_t value) {
    TORCH_CHECK(is_valid(value), "xpool received an invalid runtime role");
    return RuntimeRole{value};
  }

  /// Return this role's stable lowercase diagnostic name.
  /// \return Process-role name used in diagnostics.
  std::string_view name() const {
    switch (value_) {
    case Daemon:
      return "daemon";
    case Instance:
      return "instance";
    case AtnAgent:
      return "atnagent";
    case FfnAgent:
      return "ffnagent";
    }
    xpool::abort();
  }

  /// Compare two runtime roles.
  /// \return True when both wrappers carry the same role.
  constexpr bool operator==(const RuntimeRole &) const = default;

  /// Compare this wrapper with one same-domain enum value.
  /// \param value Stable runtime role to compare.
  /// \return True when this wrapper carries value.
  constexpr bool operator==(Type value) const { return value_ == value; }

private:
  Type value_;
};

/// Process-global native runtime identity shared by operator namespaces.
///
/// Function-local static destruction releases only host identity state; this
/// type never owns CUDA or NVSHMEM resources.
class RuntimeState {
public:
  /// Return the process-lifetime native runtime identity.
  /// \return Sole runtime identity allocated for this process.
  static RuntimeState &singleton() {
    static RuntimeState runtime;
    return runtime;
  }

  /// Lock this process to one role and its optional CUDA device.
  /// \param role Runtime role selected by Python bootstrap.
  /// \param cuda_device CUDA device for GPU roles, or null for the daemon.
  /// \throws c10::Error if the role/device pair is invalid or differs from a
  /// previous initialization.
  void initialize(RuntimeRole role, const std::optional<c10::DeviceIndex> &cuda_device);

  /// Require one exact runtime role for an operator.
  /// \param expected Required process role.
  /// \param op_name Operator name included in diagnostics.
  /// \throws c10::Error if the process is uninitialized or has another role.
  void require_role(RuntimeRole expected, std::string_view op_name) const { require_role({expected}, op_name); }

  /// Require any one of several runtime roles for an operator.
  /// \param expected Accepted process roles using OR semantics.
  /// \param op_name Operator name included in diagnostics.
  /// \throws c10::Error if the process is uninitialized or has another role.
  void require_role(std::initializer_list<RuntimeRole> expected, std::string_view op_name) const;

  /// Return the CUDA device fixed during process initialization.
  /// \param op_name Operator name included in diagnostics.
  /// \return Non-negative CUDA device ordinal owned by this process.
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
