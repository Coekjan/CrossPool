#pragma once

/// \file xpool/debug/options.hpp
/// \brief Host-side native debug option state.

#include <c10/core/Device.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <xpool/macros.hpp>

namespace xpool::debug {

/// Native loopback execution site selected during process initialization.
struct LoopbackSite {
  /// Process-local loopback site values copied to CUDA constant memory.
  enum Type : std::uint32_t {
    /// Loopback is disabled.
    None = 0,
    /// Execute directly in the SGLang instance process.
    Instance = 1,
    /// Execute in the local attention atnagent transport kernel.
    AtnAgent = 2,
    /// Execute in the remote FfnAgent kernel.
    FfnAgent = 3,
  };
};

/// Typed native loopback options installed in host and device state.
struct LoopbackOptions {
  /// Whether loopback execution is enabled.
  bool enable;
  /// Process-local execution site selected for loopback work.
  LoopbackSite::Type site;

  /// Return whether loopback execution is enabled.
  /// \return True when loopback execution is enabled.
  XPOOL_HOST_DEVICE_FN
  constexpr bool enabled() const { return enable; }

  /// Return whether two loopback option sets are identical.
  /// \return True when enable and site match.
  constexpr bool operator==(const LoopbackOptions &) const = default;
};

/// Typed native trace-observer options installed in host and device state.
struct TraceOptions {
  /// Whether trace collection is enabled.
  bool enable;
  /// Configured ring capacity; zero denotes the all-disabled default state.
  std::size_t trace_capacity;

  /// Return the effective native trace capacity.
  /// \return Configured capacity when enabled, otherwise zero.
  XPOOL_HOST_DEVICE_FN
  constexpr std::size_t capacity() const { return enable ? trace_capacity : 0U; }

  /// Return whether two trace option sets are identical.
  /// \return True when enablement and capacity match.
  constexpr bool operator==(const TraceOptions &) const = default;
};

/// Process-wide typed native debug options installed during initialization.
struct DebugOptions {
  /// Debug loopback execution settings.
  LoopbackOptions loopback;
  /// Local CUDA IPC transport trace settings.
  TraceOptions transport_observer;
  /// Cross-Agent NVSHMEM Fabric trace settings.
  TraceOptions fabric_observer;

  /// Construct process-wide debug options with every feature disabled.
  /// \post Loopback is disabled with site None and both observer capacities
  /// are zero.
  XPOOL_HOST_DEVICE_FN
  constexpr DebugOptions()
      : loopback{false, LoopbackSite::None}, transport_observer{false, 0U}, fabric_observer{false, 0U} {}

  /// Parse and validate one complete host-side debug JSON snapshot.
  /// \param json Native-only JSON emitted by the Python config facade.
  /// \return Typed options safe to copy into host and device runtime state.
  /// \throws c10::Error for missing, unknown, malformed, or inconsistent
  /// options.
  static DebugOptions parse(std::string_view json);

  /// Return whether two option sets are identical.
  /// \return True when every typed option field matches.
  constexpr bool operator==(const DebugOptions &) const = default;
};

static_assert(std::is_trivially_copyable_v<DebugOptions>);

/// Configure process-wide host and device debug options together.
/// \param debug_options Typed debug options resolved from process config.
/// \param cuda_device CUDA device whose constant memory receives the options.
/// \throws c10::Error if the device is invalid, CUDA update fails, or a later
/// call differs from the first explicit configuration.
void configure(const DebugOptions &debug_options, c10::DeviceIndex cuda_device);

/// Return a snapshot of the host-side native debug options.
/// \return Explicitly configured options, or the all-disabled default before
/// configuration.
#if !defined(__CUDACC__)
DebugOptions options();
#endif

} // namespace xpool::debug
