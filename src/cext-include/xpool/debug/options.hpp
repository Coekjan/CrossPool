#pragma once

/// \file xpool/debug/options.hpp
/// \brief Host-side native debug option state.

#include <c10/core/Device.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include <xpool/macros.hpp>

namespace xpool::debug {

/// Typed native trace-observer options installed in host and device state.
struct TraceObserverOptions {
  /// Whether trace collection is enabled.
  bool enable;
  /// Configured ring capacity; zero denotes the all-disabled default state.
  std::size_t record_capacity;

  constexpr bool operator==(const TraceObserverOptions &) const = default;
};

/// Native enablement for the immutable FFN Graph Observer snapshot.
struct GraphObserverOptions {
  /// Whether the observed Graph snapshot may cross the Python readout boundary.
  bool enable;

  constexpr bool operator==(const GraphObserverOptions &) const = default;
};

/// Native enablement and capacity for semantic MoE routing records.
struct FfnRoutingObserverOptions {
  /// Whether routing records are retained.
  bool enable;
  /// Maximum number of process-local records retained without overwrite.
  std::size_t record_capacity;

  constexpr bool operator==(const FfnRoutingObserverOptions &) const = default;
};

/// Process-wide typed native debug options installed during initialization.
struct Options {
  /// Local CUDA IPC transport trace settings.
  TraceObserverOptions transport_observer;
  /// Cross-Agent NVSHMEM Fabric trace settings.
  TraceObserverOptions fabric_observer;
  /// Immutable FFN Graph Observer settings.
  GraphObserverOptions graph_observer;
  /// Semantic MoE routing-record settings.
  FfnRoutingObserverOptions ffn_routing_observer;
  /// Construct process-wide debug options with every feature disabled.
  /// \post Both observer capacities are zero and Graph observation is disabled.
  XPOOL_HOST_DEVICE_FN
  constexpr Options()
      : transport_observer{false, 0U}, fabric_observer{false, 0U}, graph_observer{false}, ffn_routing_observer{false,
                                                                                                               0U} {}

  constexpr bool operator==(const Options &) const = default;
};

static_assert(std::is_trivially_copyable_v<Options>);

/// Link-visible host storage used only by options() and configure().
extern Options options_h;

/// Configure process-wide host debug options and, when present, their device mirror.
/// \throws c10::Error if a device is invalid, CUDA update fails, or a later
/// call differs from the first configuration.
void configure(const Options &debug_options, std::optional<c10::DeviceIndex> cuda_device);

/// Return the host-side native debug options.
/// Before configuration this is the all-disabled default.
#if !defined(__CUDACC__)
inline const Options &options() { return options_h; }
#endif

} // namespace xpool::debug
