#pragma once

/// \file xpool/devkit/adapters.cuh
/// \brief Compile-visible composition of built-in Device hook adapters.

#include <xpool/fabric/hooks.cuh>
#include <xpool/hooks/registry.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/hooks.cuh>

/// Declare or define a Device hook adapter function.
#define XPOOL_DEVICE_HOOK_FN(point, function, context)                                                                 \
  XPOOL_DEVICE_FN point::Result function(point::Context &context [[maybe_unused]])

namespace xpool::devkit::fabric_observer {

/// Device adapter that records Fabric protocol transitions.
struct DeviceAdapter {
  /// Record an AtnAgent-side transition.
  static XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricAtnAgentProtocolEvent, observe, context);
  /// Record a Coordinator-side transition.
  static XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricCoordinatorProtocolEvent, observe, context);
  /// Record an FfnAgent-side transition.
  static XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, observe, context);
};

} // namespace xpool::devkit::fabric_observer

namespace xpool::devkit::ffn_routing_observer {

/// Device adapter that records published semantic MoE routing metadata.
struct DeviceAdapter {
  /// Copy one Router Owner publication into process-local observer storage.
  static XPOOL_DEVICE_HOOK_FN(xpool::hooks::FabricFfnAgentProtocolEvent, observe, context);
};

} // namespace xpool::devkit::ffn_routing_observer

namespace xpool::devkit::transport_observer {

/// Device adapter that records Transport protocol transitions.
struct DeviceAdapter {
  /// Record an Instance-side transition.
  static XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportInstanceProtocolEvent, observe, context);
  /// Record an AtnAgent-side transition.
  static XPOOL_DEVICE_HOOK_FN(xpool::hooks::TransportAtnAgentProtocolEvent, observe, context);
};

} // namespace xpool::devkit::transport_observer

namespace xpool::devkit {

/// Compile-time catalog of built-in Device devkit adapters.
using DeviceAdapters = xpool::hooks::DeviceAdapterRegistry<xpool::devkit::transport_observer::DeviceAdapter,
                                                           xpool::devkit::fabric_observer::DeviceAdapter,
                                                           xpool::devkit::ffn_routing_observer::DeviceAdapter>;

} // namespace xpool::devkit
