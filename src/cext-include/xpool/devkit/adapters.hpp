#pragma once

/// \file xpool/devkit/adapters.hpp
/// \brief Compile-visible composition of built-in Host hook adapters.

#include <xpool/fabric/hooks.hpp>
#include <xpool/ffnagent/hooks.hpp>
#include <xpool/hooks/registry.hpp>
#include <xpool/transport/hooks.hpp>

/// Declare or define a Host hook adapter function.
#define XPOOL_HOST_HOOK_FN(point, function, context) point::Result function(point::Context &context [[maybe_unused]])

namespace xpool::devkit::fabric_observer {

/// Host lifecycle adapter for process-local Fabric trace storage.
struct HostAdapter {
  /// Install trace storage after Fabric join.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FabricJoinPostEvent, observe, context);
  /// Release trace storage before Fabric finalization.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FabricFinalizePreEvent, observe, context);
};

} // namespace xpool::devkit::fabric_observer

namespace xpool::devkit::ffn_routing_observer {

/// Host lifecycle adapter for FFN routing observation.
struct HostAdapter {
  /// Allocate process-local routing trace storage before installation.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionInstallPreEvent, observe, context);
  /// Release routing trace storage after finalization.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionFinalizePostEvent, observe, context);
};

} // namespace xpool::devkit::ffn_routing_observer

namespace xpool::devkit::graph_observer {

/// Host lifecycle adapter that inventories installed CUDA Graphs.
struct HostAdapter {
  /// Reset graph evidence before installation.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionInstallPreEvent, observe, context);
  /// Record one parameterized computation graph.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FfnPrimaryGraphParameterizePostEvent, observe, context);
  /// Record one completed executor-lane graph.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FfnLaneGraphBuildPostEvent, observe, context);
  /// Release process-local graph evidence after finalization.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionFinalizePostEvent, observe, context);
};

} // namespace xpool::devkit::graph_observer

namespace xpool::devkit::transport_observer {

/// Host lifecycle adapter for process-local Transport trace storage.
struct HostAdapter {
  /// Install trace storage after endpoint open.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointOpenPostEvent, observe, context);
  /// Release trace storage before endpoint close.
  static XPOOL_HOST_HOOK_FN(xpool::hooks::TransportEndpointClosePreEvent, observe, context);
};

} // namespace xpool::devkit::transport_observer

namespace xpool::devkit {

/// Compile-time catalog of built-in Host devkit adapters.
using HostAdapters = xpool::hooks::HostAdapterRegistry<
    xpool::devkit::fabric_observer::HostAdapter, xpool::devkit::ffn_routing_observer::HostAdapter,
    xpool::devkit::graph_observer::HostAdapter, xpool::devkit::transport_observer::HostAdapter>;

} // namespace xpool::devkit
