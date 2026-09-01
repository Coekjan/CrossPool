#pragma once

/// \file xpool/transport/hooks.hpp
/// \brief Host and host-visible Device extension points owned by Transport.

#include <c10/core/Device.h>

#include <cstdint>
#include <xpool/hooks/registry.hpp>
#include <xpool/transport/arena.hpp>

#if defined(__CUDACC__)
#include <xpool/hooks/registry.cuh>
#endif

namespace xpool::hooks {

/// Transport endpoint role that owns a Host lifecycle event.
enum class TransportEndpointSite { Instance, AtnAgent };

/// Ordered semantic transitions observed by Transport device adapters.
enum class TransportProtocolEventKind : std::uint32_t {
  RequestStagingStarted,
  RequestStagingCompleted,
  RequestPublished,
  RequestObserved,
  ExecutionStarted,
  ExecutionCompleted,
  ResultPublished,
  ResultObserved,
  OutputCopied,
  ResultAcknowledged,
  Closed,
  Count,
};

/// Observes a Transport endpoint after its native resources are open.
struct TransportEndpointOpenPostEvent final : HostObservePoint {
  /// Open endpoint metadata exposed to Host observers.
  struct Context {
    /// CUDA device that owns the endpoint.
    c10::DeviceIndex cuda_device;
    /// Process-local Transport arena.
    xpool::transport::ArenaView arena;
    /// Layout paired with the arena.
    const xpool::transport::ArenaLayout &layout;
    /// Runtime role of the endpoint.
    TransportEndpointSite site;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(TransportEndpointOpenPostEvent);
};

/// Observes a Transport endpoint before its native resources close.
struct TransportEndpointClosePreEvent final : HostObservePoint {
  /// Closing endpoint metadata exposed to Host observers.
  struct Context {
    /// CUDA device that owns the endpoint.
    c10::DeviceIndex cuda_device;
    /// Process-local Transport arena.
    xpool::transport::ArenaView arena;
    /// Layout paired with the arena.
    const xpool::transport::ArenaLayout &layout;
    /// Runtime role of the endpoint.
    TransportEndpointSite site;
  };

  /// Dispatch all registered Host observers.
  XPOOL_HOST_HOOK_POINT(TransportEndpointClosePreEvent);
};

/// Observes Instance-side Transport protocol transitions on the device.
struct TransportInstanceProtocolEvent final : DeviceObservePoint {
  /// Protocol event enumeration accepted by this Point.
  using Kind = TransportProtocolEventKind;
  struct Context;

#if defined(__CUDACC__)
  /// Dispatch all registered Device observers.
  XPOOL_DEVICE_HOOK_POINT(TransportInstanceProtocolEvent);
#endif
};

/// Observes AtnAgent-side Transport protocol transitions on the device.
struct TransportAtnAgentProtocolEvent final : DeviceObservePoint {
  /// Protocol event enumeration accepted by this Point.
  using Kind = TransportProtocolEventKind;
  struct Context;

#if defined(__CUDACC__)
  /// Dispatch all registered Device observers.
  XPOOL_DEVICE_HOOK_POINT(TransportAtnAgentProtocolEvent);
#endif
};

} // namespace xpool::hooks
