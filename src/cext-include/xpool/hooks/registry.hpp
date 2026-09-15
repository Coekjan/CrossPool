#pragma once

/// \file xpool/hooks/registry.hpp
/// \brief Typed Host and Device hook points and compile-time Host dispatch.

#include <concepts>

namespace xpool::hooks {

/// Marker for a Host observation point.
struct HostObservePoint {
  /// Observation points do not alter production control flow.
  using Result = void;
};

/// Marker for a Device observation point.
struct DeviceObservePoint {
  /// Device observation points do not alter production control flow.
  using Result = void;
};

/// Host hook point whose adapters may only observe its Context.
template <typename Point>
concept HostObserverPoint = std::derived_from<Point, HostObservePoint>;

/// Device hook point whose adapters may only observe its Context.
template <typename Point>
concept DeviceObserverPoint = std::derived_from<Point, DeviceObservePoint>;

/// Adapter that defines the observation operation for a Point.
template <typename Adapter, typename Point>
concept AdapterObserver = requires(typename Point::Context &context) {
  { Adapter::observe(context) } -> std::same_as<typename Point::Result>;
};

template <typename Point> typename Point::Result dispatch_host(typename Point::Context &context);

/// Define the uniform Host dispatch entry points on a hook Point.
#define XPOOL_HOST_HOOK_POINT(point)                                                                                   \
  static Result hooks(Context &context) { return xpool::hooks::dispatch_host<point>(context); }                        \
  static Result hooks(Context &&context) { return hooks(context); }

/// Compile-time Host adapter catalog.
template <typename... Adapters> class HostAdapterRegistry {
public:
  /// Dispatch one observation Context to every matching adapter.
  /// \param context Context observed by registered adapters.
  template <HostObserverPoint Point> static void hooks(typename Point::Context &context) {
    (observe_one<Adapters, Point>(context), ...);
  }

private:
  template <typename Adapter, typename Point> static void observe_one(typename Point::Context &context) {
    if constexpr (AdapterObserver<Adapter, Point>) {
      Adapter::observe(context);
    }
  }
};

} // namespace xpool::hooks
