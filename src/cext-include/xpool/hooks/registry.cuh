#pragma once

/// \file xpool/hooks/registry.cuh
/// \brief Compile-time dispatch over a fixed Device adapter catalog.

#include <cstddef>

#include <xpool/hooks/registry.hpp>
#include <xpool/macros.hpp>

namespace xpool::hooks {

template <typename Point> XPOOL_DEVICE_FN typename Point::Result dispatch_device(typename Point::Context &context);

/// Define the uniform Device dispatch entry points on a hook Point.
#define XPOOL_DEVICE_HOOK_POINT(point)                                                                                 \
  XPOOL_DEVICE_FN static Result hooks(Context &context) { return xpool::hooks::dispatch_device<point>(context); }      \
  XPOOL_DEVICE_FN static Result hooks(Context &&context) { return hooks(context); }

/// Compile-time Device adapter catalog.
template <typename... Adapters> class DeviceAdapterRegistry {
public:
  /// Dispatch one observation Context to every matching adapter.
  /// \param context Context observed by registered adapters.
  template <DeviceObserverPoint Point> XPOOL_DEVICE_FN static void hooks(typename Point::Context &context) {
    (observe_one<Adapters, Point>(context), ...);
  }

private:
  template <typename Adapter, typename Point>
  XPOOL_DEVICE_FN static void observe_one(typename Point::Context &context) {
    if constexpr (AdapterObserver<Adapter, Point>) {
      Adapter::observe(context);
    }
  }
};

} // namespace xpool::hooks
