#pragma once

/// \file xpool/hooks.cuh
/// \brief Typed Device extension points and built-in adapter composition.

#include <xpool/devkit/adapters.cuh>
#include <xpool/hooks/registry.cuh>

namespace xpool::hooks {

/// Dispatch a Device hook Point through the built-in adapter catalog.
/// \param context Point-specific Context passed by the production call site.
/// \return The Point result.
template <typename Point> XPOOL_DEVICE_FN typename Point::Result dispatch_device(typename Point::Context &context) {
  return xpool::devkit::DeviceAdapters::template hooks<Point>(context);
}

} // namespace xpool::hooks
