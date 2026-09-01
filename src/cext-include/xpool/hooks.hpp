#pragma once

/// \file xpool/hooks.hpp
/// \brief Typed Host hook dispatch through the built-in adapter catalog.

#include <xpool/devkit/adapters.hpp>
#include <xpool/hooks/registry.hpp>

namespace xpool::hooks {

/// Dispatch a Host hook Point through the built-in adapter catalog.
/// \param context Point-specific Context passed by the production call site.
/// \return The Point result.
template <typename Point> typename Point::Result dispatch_host(typename Point::Context &context) {
  return xpool::devkit::HostAdapters::template hooks<Point>(context);
}

} // namespace xpool::hooks
