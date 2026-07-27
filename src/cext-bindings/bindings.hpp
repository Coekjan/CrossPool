#pragma once

/// \file bindings.hpp
/// \brief Python binding assembly for the xpool native extension module.

#include <pybind11/pybind11.h>

/// Python bindings for xpool native control and value types.
namespace xpool::bindings {

/// Bind Fabric metadata, trace values, and lifecycle functions.
/// \param module Root `xpool.native` extension module.
void bind_fabric(pybind11::module_ &module);

/// Bind Transport trace values and lifecycle functions.
/// \param module Root `xpool.native` extension module.
void bind_transport(pybind11::module_ &module);

} // namespace xpool::bindings
