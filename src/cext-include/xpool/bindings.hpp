#pragma once

/// \file bindings.hpp
/// \brief Shared Python conversion and binding assembly for CrossPool.

#include <cstddef>
#include <string_view>
#include <vector>

#include <c10/core/ScalarType.h>
#include <pybind11/pybind11.h>
#include <pybind11/typing.h>

/// Python bindings for CrossPool native control and value types.
namespace xpool::bindings {

/// Copy values into a fresh typed Python tuple while the caller holds the GIL.
/// \tparam T Registered Python element representation.
/// \param values Elements copied into the result in their original order.
/// \return A Python-owned tuple preserving its element type for generated stubs.
template <typename T> pybind11::typing::Tuple<T, pybind11::ellipsis> typed_tuple(const std::vector<T> &values) {
  auto result = pybind11::typing::Tuple<T, pybind11::ellipsis>(values.size());
  for (auto index = std::size_t{0}; index < values.size(); ++index) {
    result[index] = pybind11::cast(values[index]);
  }
  return result;
}

/// Parse a Torch dtype and require supported FFN payload precision.
/// \param dtype Python dtype object interpreted by Torch's parser.
/// \param context Existing caller name used in the validation error.
/// \return FP16 or BF16 ScalarType; invalid input raises the original parsing or validation error.
c10::ScalarType require_payload_dtype(const pybind11::object &dtype, std::string_view context);

/// Bind native debug-option value types.
/// \param module Root `xpool.native` extension module.
void bind_debug(pybind11::module_ &module);

/// Bind development-only observer value types and read operations.
/// \param module Root `xpool.native` extension module.
void bind_devkit(pybind11::module_ &module);

/// Bind Fabric metadata, trace values, and lifecycle functions.
/// \param module Root `xpool.native` extension module.
void bind_fabric(pybind11::module_ &module);

/// Bind host-local Elastic Context State byte-quota channel types.
/// \param module Root `xpool.native` extension module.
void bind_ecs(pybind11::module_ &module);

/// Bind Transport trace values and lifecycle functions.
/// \param module Root `xpool.native` extension module.
void bind_transport(pybind11::module_ &module);

} // namespace xpool::bindings
