#pragma once

/// \file xpool/abi.hpp
/// \brief Compile-time identity of the native xpool ABI.

#include <cstdint>

namespace xpool::abi {

/// Native ABI version checked when the extension is loaded and in wire data.
inline constexpr std::uint32_t kVersion = 79;

} // namespace xpool::abi
