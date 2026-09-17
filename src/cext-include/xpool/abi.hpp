#pragma once

/// \file xpool/abi.hpp
/// \brief Compile-time identity of the native CrossPool ABI.

#include <cstdint>

namespace xpool::abi {

/// Native ABI version checked when the extension is loaded and in wire data.
inline constexpr std::uint32_t kVersion = 84;

} // namespace xpool::abi
