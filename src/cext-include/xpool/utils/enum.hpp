#pragma once

/// \file xpool/utils/enum.hpp
/// \brief Input constraint shared by validated native enum value types.

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <xpool/macros.hpp>

namespace xpool::utils {

/// Value accepted by one validated enum wrapper.
///
/// The input must be the wrapper's own enum type or a non-boolean integral
/// value. Unrelated enums are rejected even when they share a representation.
/// \tparam T Candidate input type.
/// \tparam Enum Wrapper-owned enum type with an unsigned representation.
template <typename T, typename Enum>
concept EnumInput = std::is_enum_v<Enum> && std::unsigned_integral<std::underlying_type_t<Enum>> &&
                    (std::same_as<std::remove_cvref_t<T>, Enum> ||
                     (std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>));

/// Compare an accepted input with one same-domain enum value without narrowing.
/// \tparam Enum Wrapper-owned enum type with an unsigned representation.
/// \tparam T Candidate input type accepted by EnumInput.
/// \param value Raw or same-domain value to inspect.
/// \param expected Declared enum value to match.
/// \return True only when both values have the same non-negative magnitude.
template <typename Enum, EnumInput<Enum> T>
XPOOL_HOST_DEVICE_FN constexpr bool enum_value_equal(T value, Enum expected) {
  if constexpr (std::same_as<std::remove_cvref_t<T>, Enum>) {
    return value == expected;
  } else {
    if constexpr (std::signed_integral<std::remove_cvref_t<T>>) {
      if (value < 0) {
        return false;
      }
    }
    return static_cast<std::uintmax_t>(value) == static_cast<std::uintmax_t>(expected);
  }
}

} // namespace xpool::utils
