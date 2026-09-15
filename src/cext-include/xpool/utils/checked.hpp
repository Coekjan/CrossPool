#pragma once

/// \file xpool/utils/checked.hpp
/// \brief Host-side checked integer arithmetic utilities.

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <c10/util/Exception.h>
#include <c10/util/safe_numerics.h>

/// Checked host-side integer arithmetic.
namespace xpool::utils::checked {

/// Non-boolean integral type accepted by checked arithmetic.
template <typename T>
concept CheckedInteger = std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>;

/// Same-type integral operands accepted by one checked operation.
template <typename T, typename... U>
concept SameCheckedIntegerOperands =
    CheckedInteger<T> && (std::same_as<std::remove_cvref_t<T>, std::remove_cvref_t<U>> && ...);

/// Non-boolean unsigned integral type accepted by checked alignment.
template <typename T>
concept CheckedUnsignedInteger = CheckedInteger<T> && std::unsigned_integral<std::remove_cvref_t<T>>;

/// Add one or more same-type integer operands without overflow.
/// \throws c10::Error if the sum overflows T.
template <typename T, typename... U>
  requires SameCheckedIntegerOperands<T, U...>
std::remove_cvref_t<T> sum(T first, U... values) {
  using Result = std::remove_cvref_t<T>;
  auto result = Result{0};
  auto overflow = c10::add_overflows(result, first, &result);
  overflow = overflow || (c10::add_overflows(result, values, &result) || ... || false);
  TORCH_CHECK(!overflow, "xpool checked integer sum overflows result type");
  return result;
}

/// Multiply one or more same-type integer operands without overflow.
/// \throws c10::Error if the product overflows T.
template <typename T, typename... U>
  requires SameCheckedIntegerOperands<T, U...>
std::remove_cvref_t<T> prod(T first, U... values) {
  using Result = std::remove_cvref_t<T>;
  if (first == 0 || ((values == 0) || ... || false)) {
    return Result{0};
  }
  auto result = Result{1};
  auto overflow = c10::mul_overflows(result, first, &result);
  overflow = overflow || (c10::mul_overflows(result, values, &result) || ... || false);
  TORCH_CHECK(!overflow, "xpool checked integer product overflows result type");
  return result;
}

/// Round an unsigned integer upward to a positive alignment of the same type.
/// \throws c10::Error if alignment is zero or the calculation overflows.
template <CheckedUnsignedInteger T> std::remove_cvref_t<T> align_up(T value, T alignment) {
  TORCH_CHECK(alignment != 0, "xpool checked alignment must be positive");
  const auto rounded = sum(value, alignment - T{1});
  return (rounded / alignment) * alignment;
}

} // namespace xpool::utils::checked
