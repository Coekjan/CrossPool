#pragma once

/// \file xpool/utils/ranges.hpp
/// \brief Host-side range materialization utilities.

#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

/// Generic host-side range utilities.
namespace xpool::utils::ranges {

/// Materialize an input range as an ordered vector.
/// \tparam R Input range whose value type is movable.
/// \param values Range to consume in iteration order.
/// \return Vector containing the range values in the same order.
template <std::ranges::input_range R>
  requires std::movable<std::ranges::range_value_t<R>>
auto to_vector(R &&values) {
  using Value = std::ranges::range_value_t<R>;
  std::vector<Value> result;
  if constexpr (std::ranges::sized_range<R>) {
    result.reserve(std::ranges::size(values));
  }
  for (auto &&value : values) {
    result.emplace_back(std::forward<decltype(value)>(value));
  }
  return result;
}

} // namespace xpool::utils::ranges
