#pragma once

/// \file xpool/utils/cooperative.cuh
/// \brief Cooperative-group work partitioning primitives.

#include <cooperative_groups/memcpy_async.h>
#include <cuda/memory>
#include <cuda/std/span>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/abort.hpp>
#include <xpool/macros.hpp>

namespace xpool::utils::cooperative {

/// Cooperative group that exposes a stable rank and participant count.
template <typename Group>
concept CooperativeGroup = requires(Group group) {
  { group.thread_rank() } -> std::integral;
  { group.size() } -> std::integral;
};

/// Copy bytes across one cooperative group and wait for completion.
/// \pre Source and destination ranges do not overlap.
/// \post Every participant observes the completed copy before return.
template <CooperativeGroup Group>
XPOOL_DEVICE_FN void copy(Group group, cuda::std::span<std::uint8_t> destination,
                          cuda::std::span<const std::uint8_t> source) {
  xpool::abort_if(destination.size() != source.size());
  if (destination.empty()) {
    return;
  }

  constexpr auto alignment = std::size_t{16};
  const auto destination_address = reinterpret_cast<std::uintptr_t>(destination.data());
  const auto source_address = reinterpret_cast<std::uintptr_t>(source.data());
  if (destination_address % alignment == 0 && source_address % alignment == 0 && destination.size() % alignment == 0) {
    cooperative_groups::memcpy_async(group, destination.data(), source.data(),
                                     cuda::aligned_size_t<alignment>{destination.size()});
  } else {
    cooperative_groups::memcpy_async(group, destination.data(), source.data(), destination.size());
  }
  cooperative_groups::wait(group);
}

/// Fill trivially-copyable elements across one cooperative group.
/// \post Every participating thread observes the completed fill before return.
template <CooperativeGroup Group, typename T>
XPOOL_DEVICE_FN void fill(Group group, cuda::std::span<T> destination, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  for (auto index = static_cast<std::size_t>(group.thread_rank()); index < destination.size();
       index += static_cast<std::size_t>(group.size())) {
    destination[index] = value;
  }
  group.sync();
}

/// Apply one unary operation across equal-size ranges using one cooperative group.
/// \post Every participating thread observes the completed transform before return.
template <CooperativeGroup Group, typename Output, typename Input, typename Operation>
XPOOL_DEVICE_FN void transform(Group group, cuda::std::span<Output> destination, cuda::std::span<const Input> source,
                               Operation operation) {
  xpool::abort_if(destination.size() != source.size());
  for (auto index = static_cast<std::size_t>(group.thread_rank()); index < destination.size();
       index += static_cast<std::size_t>(group.size())) {
    destination[index] = operation(source[index]);
  }
  group.sync();
}

} // namespace xpool::utils::cooperative
