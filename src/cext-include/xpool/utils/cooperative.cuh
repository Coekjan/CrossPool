#pragma once

/// \file xpool/utils/cooperative.cuh
/// \brief Cooperative-group work partitioning primitives.

#include <cooperative_groups/memcpy_async.h>
#include <cuda/memory>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/macros.hpp>

namespace xpool::utils::cooperative {

/// Cooperative group that exposes a stable rank and participant count.
template <typename Group>
concept CooperativeGroup = requires(Group group) {
  { group.thread_rank() } -> std::integral;
  { group.size() } -> std::integral;
};

/// Copy bytes across one cooperative group and wait for completion.
/// \tparam Group Cooperative group exposing thread_rank() and size().
/// \param group Participating cooperative group.
/// \param destination Output storage containing at least count bytes.
/// \param source Input storage containing at least count bytes.
/// \param count Number of bytes copied by the group.
/// \pre Source and destination ranges do not overlap.
/// \pre Null pointers are valid only when count is zero.
template <CooperativeGroup Group>
XPOOL_DEVICE_FN void copy(Group group, void *destination, const void *source, std::size_t count) {
  if (count == 0) {
    return;
  }

  constexpr auto alignment = std::size_t{16};
  const auto destination_address = reinterpret_cast<std::uintptr_t>(destination);
  const auto source_address = reinterpret_cast<std::uintptr_t>(source);
  if (destination_address % alignment == 0 && source_address % alignment == 0 && count % alignment == 0) {
    cooperative_groups::memcpy_async(group, destination, source, cuda::aligned_size_t<alignment>{count});
  } else {
    cooperative_groups::memcpy_async(group, destination, source, count);
  }
  cooperative_groups::wait(group);
}

/// Fill trivially-copyable elements across one cooperative group.
/// \tparam Group Cooperative group exposing thread_rank() and size().
/// \tparam T Trivially-copyable element type.
/// \param group Participating cooperative group.
/// \param destination Output storage containing at least count elements.
/// \param value Value assigned to every output element.
/// \param count Number of elements filled by the group.
/// \post Every participating thread observes the completed fill before return.
template <CooperativeGroup Group, typename T>
XPOOL_DEVICE_FN void fill(Group group, T *destination, const T &value, std::size_t count) {
  static_assert(std::is_trivially_copyable_v<T>);
  for (auto index = static_cast<std::size_t>(group.thread_rank()); index < count;
       index += static_cast<std::size_t>(group.size())) {
    destination[index] = value;
  }
  group.sync();
}

} // namespace xpool::utils::cooperative
