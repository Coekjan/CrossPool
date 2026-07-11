#pragma once

/// \file xpool/utils/queue.hpp
/// \brief Host-visible bounded ring queue layout and initialization helpers.

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace xpool::utils::queue {

/// Default device clock budget for bounded spin-wait queue operations.
inline constexpr unsigned long long kDefaultRingQueueSpinTimeoutClocks =
    5ULL * 1000ULL * 1000ULL * 1000ULL;

/// Monotonic head/tail counters and immutable capacity for one bounded ring
/// queue.
struct alignas(8) RingQueueState {
  /// Number of cells allocated for this queue. Written once by the arena owner
  /// during arena initialization and read by device producers/consumers.
  std::uint64_t capacity;
  /// Next queue position to pop. Stored in shared memory owned by the queue
  /// producer/consumer protocol.
  std::uint64_t head;
  /// Next queue position to push. Stored in shared memory owned by the queue
  /// producer/consumer protocol.
  std::uint64_t tail;
};

/// One bounded ring queue cell carrying a value.
/// \tparam T Trivially copyable value type stored in the cell.
template <typename T> struct alignas(8) RingQueueCell {
  /// Queue-generation sequence used by the bounded MPMC protocol to
  /// distinguish empty, full, and wrapped cells.
  std::uint64_t sequence;
  /// Queue payload value.
  T value;
};

/// Byte layout of one ring queue inside an externally owned arena.
struct RingQueueLayout {
  /// Byte offset of the queue capacity/head/tail state inside the arena.
  std::int64_t state_offset;
  /// Byte offset of the queue cell array inside the arena.
  std::int64_t cells_offset;
  /// Return whether two queue layouts describe the same byte geometry.
  /// \return True when queue state and cell offsets match exactly.
  bool operator==(const RingQueueLayout &) const = default;
};

static_assert(std::is_standard_layout_v<RingQueueState>);
static_assert(std::is_standard_layout_v<RingQueueLayout>);

/// Host-side image for initializing a device-resident ring queue.
/// \tparam T Trivially copyable value type stored in each queue cell.
template <typename T> struct RingQueueHostImage {
  static_assert(std::is_trivially_copyable_v<T>);

  /// Initial queue capacity/head/tail state.
  RingQueueState state;
  /// Initial queue cells.
  std::vector<RingQueueCell<T>> cells;

  /// Build an image for a queue that initially contains every value.
  /// \param values Values to make immediately available for popping.
  /// \return Full queue image with tail set to values.size().
  static RingQueueHostImage full(std::span<const T> values) {
    RingQueueHostImage image{
        RingQueueState{
            static_cast<std::uint64_t>(values.size()),
            0ULL,
            static_cast<std::uint64_t>(values.size()),
        },
        std::vector<RingQueueCell<T>>(values.size()),
    };
    for (std::size_t index = 0; index < values.size(); ++index) {
      image.cells[index] = RingQueueCell<T>{
          static_cast<std::uint64_t>(index + 1), values[index]};
    }
    return image;
  }

  /// Build an image for a queue that initially contains no values.
  /// \param capacity Number of cells allocated for the queue.
  /// \return Empty queue image with wrapped-cell sequence counters initialized.
  static RingQueueHostImage empty(std::size_t capacity) {
    RingQueueHostImage image{
        RingQueueState{
            static_cast<std::uint64_t>(capacity),
            0ULL,
            0ULL,
        },
        std::vector<RingQueueCell<T>>(capacity),
    };
    for (std::size_t index = 0; index < capacity; ++index) {
      image.cells[index] =
          RingQueueCell<T>{static_cast<std::uint64_t>(index), T{}};
    }
    return image;
  }
};

} // namespace xpool::utils::queue
