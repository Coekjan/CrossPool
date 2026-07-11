#pragma once

/// \file xpool/utils/queue.cuh
/// \brief Device-side bounded ring queue implementation.

#include <cstdint>
#include <type_traits>

#include <xpool/atomic.cuh>
#include <xpool/utils/queue.hpp>

namespace xpool::utils::queue {

/// Device-side view over a bounded ring queue stored in an external arena.
/// \tparam T Trivially copyable value type stored in each queue cell.
template <typename T> class RingQueue {
  static_assert(std::is_trivially_copyable_v<T>);

public:
  /// Construct a queue view from raw queue storage.
  /// \param state Monotonic head/tail counters.
  /// \param cells Queue cell array with capacity cells.
  __device__ RingQueue(RingQueueState *state, RingQueueCell<T> *cells)
      : state_(state), cells_(cells) {}

  /// Return a queue view at byte offsets inside an arena.
  /// \param arena Base address of the externally owned arena.
  /// \param layout Queue state/cell offsets inside arena.
  /// \return Device-side queue view.
  __device__ static RingQueue at(unsigned char *arena,
                                 const RingQueueLayout &layout) {
    return RingQueue{
        reinterpret_cast<RingQueueState *>(arena + layout.state_offset),
        reinterpret_cast<RingQueueCell<T> *>(arena + layout.cells_offset),
    };
  }

  /// Try to push one value into the queue.
  /// \param value Value to publish.
  /// \return True when value was pushed, false when the queue is full.
  __device__ bool try_push(T value) const {
    std::uint64_t tail = xpool::atomic::load_acquire(state_->tail);
    while (true) {
      const std::uint64_t head = xpool::atomic::load_acquire(state_->head);
      const auto capacity = state_->capacity;
      if (tail - head >= capacity) {
        return false;
      }
      auto *cell = &cells_[tail % capacity];
      const std::uint64_t sequence =
          xpool::atomic::load_acquire(cell->sequence);
      const auto difference =
          static_cast<long long>(sequence) - static_cast<long long>(tail);
      if (difference == 0) {
        if (xpool::atomic::compare_exchange_acq_rel(
                state_->tail, tail, tail + static_cast<std::uint64_t>(1)) ==
            tail) {
          cell->value = value;
          xpool::atomic::store_release(cell->sequence,
                                       tail + static_cast<std::uint64_t>(1));
          return true;
        }
        tail = xpool::atomic::load_acquire(state_->tail);
        continue;
      }
      if (difference < 0) {
        return false;
      }
      tail = xpool::atomic::load_acquire(state_->tail);
    }
  }

  /// Try to push one value until it succeeds or reaches a timeout.
  /// \param value Value to publish.
  /// \param timeout_clocks Device clock budget.
  /// \param relax_interval_ns Nanoseconds to sleep between failed attempts.
  /// \return True when value was pushed, false when the timeout expires.
  __device__ bool try_push_with_timeout(
      T value,
      unsigned long long timeout_clocks = kDefaultRingQueueSpinTimeoutClocks,
      unsigned int relax_interval_ns = 1000U) const {
    const unsigned long long start_clock = clock64();
    while (true) {
      if (try_push(value)) {
        return true;
      }
      if (timeout_clocks != 0ULL && clock64() - start_clock > timeout_clocks) {
        return false;
      }
      __nanosleep(relax_interval_ns);
    }
  }

  /// Try to pop one value from the queue.
  /// \param value Output receiving the popped value.
  /// \return True when a value was popped, false when the queue is empty.
  __device__ bool try_pop(T &value) const {
    std::uint64_t head = xpool::atomic::load_acquire(state_->head);
    while (true) {
      const std::uint64_t tail = xpool::atomic::load_acquire(state_->tail);
      if (tail <= head) {
        return false;
      }
      const auto capacity = state_->capacity;
      auto *cell = &cells_[head % capacity];
      const std::uint64_t sequence =
          xpool::atomic::load_acquire(cell->sequence);
      const auto difference = static_cast<long long>(sequence) -
                              static_cast<long long>(head + 1ULL);
      if (difference == 0) {
        if (xpool::atomic::compare_exchange_acq_rel(
                state_->head, head, head + static_cast<std::uint64_t>(1)) ==
            head) {
          value = cell->value;
          xpool::atomic::store_release(
              cell->sequence, head + static_cast<std::uint64_t>(capacity));
          return true;
        }
        head = xpool::atomic::load_acquire(state_->head);
        continue;
      }
      if (difference < 0) {
        return false;
      }
      head = xpool::atomic::load_acquire(state_->head);
    }
  }

  /// Try to pop one value until it succeeds or reaches a timeout.
  /// \param value Output receiving the popped value.
  /// \param timeout_clocks Device clock budget.
  /// \param relax_interval_ns Nanoseconds to sleep between failed attempts.
  /// \return True when a value was popped, false when the timeout expires.
  __device__ bool try_pop_with_timeout(
      T &value,
      unsigned long long timeout_clocks = kDefaultRingQueueSpinTimeoutClocks,
      unsigned int relax_interval_ns = 1000U) const {
    const unsigned long long start_clock = clock64();
    while (true) {
      if (try_pop(value)) {
        return true;
      }
      if (timeout_clocks != 0ULL && clock64() - start_clock > timeout_clocks) {
        return false;
      }
      __nanosleep(relax_interval_ns);
    }
  }

  /// Return the number of queue positions currently owned by producers.
  /// \return Monotonic tail minus head, bounded by queue capacity.
  __device__ std::uint64_t size() const {
    const std::uint64_t head = xpool::atomic::load_acquire(state_->head);
    const std::uint64_t tail = xpool::atomic::load_acquire(state_->tail);
    return tail - head;
  }

  /// Return whether the queue currently contains no positions.
  /// \return True when head equals tail.
  __device__ bool empty() const { return size() == 0ULL; }

  /// Return whether every queue position has been fully published.
  /// \return True when the queue is at capacity and every producer has
  /// release-published its reserved cell.
  __device__ bool full() const {
    const std::uint64_t head = xpool::atomic::load_acquire(state_->head);
    const std::uint64_t tail = xpool::atomic::load_acquire(state_->tail);
    const std::uint64_t capacity = state_->capacity;
    if (tail - head != capacity) {
      return false;
    }
    for (std::uint64_t position = head; position < tail; ++position) {
      const auto *cell = &cells_[position % capacity];
      if (xpool::atomic::load_acquire(cell->sequence) != position + 1ULL) {
        return false;
      }
    }
    return true;
  }

private:
  RingQueueState *state_;
  RingQueueCell<T> *cells_;
};

} // namespace xpool::utils::queue
