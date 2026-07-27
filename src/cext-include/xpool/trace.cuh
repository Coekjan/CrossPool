#pragma once

/// \file xpool/trace.cuh
/// \brief Device allocation over an arena-resident trace buffer.

#include <cuda/atomic>

#include <cstddef>
#include <cstdint>

#include <xpool/atomic.cuh>
#include <xpool/macros.hpp>
#include <xpool/trace.hpp>

namespace xpool::trace {

/// Device allocator for one fixed-capacity arena trace buffer.
/// \tparam Record Fixed-size trivially-copyable trace record.
template <typename Record> class Buffer {
public:
  static_assert(std::is_trivially_copyable_v<Record>);

  /// Bind a trace buffer to arena storage and metadata.
  /// \param base Byte-addressable arena base.
  /// \param layout Immutable trace region geometry.
  /// \param state Mutable trace allocation counters.
  XPOOL_DEVICE_FN Buffer(std::uint8_t *base, const BufferLayout &layout, BufferState &state)
      : base_(base), layout_(layout), state_(state) {}

  /// Reserve one record without overwriting existing records.
  /// \return Available entry, or an empty entry when tracing is disabled or
  /// capacity has been exhausted.
  XPOOL_DEVICE_FN Entry<Record> reserve() const {
    if (layout_.capacity == 0) {
      return {};
    }
    const auto sequence =
        xpool::atomic::system_atomic(state_.sequence).fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed) + 1;
    if (sequence > layout_.capacity) {
      xpool::atomic::system_atomic(state_.dropped).fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed);
      return {};
    }
    auto *records = reinterpret_cast<Record *>(base_ + layout_.records_offset);
    auto *record = &records[static_cast<std::size_t>(sequence - 1)];
    *record = Record{};
    return Entry<Record>{record, sequence};
  }

  /// Find one retained record by monotonic identity.
  /// \param sequence Identity previously returned by reserve().
  /// \return Retained record, or null for zero, disabled, or dropped ids.
  XPOOL_DEVICE_FN Record *find(std::uint64_t sequence) const {
    if (sequence == 0 || sequence > layout_.capacity) {
      return nullptr;
    }
    auto *records = reinterpret_cast<Record *>(base_ + layout_.records_offset);
    return &records[static_cast<std::size_t>(sequence - 1)];
  }

private:
  std::uint8_t *base_;
  const BufferLayout &layout_;
  BufferState &state_;
};

} // namespace xpool::trace
