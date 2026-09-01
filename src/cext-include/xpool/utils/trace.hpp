#pragma once

/// \file xpool/utils/trace.hpp
/// \brief Common process-local observer counters and event timelines.

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/abort.hpp>
#include <xpool/macros.hpp>

namespace xpool::utils::trace {

/// Monotonic reservation and overflow counters shared by trace buffers.
struct BufferState {
  /// Total number of record slots reserved since installation.
  std::uint64_t sequence;
  /// Number of reservations dropped after capacity was exhausted.
  std::uint64_t dropped;
};

/// One successfully reserved record and its process-local sequence.
template <typename Record> class Entry {
public:
#if defined(__CUDACC__)
  /// Construct an entry from reserved storage.
  /// \param record Writable record slot.
  /// \param sequence Monotonic sequence assigned to the slot.
  XPOOL_DEVICE_FN Entry(Record *record, std::uint64_t sequence) : record_(record), sequence_(sequence) {}

  /// Return the writable reserved record.
  XPOOL_DEVICE_FN Record &record() const { return *record_; }

  /// Return the monotonic process-local reservation sequence.
  XPOOL_DEVICE_FN std::uint64_t sequence() const { return sequence_; }
#endif

private:
  Record *record_ = nullptr;
  std::uint64_t sequence_ = 0;
};

/// Enumeration with a terminal Count value suitable for a Timeline.
template <typename Event>
concept EventType = std::is_enum_v<Event> && requires {
  Event::Count;
};

/// Fixed-size event bitmap and timestamp array for one semantic trace.
template <EventType Event> class Timeline {
public:
  /// Record one event after verifying required predecessors.
  /// \param timestamp Device clock value associated with the event.
  template <Event Current, Event... Required> XPOOL_HOST_DEVICE_FN void record(std::uint64_t timestamp) {
    constexpr auto current = index<Current>();
    xpool::abort_if((recorded_events_ & bit(current)) != 0);
    xpool::abort_if((!recorded(Required) || ...));
    timestamps_[current] = timestamp;
    recorded_events_ |= bit(current);
  }

  /// Test whether an event was recorded.
  XPOOL_HOST_DEVICE_FN bool recorded(Event event) const {
    const auto event_index = static_cast<std::size_t>(event);
    xpool::abort_if(event_index >= event_count);
    return (recorded_events_ & bit(event_index)) != 0;
  }

  /// Return the timestamp slot for an event.
  /// \param event Event whose timestamp is requested.
  /// \return Recorded device clock value, or zero when not recorded.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(Event event) const {
    const auto event_index = static_cast<std::size_t>(event);
    xpool::abort_if(event_index >= event_count);
    return timestamps_[event_index];
  }

private:
  static constexpr auto event_count = static_cast<std::size_t>(Event::Count);
  static_assert(event_count > 0 && event_count <= 64);

  template <Event Value> static consteval std::size_t index() {
    constexpr auto result = static_cast<std::size_t>(Value);
    static_assert(result < event_count);
    return result;
  }

  XPOOL_HOST_DEVICE_FN static constexpr std::uint64_t bit(std::size_t index) { return std::uint64_t{1} << index; }

  std::uint64_t recorded_events_ = 0;
  std::array<std::uint64_t, event_count> timestamps_{};
};

static_assert(std::is_trivially_copyable_v<BufferState>);

} // namespace xpool::utils::trace
