#pragma once

/// \file xpool/trace.hpp
/// \brief Common fixed-record trace arena geometry and event timelines.

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/abort.hpp>
#include <xpool/macros.hpp>

namespace xpool::trace {

/// Byte geometry of one arena-resident fixed-record trace buffer.
struct BufferLayout {
  /// Byte offset of the first record, or zero when tracing is disabled.
  std::size_t records_offset;
  /// Maximum number of records retained by this arena.
  std::size_t capacity;
};

/// Mutable counters for one arena-resident trace buffer.
struct BufferState {
  /// Number of trace identities requested since arena creation.
  std::uint64_t sequence;
  /// Number of new traces dropped after capacity was exhausted.
  std::uint64_t dropped;
};

/// One successfully reserved trace-buffer entry.
/// \tparam Record Fixed-size trivially-copyable trace record.
template <typename Record> class Entry {
public:
#if defined(__CUDACC__)
  /// Construct an unavailable entry.
  XPOOL_DEVICE_FN Entry() = default;

  /// Construct one reserved entry.
  /// \param record Reserved record storage.
  /// \param sequence Monotonic trace identity assigned to the record.
  XPOOL_DEVICE_FN Entry(Record *record, std::uint64_t sequence)
      : record_(record), sequence_(sequence) {}

  /// Return whether reservation succeeded.
  /// \return True when this entry owns a reserved record.
  XPOOL_DEVICE_FN explicit operator bool() const { return record_ != nullptr; }

  /// Return the reserved record.
  /// \return Mutable arena record.
  /// \pre This entry is available.
  XPOOL_DEVICE_FN Record &record() const {
    xpool::abort_if(record_ == nullptr);
    return *record_;
  }

  /// Return the monotonic identity assigned during reservation.
  /// \return Positive identity, or zero for an unavailable entry.
  XPOOL_DEVICE_FN std::uint64_t sequence() const { return sequence_; }
#endif

private:
  Record *record_ = nullptr;
  std::uint64_t sequence_ = 0;
};

/// Closed enum suitable for indexing one fixed trace timeline.
template <typename Event>
concept EventType = std::is_enum_v<Event> && requires { Event::Count; };

/// Fixed timestamps and completion bits for one event dependency graph.
/// \tparam Event Closed event enum ending in Count.
template <EventType Event> class Timeline {
public:
  /// Record one event after validating its compile-time prerequisites.
  /// \tparam Current Event being completed.
  /// \tparam Required Events that must already be complete.
  /// \param timestamp Device-global timestamp in nanoseconds.
  /// \pre Current has not been recorded and every Required event has.
  template <Event Current, Event... Required> XPOOL_HOST_DEVICE_FN void record(std::uint64_t timestamp) {
    constexpr auto current = index<Current>();
    xpool::abort_if((recorded_events_ & bit(current)) != 0);
    xpool::abort_if((!recorded(Required) || ...));
    timestamps_[current] = timestamp;
    recorded_events_ |= bit(current);
  }

  /// Return whether an event has been recorded.
  /// \param event Event to inspect.
  /// \return True when the event timestamp was committed.
  XPOOL_HOST_DEVICE_FN bool recorded(Event event) const {
    const auto event_index = static_cast<std::size_t>(event);
    xpool::abort_if(event_index >= event_count);
    return (recorded_events_ & bit(event_index)) != 0;
  }

  /// Return one event timestamp.
  /// \param event Event to inspect.
  /// \return Timestamp in nanoseconds, or zero when not recorded.
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

static_assert(std::is_trivially_copyable_v<BufferLayout>);
static_assert(std::is_trivially_copyable_v<BufferState>);

} // namespace xpool::trace
