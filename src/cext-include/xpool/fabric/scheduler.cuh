#pragma once

/// \file xpool/fabric/scheduler.cuh
/// \brief Device implementation of Coordinator-private scheduling.

#include <cuda/std/optional>
#include <cuda/std/variant>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <xpool/abort.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/macros.hpp>

namespace xpool::fabric {

XPOOL_DEVICE_FN inline bool Scheduler::has_unresolved_invocation(std::size_t instance_index) const {
  xpool::abort_if(entries_ == nullptr || instance_index >= instance_count_);
  return entries_[instance_index].state != SchedulerEntry::State::Idle;
}

XPOOL_DEVICE_FN inline std::uint64_t Scheduler::enqueue(const Invocation &invocation) {
  xpool::abort_if(entries_ == nullptr || invocation.validate() != xpool::ffn::ResultCode::Ok ||
                  invocation.key.instance_index >= instance_count_);
  auto &entry = entries_[invocation.key.instance_index];
  xpool::abort_if(entry.state != SchedulerEntry::State::Idle ||
                  entry.completed_sequence == std::numeric_limits<std::uint64_t>::max() ||
                  invocation.key.invocation_sequence != entry.completed_sequence + 1);
  entry.invocation = invocation;
  entry.executor_lane_index = 0;
  if (auto *fifo = cuda::std::get_if<FifoState>(&state_)) {
    ++fifo->next_ticket;
    xpool::abort_if(fifo->next_ticket == 0);
    entry.ready_ticket = fifo->next_ticket;
  } else {
    entry.ready_ticket = 0;
  }
  entry.state = SchedulerEntry::State::Ready;
  return entry.ready_ticket;
}

XPOOL_DEVICE_FN inline std::size_t Scheduler::find_idle_lane() const {
  // Lane occupancy is derived from active Instance entries; there is no second
  // occupancy table that could drift from the scheduler state machine.
  for (auto lane = std::size_t{0}; lane < executor_lane_count_; ++lane) {
    auto active = false;
    for (auto instance_index = std::size_t{0}; instance_index < instance_count_; ++instance_index) {
      const auto &entry = entries_[instance_index];
      if (entry.state == SchedulerEntry::State::Active && entry.executor_lane_index == lane) {
        active = true;
        break;
      }
    }
    if (!active) {
      return lane;
    }
  }
  return executor_lane_count_;
}

XPOOL_DEVICE_FN inline Scheduler::Decision Scheduler::admit(std::size_t instance_index,
                                                                  std::size_t executor_lane_index) {
  auto &entry = entries_[instance_index];
  xpool::abort_if(entry.state != SchedulerEntry::State::Ready || executor_lane_index >= executor_lane_count_);
  entry.executor_lane_index = executor_lane_index;
  entry.state = SchedulerEntry::State::Active;
  return Decision{entry.invocation, executor_lane_index};
}

XPOOL_DEVICE_FN inline cuda::std::optional<Scheduler::Decision> Scheduler::try_schedule() {
  xpool::abort_if(entries_ == nullptr || instance_count_ == 0 || executor_lane_count_ == 0);
  const auto lane = find_idle_lane();
  if (lane == executor_lane_count_) {
    return cuda::std::nullopt;
  }

  auto has_ready = false;
  for (auto instance_index = std::size_t{0}; instance_index < instance_count_; ++instance_index) {
    if (entries_[instance_index].state == SchedulerEntry::State::Ready) {
      has_ready = true;
      break;
    }
  }
  if (!has_ready) {
    return cuda::std::nullopt;
  }

  auto selected = instance_count_;
  if (cuda::std::holds_alternative<FifoState>(state_)) {
    auto selected_ticket = std::numeric_limits<std::uint64_t>::max();
    for (auto instance_index = std::size_t{0}; instance_index < instance_count_; ++instance_index) {
      const auto &entry = entries_[instance_index];
      if (entry.state == SchedulerEntry::State::Ready && entry.ready_ticket < selected_ticket) {
        selected = instance_index;
        selected_ticket = entry.ready_ticket;
      }
    }
  } else {
    auto &generator = cuda::std::get<RandomState>(state_).generator;
    auto selected_score = std::numeric_limits<std::uint64_t>::max();
    // Draw once per ready Instance and choose the minimum score so iteration
    // order does not become the random policy's tie-breaking mechanism.
    for (auto instance_index = std::size_t{0}; instance_index < instance_count_; ++instance_index) {
      if (entries_[instance_index].state != SchedulerEntry::State::Ready) {
        continue;
      }
      const auto score = generator.next();
      if (selected == instance_count_ || score < selected_score) {
        selected = instance_index;
        selected_score = score;
      }
    }
  }
  xpool::abort_if(selected == instance_count_);
  return admit(selected, lane);
}

XPOOL_DEVICE_FN inline cuda::std::optional<Scheduler::Decision>
Scheduler::active_decision(std::size_t instance_index) const {
  xpool::abort_if(entries_ == nullptr || instance_index >= instance_count_);
  const auto &entry = entries_[instance_index];
  if (entry.state != SchedulerEntry::State::Active) {
    return cuda::std::nullopt;
  }
  return Decision{entry.invocation, entry.executor_lane_index};
}

XPOOL_DEVICE_FN inline void Scheduler::release(const InvocationKey &key,
                                                  std::size_t executor_lane_index) {
  xpool::abort_if(entries_ == nullptr || !key.valid() || key.instance_index >= instance_count_ ||
                  executor_lane_index >= executor_lane_count_);
  auto &entry = entries_[key.instance_index];
  xpool::abort_if(entry.state != SchedulerEntry::State::Active || entry.invocation.key != key ||
                  entry.executor_lane_index != executor_lane_index);
  entry.completed_sequence = key.invocation_sequence;
  entry.invocation = {};
  entry.ready_ticket = 0;
  entry.executor_lane_index = 0;
  entry.state = SchedulerEntry::State::Idle;
}

} // namespace xpool::fabric
