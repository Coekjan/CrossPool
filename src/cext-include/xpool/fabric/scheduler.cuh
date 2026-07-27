#pragma once

/// \file xpool/fabric/scheduler.cuh
/// \brief Device implementation of model-level Fabric scheduling.

#include <cuda/atomic>
#include <cuda/std/variant>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <xpool/abort.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/macros.hpp>

namespace xpool::fabric {

XPOOL_DEVICE_FN inline std::size_t FfnScheduler::find_idle_executor(const FabricArenaView &arena) {
  const auto &layout = arena.layout();
  for (auto executor_index = std::size_t{0}; executor_index < layout.executor_count; ++executor_index) {
    auto active = false;
    for (auto model_index = std::size_t{0}; model_index < layout.model_count; ++model_index) {
      const auto &entry = arena.scheduler_entry(model_index);
      if (entry.state() == FfnSchedulerEntry::State::Active && entry.executor_index() == executor_index) {
        active = true;
        break;
      }
    }
    if (!active) {
      return executor_index;
    }
  }
  return layout.executor_count;
}

XPOOL_DEVICE_FN inline FfnSchedulerEntry::State FfnSchedulerEntry::state() const {
  auto state = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>{const_cast<std::uint32_t &>(state_)};
  const auto value = state.load(cuda::memory_order_acquire);
  xpool::abort_if(value > static_cast<std::uint32_t>(State::Active));
  return static_cast<State>(value);
}

XPOOL_DEVICE_FN inline std::uint64_t FfnSchedulerEntry::next_sequence() const {
  xpool::abort_if(completed_sequence_ == std::numeric_limits<std::uint64_t>::max());
  return completed_sequence_ + 1;
}

XPOOL_DEVICE_FN inline void FfnSchedulerEntry::enqueue(const FfnInvocation &invocation, std::uint64_t ready_ticket,
                                                       std::uint64_t scheduler_trace_id) {
  xpool::abort_if(state() != State::Idle || invocation.validate() != xpool::abi::FfnResultCode::Ok ||
                  invocation.key.invocation_sequence != next_sequence());
  invocation_ = invocation;
  ready_ticket_ = ready_ticket;
  scheduler_trace_id_ = scheduler_trace_id;
  executor_index_ = 0;
  auto state = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>{state_};
  state.store(static_cast<std::uint32_t>(State::Ready), cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline void FfnSchedulerEntry::admit(std::size_t executor_index) {
  xpool::abort_if(state() != State::Ready);
  executor_index_ = executor_index;
  auto state = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>{state_};
  state.store(static_cast<std::uint32_t>(State::Active), cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline void FfnSchedulerEntry::release(const FfnInvocationKey &key, std::size_t executor_index) {
  xpool::abort_if(state() != State::Active || invocation_.key != key || executor_index_ != executor_index);
  completed_sequence_ = key.invocation_sequence;
  invocation_ = {};
  ready_ticket_ = 0;
  scheduler_trace_id_ = 0;
  executor_index_ = 0;
  auto state = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>{state_};
  state.store(static_cast<std::uint32_t>(State::Idle), cuda::memory_order_release);
}

XPOOL_DEVICE_FN inline std::uint64_t FfnScheduler::RandomState::next() {
  state_ += 0x9e3779b97f4a7c15ULL;
  auto value = state_;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

XPOOL_DEVICE_FN inline FfnScheduler::Decision FfnScheduler::admit(FfnSchedulerEntry &entry,
                                                                  std::size_t executor_index) {
  const auto invocation = entry.invocation();
  entry.admit(executor_index);
  auto decision = Decision{};
  decision.scheduled_ = true;
  decision.invocation_ = invocation;
  decision.executor_index_ = executor_index;
  return decision;
}

XPOOL_DEVICE_FN inline void FfnScheduler::EnqueueVisitor::operator()(FifoState &state) const {
  ++state.next_ticket_;
  xpool::abort_if(state.next_ticket_ == 0);
  if (trace != nullptr) {
    trace->fifo_enqueued(state.next_ticket_);
  }
  arena.scheduler_entry(invocation.key.model_index).enqueue(invocation, state.next_ticket_, trace_id);
}

XPOOL_DEVICE_FN inline void FfnScheduler::EnqueueVisitor::operator()(RandomState &) const {
  if (trace != nullptr) {
    trace->random_enqueued();
  }
  arena.scheduler_entry(invocation.key.model_index).enqueue(invocation, 0, trace_id);
}

XPOOL_DEVICE_FN inline FfnScheduler::Decision FfnScheduler::ScheduleVisitor::operator()(FifoState &) const {
  const auto &layout = arena.layout();
  auto selected_model = layout.model_count;
  auto selected_ticket = std::numeric_limits<std::uint64_t>::max();
  for (auto model_index = std::size_t{0}; model_index < layout.model_count; ++model_index) {
    const auto &entry = arena.scheduler_entry(model_index);
    if (entry.state() == FfnSchedulerEntry::State::Ready && entry.ready_ticket() != 0 &&
        entry.ready_ticket() < selected_ticket) {
      selected_model = model_index;
      selected_ticket = entry.ready_ticket();
    }
  }
  return selected_model == layout.model_count
             ? Decision{}
             : admit(arena.scheduler_entry(selected_model), executor_index);
}

XPOOL_DEVICE_FN inline FfnScheduler::Decision FfnScheduler::ScheduleVisitor::operator()(RandomState &state) const {
  const auto &layout = arena.layout();
  auto has_candidate = false;
  for (auto model_index = std::size_t{0}; model_index < layout.model_count; ++model_index) {
    if (arena.scheduler_entry(model_index).state() == FfnSchedulerEntry::State::Ready) {
      has_candidate = true;
      break;
    }
  }
  if (!has_candidate) {
    return {};
  }

  auto selected_model = layout.model_count;
  auto selected_score = std::numeric_limits<std::uint64_t>::max();
  for (auto model_index = std::size_t{0}; model_index < layout.model_count; ++model_index) {
    const auto &entry = arena.scheduler_entry(model_index);
    if (entry.state() != FfnSchedulerEntry::State::Ready) {
      continue;
    }
    const auto score = state.next();
    if (selected_model == layout.model_count || score < selected_score) {
      selected_model = model_index;
      selected_score = score;
    }
  }
  return admit(arena.scheduler_entry(selected_model), executor_index);
}

XPOOL_DEVICE_FN inline void FfnScheduler::enqueue(const FabricArenaView &arena, const FfnInvocation &invocation) {
  xpool::abort_if(invocation.validate() != xpool::abi::FfnResultCode::Ok ||
                  invocation.key.model_index >= arena.layout().model_count);
  const auto &model = arena.model_layout(invocation.key.model_index);
  xpool::abort_if(invocation.layer_ordinal >= model.layer_count);
  const auto &layer = arena.layer_layout(model.layer_begin + invocation.layer_ordinal);
  const auto trace_entry = arena.reserve_trace();
  auto *trace = trace_entry ? &trace_entry.record() : nullptr;
  if (trace != nullptr) {
    trace->begin_coordinator(trace_entry.sequence(), invocation, layer.layer_id);
  }
  cuda::std::visit(EnqueueVisitor{arena, invocation, trace, trace_entry.sequence()}, state_);
}

XPOOL_DEVICE_FN inline FfnScheduler::Decision FfnScheduler::schedule(const FabricArenaView &arena) {
  const auto executor_index = find_idle_executor(arena);
  if (executor_index == arena.layout().executor_count) {
    return {};
  }
  return cuda::std::visit(ScheduleVisitor{arena, executor_index}, state_);
}

XPOOL_DEVICE_FN inline void FfnScheduler::release(const FabricArenaView &arena, const FfnInvocationKey &key,
                                                  std::size_t executor_index) {
  xpool::abort_if(!key.valid() || key.model_index >= arena.layout().model_count ||
                  executor_index >= arena.layout().executor_count);
  arena.scheduler_entry(key.model_index).release(key, executor_index);
}

} // namespace xpool::fabric
