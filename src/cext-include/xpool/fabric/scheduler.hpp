#pragma once

/// \file xpool/fabric/scheduler.hpp
/// \brief Typed host policy and device Scheduler for Fabric Executor admission.

#include <cuda/std/variant>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/macros.hpp>

namespace xpool::fabric {

/// Immutable host-side policy used to construct one arena Scheduler.
class FfnSchedulerPolicy {
public:
  /// Supported scheduling algorithms.
  enum Type : std::uint32_t {
    /// Select the eligible Ready invocation with the smallest ticket.
    Fifo = 1,
    /// Select one eligible Ready invocation with deterministic SplitMix64.
    Random = 2,
  };

  /// Construct a FIFO policy with no seed state.
  /// \return FIFO scheduling policy.
  static FfnSchedulerPolicy fifo();

  /// Construct a random policy with one explicit nonzero seed.
  /// \param seed Nonzero deterministic SplitMix64 seed.
  /// \return Random scheduling policy.
  static FfnSchedulerPolicy random(std::uint64_t seed);

  /// Return the semantic policy selected by the active variant.
  /// \return Fifo or Random.
  Type type() const;

  /// Compare policy alternatives and their immutable values.
  /// \return True when policy kind and seed state are equal.
  bool operator==(const FfnSchedulerPolicy &) const = default;

private:
  friend class FfnScheduler;

  struct FifoPolicy {
    constexpr bool operator==(const FifoPolicy &) const = default;
  };

  struct RandomPolicy {
    std::uint64_t seed;

    constexpr bool operator==(const RandomPolicy &) const = default;
  };

  using State = std::variant<FifoPolicy, RandomPolicy>;

  explicit FfnSchedulerPolicy(State state) : state_(std::move(state)) {}

  State state_;
};

inline FfnSchedulerPolicy FfnSchedulerPolicy::fifo() {
  return FfnSchedulerPolicy{FifoPolicy{}};
}

inline FfnSchedulerPolicy FfnSchedulerPolicy::random(std::uint64_t seed) {
  TORCH_CHECK(seed != 0, "xpool random FFN scheduling requires a nonzero seed");
  return FfnSchedulerPolicy{RandomPolicy{.seed = seed}};
}

inline FfnSchedulerPolicy::Type FfnSchedulerPolicy::type() const {
  if (std::holds_alternative<FifoPolicy>(state_)) {
    return Fifo;
  }
  if (std::holds_alternative<RandomPolicy>(state_)) {
    return Random;
  }
  xpool::abort();
}

/// Model-scoped Scheduler lifecycle and active Executor lease.
class FfnSchedulerEntry {
public:
  /// Scheduler entry lifecycle states.
  enum class State : std::uint32_t {
    /// No invocation is queued or active for this model.
    Idle = 0,
    /// One complete invocation is eligible for scheduling.
    Ready = 1,
    /// One invocation owns a distributed Executor lease.
    Active = 2,
  };

#if defined(__CUDACC__)
  /// Acquire-load the current lifecycle state.
  /// \return Idle, Ready, or Active.
  XPOOL_DEVICE_FN State state() const;

  /// Return the only sequence eligible after retained publication history.
  /// \return Completed sequence plus one.
  XPOOL_DEVICE_FN std::uint64_t next_sequence() const;

  /// Return the invocation stored by the latest enqueue.
  /// \return Current queued or active invocation.
  XPOOL_DEVICE_FN const FfnInvocation &invocation() const { return invocation_; }

  /// Return the positive FIFO ticket, or zero for Random policy.
  /// \return Queue ticket selected during enqueue.
  XPOOL_DEVICE_FN std::uint64_t ready_ticket() const { return ready_ticket_; }

  /// Return the retained Coordinator trace identity, or zero when unavailable.
  /// \return Coordinator-local trace sequence.
  XPOOL_DEVICE_FN std::uint64_t scheduler_trace_id() const { return scheduler_trace_id_; }

  /// Return the Executor leased while this entry is Active.
  /// \return Active Executor index, or zero outside Active.
  XPOOL_DEVICE_FN std::size_t executor_index() const { return executor_index_; }

  /// Publish one complete invocation as Ready.
  /// \param invocation Complete model-level invocation.
  /// \param ready_ticket Positive FIFO ticket, or zero for Random.
  /// \param scheduler_trace_id Coordinator trace sequence, or zero when absent.
  XPOOL_DEVICE_FN void enqueue(const FfnInvocation &invocation, std::uint64_t ready_ticket,
                               std::uint64_t scheduler_trace_id);

  /// Lease one idle Executor and move Ready to Active.
  /// \param executor_index Idle distributed Executor to lease.
  XPOOL_DEVICE_FN void admit(std::size_t executor_index);

  /// Release one matching active invocation and return to Idle.
  /// \param key Active invocation identity.
  /// \param executor_index Executor currently leased by the invocation.
  XPOOL_DEVICE_FN void release(const FfnInvocationKey &key, std::size_t executor_index);
#endif

private:
  FfnInvocation invocation_{};
  std::uint64_t completed_sequence_ = 0;
  std::uint64_t ready_ticket_ = 0;
  std::uint64_t scheduler_trace_id_ = 0;
  std::size_t executor_index_ = 0;
  std::uint32_t state_ = 0;
};

/// Mutable Coordinator-owned scheduler stored in the Fabric arena.
class FfnScheduler {
public:
  /// Result of one scheduling attempt.
  class Decision {
  public:
#if defined(__CUDACC__)
    /// Return whether one invocation was scheduled.
    /// \return True when this decision owns a valid invocation and Executor.
    XPOOL_DEVICE_FN explicit operator bool() const { return scheduled_; }

    /// Return the scheduled invocation.
    /// \return Invocation selected by the Scheduler.
    XPOOL_DEVICE_FN const FfnInvocation &invocation() const {
      xpool::abort_if(!scheduled_);
      return invocation_;
    }

    /// Return the Executor leased by this decision.
    /// \return Distributed Executor index selected by the Scheduler.
    XPOOL_DEVICE_FN std::size_t executor_index() const {
      xpool::abort_if(!scheduled_);
      return executor_index_;
    }
#endif

  private:
    friend class FfnScheduler;

    bool scheduled_ = false;
    FfnInvocation invocation_{};
    std::size_t executor_index_ = 0;
  };

  /// Construct mutable scheduler state from one immutable policy.
  /// \param policy Valid host-side scheduling policy.
  /// \return Arena-storable mutable Scheduler state.
  static FfnScheduler from(const FfnSchedulerPolicy &policy);

#if defined(__CUDACC__)
  /// Enqueue one complete model-level invocation.
  /// \param arena Coordinator-local Fabric arena view.
  /// \param invocation Complete invocation eligible for model-level queuing.
  XPOOL_DEVICE_FN void enqueue(const FabricArenaView &arena, const FfnInvocation &invocation);

  /// Select one eligible Ready invocation and idle Executor.
  /// \param arena Coordinator-local Fabric arena view.
  /// \return Scheduled decision, or an empty decision when no pair is eligible.
  XPOOL_DEVICE_FN Decision schedule(const FabricArenaView &arena);

  /// Release one matching model and Executor lease.
  /// \param arena Coordinator-local Fabric arena view.
  /// \param key Completed invocation identity.
  /// \param executor_index Executor leased to the completed invocation.
  XPOOL_DEVICE_FN void release(const FabricArenaView &arena, const FfnInvocationKey &key,
                               std::size_t executor_index);
#endif

private:
  struct FifoState {
    std::uint64_t next_ticket_;
  };

  struct RandomState {
    std::uint64_t state_;

#if defined(__CUDACC__)
    XPOOL_DEVICE_FN std::uint64_t next();
#endif
  };

  using State = cuda::std::variant<FifoState, RandomState>;

#if defined(__CUDACC__)
  struct EnqueueVisitor {
    const FabricArenaView &arena;
    const FfnInvocation &invocation;
    FabricTraceRecord *trace;
    std::uint64_t trace_id;

    XPOOL_DEVICE_FN void operator()(FifoState &state) const;
    XPOOL_DEVICE_FN void operator()(RandomState &state) const;
  };

  struct ScheduleVisitor {
    const FabricArenaView &arena;
    std::size_t executor_index;

    XPOOL_DEVICE_FN Decision operator()(FifoState &state) const;
    XPOOL_DEVICE_FN Decision operator()(RandomState &state) const;
  };

  XPOOL_DEVICE_FN static std::size_t find_idle_executor(const FabricArenaView &arena);
  XPOOL_DEVICE_FN static Decision admit(FfnSchedulerEntry &entry, std::size_t executor_index);
#endif

  explicit FfnScheduler(State state) : state_(std::move(state)) {}

  State state_;
};

static_assert(std::is_standard_layout_v<FfnSchedulerEntry>);
static_assert(std::is_trivially_copyable_v<FfnSchedulerEntry>);
static_assert(std::is_trivially_copyable_v<FfnScheduler>);

} // namespace xpool::fabric
