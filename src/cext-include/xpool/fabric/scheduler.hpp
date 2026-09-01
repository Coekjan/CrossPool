#pragma once

/// \file xpool/fabric/scheduler.hpp
/// \brief Host policy and Coordinator-private device Scheduler.

#include <cuda/std/optional>
#include <cuda/std/variant>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include <c10/util/Exception.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/random.hpp>

namespace xpool::fabric {

/// Immutable host-side policy used to construct one private Scheduler.
class SchedulerPolicy {
public:
  /// Supported scheduling policy implementations.
  enum Type : std::uint32_t {
    Fifo = 1,
    Random = 2,
  };

  /// Construct deterministic FIFO scheduling policy.
  static SchedulerPolicy fifo();
  /// Construct deterministic pseudo-random scheduling policy.
  /// \throws c10::Error when seed is zero.
  static SchedulerPolicy random(std::uint64_t seed);
  /// Return the selected policy implementation.
  Type type() const;
  bool operator==(const SchedulerPolicy &other) const = default;

private:
  friend class Scheduler;

  struct FifoPolicy {
    constexpr bool operator==(const FifoPolicy &) const = default;
  };

  struct RandomPolicy {
    std::uint64_t seed;
    constexpr bool operator==(const RandomPolicy &) const = default;
  };

  using State = std::variant<FifoPolicy, RandomPolicy>;
  explicit SchedulerPolicy(State state) : state_(std::move(state)) {}
  State state_;
};

inline SchedulerPolicy SchedulerPolicy::fifo() {
  return SchedulerPolicy{FifoPolicy{}};
}

inline SchedulerPolicy SchedulerPolicy::random(std::uint64_t seed) {
  TORCH_CHECK(seed != 0, "xpool random FFN scheduling requires a nonzero seed");
  return SchedulerPolicy{RandomPolicy{.seed = seed}};
}

inline SchedulerPolicy::Type SchedulerPolicy::type() const {
  if (std::holds_alternative<FifoPolicy>(state_)) {
    return Fifo;
  }
  if (std::holds_alternative<RandomPolicy>(state_)) {
    return Random;
  }
  xpool::abort();
}

/// Private state for one Instance's queued or active invocation.
struct SchedulerEntry {
  /// Lifecycle of one instance's scheduler slot.
  enum class State : std::uint32_t {
    Idle = 0,
    Ready = 1,
    Active = 2,
  };

  /// Latest invocation submitted by the instance.
  Invocation invocation{};
  /// Last invocation sequence released for the instance.
  std::uint64_t completed_sequence = 0;
  /// FIFO ticket assigned when the invocation became ready.
  std::uint64_t ready_ticket = 0;
  /// Lane assigned while the invocation is active.
  std::size_t executor_lane_index = 0;
  /// Current slot lifecycle state.
  State state = State::Idle;
};

/// Coordinator-private Scheduler over caller-owned device storage.
class Scheduler {
public:
  /// Immutable scheduling result for one admitted invocation.
  class Decision {
  public:
#if defined(__CUDACC__)
    /// Return the admitted invocation.
    XPOOL_DEVICE_FN const Invocation &invocation() const { return invocation_; }
    /// Return the lane assigned to the invocation.
    XPOOL_DEVICE_FN std::size_t executor_lane_index() const { return executor_lane_index_; }
#endif

  private:
    friend class Scheduler;
#if defined(__CUDACC__)
    XPOOL_DEVICE_FN Decision(const Invocation &invocation, std::size_t executor_lane_index)
        : invocation_(invocation), executor_lane_index_(executor_lane_index) {}
#endif

    Invocation invocation_{};
    std::size_t executor_lane_index_ = 0;
  };

  /// Construct a scheduler over caller-owned device storage.
  /// \throws c10::Error when storage is null or either count is zero.
  /// \pre entries remains alive and Device-accessible while the Scheduler is used.
  static Scheduler from(const SchedulerPolicy &policy, SchedulerEntry *entries,
                           std::size_t instance_count, std::size_t executor_lane_count);

#if defined(__CUDACC__)
  /// Test whether an instance has a queued or active invocation.
  XPOOL_DEVICE_FN bool has_unresolved_invocation(std::size_t instance_index) const;
  /// Enqueue a ready invocation under the configured policy.
  /// \return Monotonic FIFO ticket, or zero for random scheduling.
  XPOOL_DEVICE_FN std::uint64_t enqueue(const Invocation &invocation);
  /// Admit one ready invocation when a lane is idle.
  /// \return A decision, or no value when no scheduling pair is available.
  XPOOL_DEVICE_FN cuda::std::optional<Decision> try_schedule();
  /// Look up the active decision for an instance.
  XPOOL_DEVICE_FN cuda::std::optional<Decision> active_decision(std::size_t instance_index) const;
  /// Release an active invocation and its lane.
  XPOOL_DEVICE_FN void release(const InvocationKey &key, std::size_t executor_lane_index);
#endif

private:
  struct FifoState {
    std::uint64_t next_ticket;
  };

  struct RandomState {
    xpool::utils::random::SplitMix64 generator;
  };

  using State = cuda::std::variant<FifoState, RandomState>;

  Scheduler(State state, SchedulerEntry *entries, std::size_t instance_count,
               std::size_t executor_lane_count)
      : state_(std::move(state)), entries_(entries), instance_count_(instance_count),
        executor_lane_count_(executor_lane_count) {}

#if defined(__CUDACC__)
  XPOOL_DEVICE_FN std::size_t find_idle_lane() const;
  XPOOL_DEVICE_FN Decision admit(std::size_t instance_index, std::size_t executor_lane_index);
#endif

  State state_;
  SchedulerEntry *entries_;
  std::size_t instance_count_;
  std::size_t executor_lane_count_;
};

static_assert(std::is_standard_layout_v<SchedulerEntry>);
static_assert(std::is_trivially_copyable_v<SchedulerEntry>);
static_assert(std::is_trivially_copyable_v<Scheduler>);

} // namespace xpool::fabric
