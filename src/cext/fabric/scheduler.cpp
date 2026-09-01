#include <c10/util/Exception.h>

#include <concepts>
#include <type_traits>
#include <variant>

#include <xpool/abort.hpp>
#include <xpool/fabric/scheduler.hpp>

namespace xpool::fabric {

Scheduler Scheduler::from(const SchedulerPolicy &policy, SchedulerEntry *entries,
                                std::size_t instance_count, std::size_t executor_lane_count) {
  TORCH_CHECK(entries != nullptr && instance_count != 0 && executor_lane_count != 0,
              "xpool FFN Scheduler requires caller-owned storage and positive counts");
  return std::visit(
      [=]<typename Policy>(const Policy &value) {
        if constexpr (std::same_as<Policy, SchedulerPolicy::FifoPolicy>) {
          return Scheduler{FifoState{.next_ticket = 0}, entries, instance_count, executor_lane_count};
        } else {
          return Scheduler{RandomState{.generator = xpool::utils::random::SplitMix64{value.seed}}, entries,
                              instance_count, executor_lane_count};
        }
      },
      policy.state_);
}

} // namespace xpool::fabric
