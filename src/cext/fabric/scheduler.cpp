#include <c10/util/Exception.h>

#include <concepts>
#include <type_traits>
#include <variant>

#include <xpool/abort.hpp>
#include <xpool/fabric/scheduler.hpp>

namespace xpool::fabric {

FfnScheduler FfnScheduler::from(const FfnSchedulerPolicy &policy) {
  return std::visit(
      []<typename Policy>(const Policy &value) {
        if constexpr (std::same_as<Policy, FfnSchedulerPolicy::FifoPolicy>) {
          return FfnScheduler{FifoState{.next_ticket_ = 0}};
        } else {
          return FfnScheduler{RandomState{.state_ = value.seed}};
        }
      },
      policy.state_);
}

} // namespace xpool::fabric
