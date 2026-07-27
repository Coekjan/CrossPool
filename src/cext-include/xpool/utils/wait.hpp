#pragma once

/// \file xpool/utils/wait.hpp
/// \brief Shared host and device polling outcomes plus host-side waits.

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>

namespace xpool::utils::wait {

/// Side-effecting polling operation that reports whether progress completed.
template <typename F>
concept Poll = std::invocable<F &> && std::same_as<std::invoke_result_t<F &>, bool>;

/// Terminal outcome of one bounded polling operation.
enum class Result : std::uint32_t {
  /// The readiness predicate completed the wait.
  Ready,
  /// The cancellation predicate stopped the wait before readiness.
  Cancelled,
  /// The deadline expired before readiness or cancellation.
  TimedOut,
};

/// Poll host readiness until it succeeds, cancellation wins, or time expires.
/// \tparam Ready Callable returning true when the operation completed.
/// \tparam Cancelled Callable returning true when the operation must stop.
/// \param deadline Monotonic host deadline governing this wait.
/// \param ready Readiness predicate evaluated first on every iteration.
/// \param cancelled Cancellation predicate evaluated after readiness.
/// \param interval Maximum sleep duration after an unsuccessful iteration.
/// \return Terminal wait outcome.
template <Poll Ready, Poll Cancelled>
Result until(std::chrono::steady_clock::time_point deadline, Ready ready,
             Cancelled cancelled,
             std::chrono::steady_clock::duration interval) {
  while (true) {
    if (ready()) {
      return Result::Ready;
    }
    if (cancelled()) {
      return Result::Cancelled;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return Result::TimedOut;
    }
    std::this_thread::sleep_for(std::min(interval, deadline - now));
  }
}

/// Poll host readiness until it succeeds or time expires.
/// \tparam Ready Callable returning true when the operation completed.
/// \param deadline Monotonic host deadline governing this wait.
/// \param ready Readiness predicate evaluated on every iteration.
/// \param interval Maximum sleep duration after an unsuccessful iteration.
/// \return Ready or TimedOut.
template <Poll Ready>
Result until(std::chrono::steady_clock::time_point deadline, Ready ready,
             std::chrono::steady_clock::duration interval) {
  return until(deadline, std::move(ready), [] { return false; }, interval);
}

} // namespace xpool::utils::wait
