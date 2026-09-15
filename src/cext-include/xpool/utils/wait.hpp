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

/// Outcome of one bounded polling observation.
enum class Status : std::uint32_t {
  /// No terminal predicate matched; the caller may schedule another probe.
  Pending = 0,
  /// The readiness predicate completed the wait.
  Ready = 1,
  /// The cancellation predicate stopped the wait before readiness.
  Cancelled = 2,
  /// The deadline expired before readiness or cancellation.
  TimedOut = 3,
};

/// Poll host readiness until it succeeds, cancellation wins, or time expires.
template <Poll Ready, Poll Cancelled>
Status until(std::chrono::steady_clock::time_point deadline, Ready ready, Cancelled cancelled,
             std::chrono::steady_clock::duration interval) {
  while (true) {
    if (ready()) {
      return Status::Ready;
    }
    if (cancelled()) {
      return Status::Cancelled;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return Status::TimedOut;
    }
    std::this_thread::sleep_for(std::min(interval, deadline - now));
  }
}

/// Poll host readiness until it succeeds or time expires.
template <Poll Ready>
Status until(std::chrono::steady_clock::time_point deadline, Ready ready,
             std::chrono::steady_clock::duration interval) {
  return until(deadline, std::move(ready), [] { return false; }, interval);
}

} // namespace xpool::utils::wait
