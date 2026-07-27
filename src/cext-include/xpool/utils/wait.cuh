#pragma once

/// \file xpool/utils/wait.cuh
/// \brief Device-side bounded and cancellation-aware polling primitives.

#include <cstdint>
#include <limits>

#include <xpool/abort.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/time.cuh>
#include <xpool/utils/wait.hpp>

namespace xpool::utils::wait {

/// Default liveness deadline for one bounded device protocol phase.
inline constexpr std::uint64_t kDefaultTimeoutNanoseconds = 5'000'000'000ULL;
/// Default sleep interval between unsuccessful device polling attempts.
inline constexpr std::uint32_t kDefaultRelaxNanoseconds = 1'000U;

/// Yield one device thread between unsuccessful protocol polls.
/// \param nanoseconds Approximate sleep duration passed to the CUDA intrinsic.
XPOOL_DEVICE_FN inline void relax(
    std::uint32_t nanoseconds = kDefaultRelaxNanoseconds) {
  __nanosleep(nanoseconds);
}

/// Device-global elapsed-time deadline used by polling protocols.
class Deadline {
public:
  /// Start a deadline at the current device-global timestamp.
  /// \param timeout_nanoseconds Positive or zero elapsed-time budget.
  /// \return Deadline expiring after the supplied duration.
  /// \pre timeout_nanoseconds is not UINT64_MAX; use never() instead.
  XPOOL_DEVICE_FN static Deadline after(std::uint64_t timeout_nanoseconds) {
    xpool::abort_if(timeout_nanoseconds == kNever);
    return Deadline{xpool::utils::time::now(), timeout_nanoseconds};
  }

  /// Construct a deadline that never expires.
  /// \return Unbounded deadline for shutdown-aware resident polling.
  XPOOL_DEVICE_FN static Deadline never() { return Deadline{0, kNever}; }

  /// Return whether this bounded deadline has expired.
  /// \return False for never(); otherwise true once its elapsed budget passed.
  XPOOL_DEVICE_FN bool expired() const {
    return timeout_nanoseconds_ != kNever && xpool::utils::time::now() - started_at_ >= timeout_nanoseconds_;
  }

private:
  static constexpr std::uint64_t kNever = std::numeric_limits<std::uint64_t>::max();

  XPOOL_DEVICE_FN Deadline(std::uint64_t started_at, std::uint64_t timeout_nanoseconds)
      : started_at_(started_at), timeout_nanoseconds_(timeout_nanoseconds) {}

  std::uint64_t started_at_;
  std::uint64_t timeout_nanoseconds_;
};

/// Poll readiness until it succeeds, cancellation wins, or a deadline expires.
/// \tparam Ready Callable returning true when the operation completed.
/// \tparam Cancelled Callable returning true when the operation must stop.
/// \param deadline Elapsed-time deadline governing this wait.
/// \param ready Readiness predicate evaluated first on every iteration.
/// \param cancelled Cancellation predicate evaluated after readiness.
/// \param relax_nanoseconds Sleep interval after an unsuccessful iteration.
/// \return Terminal wait outcome.
template <Poll Ready, Poll Cancelled>
XPOOL_DEVICE_FN Result until(const Deadline &deadline, Ready ready, Cancelled cancelled,
                                             std::uint32_t relax_nanoseconds = kDefaultRelaxNanoseconds) {
  while (true) {
    if (ready()) {
      return Result::Ready;
    }
    if (cancelled()) {
      return Result::Cancelled;
    }
    if (deadline.expired()) {
      return Result::TimedOut;
    }
    relax(relax_nanoseconds);
  }
}

/// Poll readiness until it succeeds or a deadline expires.
/// \tparam Ready Callable returning true when the operation completed.
/// \param deadline Elapsed-time deadline governing this wait.
/// \param ready Readiness predicate evaluated on every iteration.
/// \param relax_nanoseconds Sleep interval after an unsuccessful iteration.
/// \return Ready or TimedOut.
template <Poll Ready>
XPOOL_DEVICE_FN Result until(const Deadline &deadline, Ready ready,
                                             std::uint32_t relax_nanoseconds = kDefaultRelaxNanoseconds) {
  while (true) {
    if (ready()) {
      return Result::Ready;
    }
    if (deadline.expired()) {
      return Result::TimedOut;
    }
    relax(relax_nanoseconds);
  }
}

} // namespace xpool::utils::wait
