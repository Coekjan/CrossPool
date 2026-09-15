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
XPOOL_DEVICE_FN inline void relax(std::uint32_t nanoseconds = kDefaultRelaxNanoseconds) { __nanosleep(nanoseconds); }

/// Device-global elapsed-time deadline used by polling protocols.
class Deadline {
public:
  /// Start a deadline at the current device-global timestamp.
  /// \pre timeout_nanoseconds is not UINT64_MAX; use never() instead.
  XPOOL_DEVICE_FN static Deadline after(std::uint64_t timeout_nanoseconds) {
    xpool::abort_if(timeout_nanoseconds == kNever);
    return Deadline{xpool::utils::time::now(), timeout_nanoseconds};
  }

  /// Reconstruct one non-renewable deadline across one-shot graph probes.
  XPOOL_DEVICE_FN static Deadline from_start(std::uint64_t started_at, std::uint64_t timeout_nanoseconds) {
    xpool::abort_if(timeout_nanoseconds == kNever);
    return Deadline{started_at, timeout_nanoseconds};
  }

  /// Construct a deadline that never expires.
  XPOOL_DEVICE_FN static Deadline never() { return Deadline{0, kNever}; }

  /// Return whether this bounded deadline has expired.
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

/// Evaluate one polling attempt without sleeping or retaining the CTA.
/// Readiness wins over cancellation and timeout when predicates become true
/// during the same observation.
template <Poll Ready, Poll Cancelled>
XPOOL_DEVICE_FN Status poll_once(const Deadline &deadline, Ready ready, Cancelled cancelled) {
  if (ready()) {
    return Status::Ready;
  }
  if (cancelled()) {
    return Status::Cancelled;
  }
  if (deadline.expired()) {
    return Status::TimedOut;
  }
  return Status::Pending;
}

/// Poll readiness until it succeeds, cancellation wins, or a deadline expires.
template <Poll Ready, Poll Cancelled>
XPOOL_DEVICE_FN Status until(const Deadline &deadline, Ready ready, Cancelled cancelled,
                             std::uint32_t relax_nanoseconds = kDefaultRelaxNanoseconds) {
  while (true) {
    switch (poll_once(deadline, ready, cancelled)) {
    case Status::Ready:
      return Status::Ready;
    case Status::Cancelled:
      return Status::Cancelled;
    case Status::TimedOut:
      return Status::TimedOut;
    case Status::Pending:
      relax(relax_nanoseconds);
      break;
    }
  }
}

/// Poll readiness until it succeeds or a deadline expires.
template <Poll Ready>
XPOOL_DEVICE_FN Status until(const Deadline &deadline, Ready ready,
                             std::uint32_t relax_nanoseconds = kDefaultRelaxNanoseconds) {
  return until(deadline, ready, [] { return false; }, relax_nanoseconds);
}

} // namespace xpool::utils::wait
