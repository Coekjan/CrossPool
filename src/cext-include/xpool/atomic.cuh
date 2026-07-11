#pragma once

/// \file xpool/atomic.cuh
/// \brief System-scope CUDA atomic helpers for shared CUDA state.
///
/// Transport arenas are CUDA IPC allocations mapped by multiple processes.
/// Descriptor status words, shutdown flags, and queue sequence counters must
/// therefore use system-scope atomics. Device- or block-scope atomics are not a
/// valid substitute for these helpers.

#include <cuda/atomic>

#include <type_traits>

namespace xpool::atomic {

/// Return a system-scope atomic reference for one IPC-visible CUDA scalar.
/// \tparam T Trivially copyable scalar type supported by CUDA atomics.
/// \param value CUDA-accessible scalar shared across contexts or devices.
/// \return Atomic reference with system visibility.
template <typename T>
__device__ inline cuda::atomic_ref<T, cuda::thread_scope_system>
system_atomic(T &value) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>(value);
}

/// Acquire-load a shared CUDA scalar with system visibility.
/// \tparam T Scalar type supported by CUDA atomics.
/// \param value CUDA-accessible scalar.
/// \return Loaded scalar value.
template <typename T> __device__ inline T load_acquire(T &value) {
  return system_atomic(value).load(cuda::memory_order_acquire);
}

/// Release-store a shared CUDA scalar with system visibility.
/// \tparam T Scalar type supported by CUDA atomics.
/// \tparam Desired Value type convertible to T.
/// \param value CUDA-accessible scalar.
/// \param desired Value to publish.
template <typename T, typename Desired>
__device__ inline void store_release(T &value, Desired desired) {
  static_assert(std::is_convertible_v<Desired, T>);
  system_atomic(value).store(static_cast<T>(desired),
                             cuda::memory_order_release);
}

/// Acq-rel compare-and-exchange a shared CUDA scalar.
/// \tparam T Scalar type supported by CUDA atomics.
/// \tparam Expected Expected value type convertible to T.
/// \tparam Desired Desired value type convertible to T.
/// \param value CUDA-accessible scalar.
/// \param expected Expected value for the exchange.
/// \param desired Value to publish when expected matches.
/// \return Observed previous value.
template <typename T, typename Expected, typename Desired>
__device__ inline T compare_exchange_acq_rel(T &value, Expected expected,
                                             Desired desired) {
  static_assert(std::is_convertible_v<Expected, T>);
  static_assert(std::is_convertible_v<Desired, T>);
  T observed = static_cast<T>(expected);
  (void)system_atomic(value).compare_exchange_strong(
      observed, static_cast<T>(desired), cuda::memory_order_acq_rel,
      cuda::memory_order_acquire);
  return observed;
}

} // namespace xpool::atomic
