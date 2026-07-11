#pragma once

/// \file xpool/utils/arith.cuh
/// \brief Device-side checked integer arithmetic helpers.

#include <cstdint>
#include <xpool/utils/device.cuh>

/// Checked arithmetic utilities shared by xpool CUDA device code.
namespace xpool::utils::arith {

/// Add two uint64 values and trap on overflow.
/// \param left Left operand.
/// \param right Right operand.
/// \return Exact sum when it fits in uint64.
__device__ __forceinline__ std::uint64_t checked_add(std::uint64_t left,
                                                     std::uint64_t right) {
  constexpr std::uint64_t max_value = ~std::uint64_t{0};
  xpool::utils::device::trap_if(left > max_value - right);
  return left + right;
}

/// Multiply two uint64 values and trap on overflow.
/// \param left Left operand.
/// \param right Right operand.
/// \return Exact product when it fits in uint64.
__device__ __forceinline__ std::uint64_t checked_mul(std::uint64_t left,
                                                     std::uint64_t right) {
  constexpr std::uint64_t max_value = ~std::uint64_t{0};
  xpool::utils::device::trap_if(right != 0U && left > max_value / right);
  return left * right;
}

/// Convert a signed int64 value to uint64 and trap when negative.
/// \param value Signed value expected to be non-negative.
/// \return Exact uint64 value.
__device__ __forceinline__ std::uint64_t
checked_nonnegative(std::int64_t value) {
  xpool::utils::device::trap_if(value < 0);
  return static_cast<std::uint64_t>(value);
}

} // namespace xpool::utils::arith
