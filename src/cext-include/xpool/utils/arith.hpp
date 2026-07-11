#pragma once

/// \file xpool/utils/arith.hpp
/// \brief Host-side checked integer arithmetic helpers for native geometry.

#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>

/// Checked arithmetic utilities used by xpool native host code.
namespace xpool::utils::arith {

/// Add two non-negative int64 values and reject overflow.
/// \param left Left non-negative operand.
/// \param right Right non-negative operand.
/// \param context Human-readable expression being computed.
/// \return The exact int64 sum.
/// \throws c10::Error if either operand is negative or the sum overflows.
inline std::int64_t checked_nonnegative_add(std::int64_t left,
                                            std::int64_t right,
                                            const char *context) {
  TORCH_CHECK(left >= 0 && right >= 0,
              "xpool checked arithmetic requires non-negative operands for ",
              context);
  std::int64_t output = 0;
#if defined(__GNUC__) || defined(__clang__)
  bool overflow = __builtin_add_overflow(left, right, &output);
#else
  bool overflow = left > std::numeric_limits<std::int64_t>::max() - right;
  if (!overflow) {
    output = left + right;
  }
#endif
  TORCH_CHECK(!overflow,
              "xpool checked arithmetic overflows int64 while computing ",
              context);
  return output;
}

/// Multiply two non-negative int64 values and reject overflow.
/// \param left Left non-negative operand.
/// \param right Right non-negative operand.
/// \param context Human-readable expression being computed.
/// \return The exact int64 product.
/// \throws c10::Error if either operand is negative or the product overflows.
inline std::int64_t checked_nonnegative_mul(std::int64_t left,
                                            std::int64_t right,
                                            const char *context) {
  TORCH_CHECK(left >= 0 && right >= 0,
              "xpool checked arithmetic requires non-negative operands for ",
              context);
  std::int64_t output = 0;
#if defined(__GNUC__) || defined(__clang__)
  bool overflow = __builtin_mul_overflow(left, right, &output);
#else
  bool overflow =
      right != 0 && left > std::numeric_limits<std::int64_t>::max() / right;
  if (!overflow) {
    output = left * right;
  }
#endif
  TORCH_CHECK(!overflow,
              "xpool checked arithmetic overflows int64 while computing ",
              context);
  return output;
}

/// Convert a non-negative int64 value to uint32 and reject narrowing.
/// \param value Non-negative value to convert.
/// \param context Human-readable expression being converted.
/// \return The exact uint32 value.
/// \throws c10::Error if value is negative or does not fit uint32.
inline std::uint32_t checked_u32(std::int64_t value, const char *context) {
  TORCH_CHECK(value >= 0 &&
                  value <= static_cast<std::int64_t>(
                               std::numeric_limits<std::uint32_t>::max()),
              "xpool checked arithmetic cannot convert ", context,
              " to uint32 without narrowing");
  return static_cast<std::uint32_t>(value);
}

/// Round a non-negative int64 value up to a positive alignment.
/// \param value Non-negative value to align.
/// \param alignment Positive byte or element alignment.
/// \param context Human-readable expression being aligned.
/// \return The smallest aligned int64 value greater than or equal to value.
/// \throws c10::Error if value is negative, alignment is non-positive, or the
/// rounded value overflows int64.
inline std::int64_t align_up(std::int64_t value, std::int64_t alignment,
                             const char *context) {
  TORCH_CHECK(alignment > 0,
              "xpool checked arithmetic requires positive alignment for ",
              context);
  std::int64_t rounded = checked_nonnegative_add(value, alignment - 1, context);
  return (rounded / alignment) * alignment;
}

} // namespace xpool::utils::arith

/// Checked uint32 conversion with the expression text as diagnostic context.
#define XPOOL_CHECKED_U32(value)                                               \
  ::xpool::utils::arith::checked_u32((value), #value)

/// Checked non-negative int64 addition with expression diagnostics.
#define XPOOL_CHECKED_ADD(left, right)                                         \
  ::xpool::utils::arith::checked_nonnegative_add((left), (right),              \
                                                 #left " + " #right)

/// Checked non-negative int64 multiplication with expression diagnostics.
#define XPOOL_CHECKED_MUL(left, right)                                         \
  ::xpool::utils::arith::checked_nonnegative_mul((left), (right),              \
                                                 #left " * " #right)

/// Checked non-negative int64 triple multiplication with expression
/// diagnostics.
#define XPOOL_CHECKED_MUL3(left, middle, right)                                \
  XPOOL_CHECKED_MUL(XPOOL_CHECKED_MUL((left), (middle)), (right))

/// Checked non-negative int64 alignment with expression diagnostics.
#define XPOOL_ALIGN_UP(value, alignment)                                       \
  ::xpool::utils::arith::align_up((value), (alignment),                        \
                                  "align_up(" #value ", " #alignment ")")

/// Checked non-negative int64 multiplication followed by alignment.
#define XPOOL_ALIGN_UP_MUL(left, right, alignment)                             \
  XPOOL_ALIGN_UP(XPOOL_CHECKED_MUL((left), (right)), (alignment))

/// Checked non-negative int64 triple multiplication followed by alignment.
#define XPOOL_ALIGN_UP_MUL3(left, middle, right, alignment)                    \
  XPOOL_ALIGN_UP(XPOOL_CHECKED_MUL3((left), (middle), (right)), (alignment))

/// Checked non-negative int64 addition where the right operand is a product.
#define XPOOL_CHECKED_ADD_MUL(base, left, right)                               \
  XPOOL_CHECKED_ADD((base), XPOOL_CHECKED_MUL((left), (right)))

/// Checked non-negative int64 addition where the right operand is an aligned
/// product.
#define XPOOL_CHECKED_ADD_ALIGN_UP_MUL(base, left, right, alignment)           \
  XPOOL_CHECKED_ADD((base), XPOOL_ALIGN_UP_MUL((left), (right), (alignment)))
