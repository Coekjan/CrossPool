#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <xpool/utils/checked.hpp>

static_assert(xpool::utils::checked::CheckedInteger<std::int32_t>);
static_assert(xpool::utils::checked::CheckedInteger<std::uint64_t>);
static_assert(!xpool::utils::checked::CheckedInteger<bool>);
static_assert(xpool::utils::checked::SameCheckedIntegerOperands<std::size_t, std::size_t>);
static_assert(!xpool::utils::checked::SameCheckedIntegerOperands<std::uint32_t, std::uint64_t>);
static_assert(xpool::utils::checked::CheckedUnsignedInteger<std::uint32_t>);
static_assert(!xpool::utils::checked::CheckedUnsignedInteger<std::int32_t>);

TEST(CheckedSizeTest, SumsSizeOperands) {
  EXPECT_EQ(xpool::utils::checked::sum(std::size_t{0}), 0U);
  EXPECT_EQ(xpool::utils::checked::sum(std::size_t{1}, std::size_t{2}, std::size_t{3}), 6U);
  EXPECT_THROW((void)xpool::utils::checked::sum(std::numeric_limits<std::size_t>::max(), std::size_t{1}), c10::Error);
}

TEST(CheckedSizeTest, MultipliesSizeOperands) {
  EXPECT_EQ(xpool::utils::checked::prod(std::size_t{7}), 7U);
  EXPECT_EQ(xpool::utils::checked::prod(std::size_t{2}, std::size_t{3}, std::size_t{4}), 24U);
  EXPECT_EQ(xpool::utils::checked::prod(std::numeric_limits<std::size_t>::max(), std::size_t{2}, std::size_t{0}), 0U);
  EXPECT_THROW((void)xpool::utils::checked::prod(std::numeric_limits<std::size_t>::max(), std::size_t{2}), c10::Error);
}

TEST(CheckedIntegerTest, PreservesSignedOperandType) {
  static_assert(std::same_as<decltype(xpool::utils::checked::sum(std::int32_t{1}, std::int32_t{2})), std::int32_t>);
  static_assert(std::same_as<decltype(xpool::utils::checked::prod(std::uint64_t{2}, std::uint64_t{3})), std::uint64_t>);
  EXPECT_EQ(xpool::utils::checked::sum(std::int32_t{-4}, std::int32_t{7}), 3);
  EXPECT_EQ(xpool::utils::checked::prod(std::int64_t{-3}, std::int64_t{4}), -12);
  EXPECT_THROW((void)xpool::utils::checked::sum(std::numeric_limits<std::int32_t>::max(), std::int32_t{1}), c10::Error);
  EXPECT_THROW((void)xpool::utils::checked::prod(std::numeric_limits<std::int64_t>::min(), std::int64_t{-1}),
               c10::Error);
}

TEST(CheckedSizeTest, AlignsSizeOperands) {
  EXPECT_EQ(xpool::utils::checked::align_up(std::size_t{0}, std::size_t{256}), 0U);
  EXPECT_EQ(xpool::utils::checked::align_up(std::size_t{1}, std::size_t{256}), 256U);
  EXPECT_EQ(xpool::utils::checked::align_up(std::size_t{256}, std::size_t{256}), 256U);
  EXPECT_EQ(xpool::utils::checked::align_up(std::size_t{257}, std::size_t{256}), 512U);
  EXPECT_EQ(xpool::utils::checked::align_up(std::uint32_t{7}, std::uint32_t{6}), 12U);
  EXPECT_THROW((void)xpool::utils::checked::align_up(std::size_t{1}, std::size_t{0}), c10::Error);
  EXPECT_THROW((void)xpool::utils::checked::align_up(std::numeric_limits<std::size_t>::max(), std::size_t{2}),
               c10::Error);
}
