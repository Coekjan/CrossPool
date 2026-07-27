#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <xpool/utils/ranges.hpp>

TEST(RangesTest, MaterializesEmptyRange) {
  const std::vector<int> values;

  EXPECT_TRUE(xpool::utils::ranges::to_vector(
                  values | std::views::transform([](const auto value) { return std::to_string(value); }))
                  .empty());
}

TEST(RangesTest, MaterializesProjectedValuesInOrder) {
  const std::vector values{1, 2, 3};

  EXPECT_EQ(xpool::utils::ranges::to_vector(
                values | std::views::transform([](const auto value) { return std::to_string(value * 2); })),
            (std::vector<std::string>{"2", "4", "6"}));
}

TEST(RangesTest, MovesPrvaluesIntoResult) {
  const std::vector values{1, 2};

  auto pointers = xpool::utils::ranges::to_vector(
      values | std::views::transform([](const auto value) { return std::make_unique<int>(value); }));

  ASSERT_EQ(pointers.size(), 2U);
  EXPECT_EQ(*pointers[0], 1);
  EXPECT_EQ(*pointers[1], 2);
}
