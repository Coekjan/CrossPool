#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <xpool/utils/layout.hpp>

TEST(LayoutPlanTest, AlignsEachRegionByItsOwnRequirement) {
  using xpool::utils::layout::LayoutPlan;
  using xpool::utils::layout::LayoutRegionSpec;

  const auto specs = std::to_array<LayoutRegionSpec>({
      LayoutRegionSpec::bytes("first", 1),
      LayoutRegionSpec::array<std::uint32_t>("second", 2),
  });
  const LayoutPlan plan{specs, std::size_t{256}};

  EXPECT_EQ(plan[0].offset, 0U);
  EXPECT_EQ(plan[0].bytes, 1U);
  EXPECT_EQ(plan[1].offset, alignof(std::uint32_t));
  EXPECT_EQ(plan[1].bytes, 2U * sizeof(std::uint32_t));
  EXPECT_EQ(plan[1].alignment, alignof(std::uint32_t));
  EXPECT_EQ(plan.total_bytes, 256U);
}

TEST(LayoutPlanTest, RejectsArrayAndAlignmentOverflow) {
  using xpool::utils::layout::LayoutPlan;
  using xpool::utils::layout::LayoutRegionSpec;

  const auto overflowing_count = std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) + 1;
  EXPECT_THROW((void)LayoutRegionSpec::array<std::uint64_t>("overflowing array", overflowing_count), c10::Error);

  const std::array specs{LayoutRegionSpec::bytes("overflowing alignment", std::numeric_limits<std::size_t>::max())};
  EXPECT_THROW(([&] {
                 const LayoutPlan plan(specs, std::size_t{256});
                 (void)plan;
               }()),
               c10::Error);
}
