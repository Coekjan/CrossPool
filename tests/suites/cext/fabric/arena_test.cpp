#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/utils/checked.hpp>

static_assert(std::same_as<decltype(xpool::fabric::FabricArenaLayout{}.model_count), std::size_t>);
static_assert(std::same_as<decltype(xpool::fabric::FabricArenaLayout{}.coordinator_pe()), int>);
static_assert(std::same_as<decltype(xpool::fabric::FabricArenaLayout{}.model_layouts_offset), std::size_t>);
static_assert(std::same_as<decltype(xpool::fabric::FfnSubmission{}.payload_rows), std::size_t>);

TEST(FabricArenaTest, EmptyOwnerSupportsMoveAndIdempotentDestroy) {
  auto source = xpool::fabric::FabricArena{};
  auto destination = xpool::fabric::FabricArena{std::move(source)};

  EXPECT_FALSE(static_cast<bool>(source));
  EXPECT_FALSE(static_cast<bool>(destination));
  EXPECT_NO_THROW(destination.destroy());

  auto replacement = xpool::fabric::FabricArena{};
  EXPECT_NO_THROW(replacement = std::move(destination));
  EXPECT_FALSE(static_cast<bool>(replacement));
}

TEST(FabricArenaLayoutTest, PlansTypedAlignedRegionsFromTopology) {
  const auto layout = xpool::fabric::FabricArenaLayout::create(1, 1, 2, 2, 2, 512, 512);

  EXPECT_EQ(layout.header.magic, xpool::fabric::kFabricArenaMagic);
  EXPECT_EQ(layout.header.abi_version, xpool::abi::kAbiVersion);
  EXPECT_EQ(layout.header.layout_size, sizeof(xpool::fabric::FabricArenaLayout));
  EXPECT_EQ(layout.atnagent_count, 1U);
  EXPECT_EQ(layout.ffnagent_count, 1U);
  EXPECT_EQ(layout.coordinator_pe(), 1);
  EXPECT_EQ(layout.model_count, 2U);
  EXPECT_EQ(layout.layer_count, 2U);
  EXPECT_EQ(layout.executor_count, 2U);
  const std::array offsets{layout.header.state_offset,
                           layout.model_layouts_offset,
                           layout.layer_layouts_offset,
                           layout.scheduler_offset,
                           layout.scheduler_entries_offset,
                           layout.submission_publications_offset,
                           layout.admission_publications_offset,
                           layout.invocation_publications_offset,
                           layout.input_ready_publications_offset,
                           layout.ffnagent_completion_publications_offset,
                           layout.result_publications_offset,
                           layout.acknowledgement_publications_offset,
                           layout.model_input_payloads_offset,
                           layout.model_output_payloads_offset,
                           layout.executor_input_payloads_offset,
                           layout.executor_output_payloads_offset};
  EXPECT_EQ(layout.header.state_offset % alignof(xpool::fabric::FabricArenaState), 0U);
  EXPECT_EQ(layout.header.state_offset,
            xpool::utils::checked::align_up(
                std::size_t{sizeof(xpool::fabric::FabricArenaLayout)},
                std::size_t{alignof(xpool::fabric::FabricArenaState)}));
  EXPECT_EQ(layout.model_layouts_offset % alignof(xpool::fabric::FabricModelLayout), 0U);
  EXPECT_EQ(layout.layer_layouts_offset % alignof(xpool::fabric::FabricLayerLayout), 0U);
  EXPECT_EQ(layout.submission_publications_offset %
                xpool::fabric::kFabricPublicationAlignment,
            0U);
  EXPECT_EQ(layout.model_input_payloads_offset %
                xpool::arena::kPayloadAlignment,
            0U);
  EXPECT_EQ(layout.executor_output_payloads_offset %
                xpool::arena::kPayloadAlignment,
            0U);
  EXPECT_TRUE(std::ranges::is_sorted(offsets));
  EXPECT_GE(layout.model_layouts_offset, layout.header.state_offset + sizeof(xpool::fabric::FabricArenaState));
  EXPECT_GE(layout.layer_layouts_offset,
            layout.model_layouts_offset + layout.model_count * sizeof(xpool::fabric::FabricModelLayout));
  EXPECT_GE(layout.header.total_bytes,
            layout.executor_output_payloads_offset +
                layout.executor_count * layout.executor_payload_capacity_bytes);
  EXPECT_EQ(layout.executor_payload_capacity_bytes, 512U);
  EXPECT_EQ(layout.trace.capacity, 0U);
  EXPECT_NO_THROW(layout.validate());
}

TEST(FabricArenaLayoutTest, RejectsInvalidTopologyAndCapacities) {
  EXPECT_THROW(xpool::fabric::FabricArenaLayout::create(0, 1, 1, 1, 1, 256, 256),
               c10::Error);
  EXPECT_THROW(xpool::fabric::FabricArenaLayout::create(1, 1, 0, 1, 1, 256, 256),
               c10::Error);
  EXPECT_THROW(xpool::fabric::FabricArenaLayout::create(1, 1, 1, 0, 1, 256, 256),
               c10::Error);
  EXPECT_THROW(xpool::fabric::FabricArenaLayout::create(1, 1, 1, 1, 1, 0, 256),
               c10::Error);
}

TEST(FabricArenaLayoutTest, RejectsNonCanonicalGeometry) {
  const auto layout = xpool::fabric::FabricArenaLayout::create(1, 1, 2, 2, 2, 512, 512);

  auto shifted_state = layout;
  ++shifted_state.header.state_offset;
  EXPECT_THROW(shifted_state.validate(), c10::Error);

  auto incompatible_version = layout;
  ++incompatible_version.header.abi_version;
  EXPECT_THROW(incompatible_version.validate(), c10::Error);

  auto incompatible_layout_size = layout;
  --incompatible_layout_size.header.layout_size;
  EXPECT_THROW(incompatible_layout_size.validate(), c10::Error);

  auto incompatible_magic = layout;
  ++incompatible_magic.header.magic;
  EXPECT_THROW(incompatible_magic.validate(), c10::Error);

  auto inconsistent_trace = layout;
  inconsistent_trace.trace.capacity = 1;
  EXPECT_THROW(inconsistent_trace.validate(), c10::Error);
}

TEST(FabricArenaLayoutTest, PreservesCountsBeyondUint32) {
  const auto beyond_uint32 = std::size_t{std::numeric_limits<std::uint32_t>::max()} + 1U;
  const auto layout =
      xpool::fabric::FabricArenaLayout::create(1, 1, 1, beyond_uint32, 1, 256, 256);

  EXPECT_EQ(layout.model_count, beyond_uint32);
  EXPECT_NO_THROW(layout.validate());
}
