#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/utils/checked.hpp>

TEST(FabricArenaTest, EmptyOwnerSupportsMoveAndIdempotentDestroy) {
  auto source = xpool::fabric::Arena{};
  auto destination = xpool::fabric::Arena{std::move(source)};

  EXPECT_FALSE(static_cast<bool>(source));
  EXPECT_FALSE(static_cast<bool>(destination));
  EXPECT_NO_THROW(destination.destroy());

  auto replacement = xpool::fabric::Arena{};
  EXPECT_NO_THROW(replacement = std::move(destination));
  EXPECT_FALSE(static_cast<bool>(replacement));
}

TEST(FabricArenaLayoutTest, PlansAlignedContiguousLanePayloadStorageWhenRoutingExists) {
  const auto layout = xpool::fabric::ArenaLayout::create(2, 3, 2, 2, 3, 3, 6, 300, 32);

  EXPECT_EQ(layout.header.magic, xpool::fabric::kFabricArenaMagic);
  EXPECT_EQ(layout.header.abi_version, xpool::abi::kVersion);
  EXPECT_EQ(layout.header.layout_size, sizeof(xpool::fabric::ArenaLayout));
  EXPECT_EQ(layout.atnagent_count, 2U);
  EXPECT_EQ(layout.ffnagent_count, 3U);
  EXPECT_EQ(layout.coordinator_pe(), 2);
  EXPECT_EQ(layout.instance_count, 2U);
  EXPECT_EQ(layout.layer_entry_count, 3U);
  EXPECT_EQ(layout.executor_lane_count, 2U);

  const std::array offsets{
      layout.header.state_offset,
      layout.instance_entries_offset_bytes,
      layout.layer_entries_offset_bytes,
      layout.atnagent_pes_offset_bytes,
      layout.ffnagent_pes_offset_bytes,
      layout.submission_publications_offset_bytes,
      layout.admission_publications_offset_bytes,
      layout.lane_execution_publications_offset_bytes,
      layout.input_ready_publications_offset_bytes,
      layout.routing_metadata_ready_publications_offset_bytes,
      layout.partial_ready_publications_offset_bytes,
      layout.ffnagent_completion_publications_offset_bytes,
      layout.output_commit_publications_offset_bytes,
      layout.output_acknowledgement_publications_offset_bytes,
      layout.lane_payload_storage_offset_bytes,
      layout.routing_metadata_offset_bytes,
  };
  EXPECT_TRUE(std::ranges::is_sorted(offsets));
  EXPECT_EQ(layout.header.state_offset % alignof(xpool::fabric::ArenaState), 0U);
  EXPECT_EQ(layout.header.state_offset,
            xpool::utils::checked::align_up(sizeof(xpool::fabric::ArenaLayout), alignof(xpool::fabric::ArenaState)));
  EXPECT_EQ(layout.instance_entries_offset_bytes % alignof(xpool::fabric::InstanceEntry), 0U);
  EXPECT_EQ(layout.layer_entries_offset_bytes % alignof(xpool::fabric::LayerEntry), 0U);
  EXPECT_EQ(layout.submission_publications_offset_bytes % xpool::fabric::kFabricPublicationAlignment, 0U);
  EXPECT_EQ(layout.lane_payload_capacity_bytes,
            xpool::utils::checked::align_up(std::size_t{300}, xpool::arena::kPayloadAlignment));
  EXPECT_EQ(layout.lane_payload_storage_offset_bytes % xpool::arena::kPayloadAlignment, 0U);
  EXPECT_EQ(layout.routing_metadata_offset_bytes % xpool::arena::kPayloadAlignment, 0U);
  EXPECT_GE(layout.routing_metadata_offset_bytes,
            layout.lane_payload_storage_offset_bytes + xpool::fabric::kExecutorLanePayloadBufferCount *
                                                           layout.executor_lane_count *
                                                           layout.lane_payload_capacity_bytes);
  EXPECT_GE(layout.header.total_bytes,
            layout.routing_metadata_offset_bytes + layout.executor_lane_count * layout.routing_metadata_stride_bytes);
  EXPECT_NO_THROW(layout.validate());
  EXPECT_EQ(xpool::fabric::arena_allocation_bytes(2, 3, 2, 2, 3, 3, 6, 300, 32), layout.header.total_bytes);
}

TEST(FabricArenaLayoutTest, OmitsBothRoutingRegionsForDenseOnlyGeometry) {
  const auto layout = xpool::fabric::ArenaLayout::create(1, 1, 1, 1, 1, 1, 1, 255, 0);

  EXPECT_EQ(layout.routing_metadata_ready_publications_offset_bytes, 0U);
  EXPECT_EQ(layout.routing_metadata_offset_bytes, 0U);
  EXPECT_EQ(layout.routing_metadata_stride_bytes, 0U);
  EXPECT_EQ(layout.lane_payload_capacity_bytes, 256U);
  EXPECT_NO_THROW(layout.validate());
}

TEST(FabricArenaLayoutTest, RejectsInvalidTopologyAndCapacities) {
  EXPECT_THROW(xpool::fabric::ArenaLayout::create(0, 1, 1, 1, 1, 1, 1, 256, 0), c10::Error);
  EXPECT_THROW(xpool::fabric::ArenaLayout::create(1, 1, 0, 1, 1, 1, 1, 256, 0), c10::Error);
  EXPECT_THROW(xpool::fabric::ArenaLayout::create(1, 1, 1, 0, 1, 1, 1, 256, 0), c10::Error);
  EXPECT_THROW(xpool::fabric::ArenaLayout::create(1, 1, 1, 1, 1, 1, 1, 0, 0), c10::Error);
}

TEST(FabricArenaLayoutTest, RejectsNonCanonicalGeometry) {
  const auto layout = xpool::fabric::ArenaLayout::create(1, 1, 1, 1, 1, 1, 1, 256, 0);

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
}

TEST(FabricArenaLayoutTest, PreservesCountsBeyondUint32) {
  const auto beyond_uint32 = std::size_t{std::numeric_limits<std::uint32_t>::max()} + 1U;
  const auto layout = xpool::fabric::ArenaLayout::create(1, 1, beyond_uint32, 1, 1, 1, 1, 256, 0);

  EXPECT_EQ(layout.instance_count, beyond_uint32);
  EXPECT_NO_THROW(layout.validate());
}
