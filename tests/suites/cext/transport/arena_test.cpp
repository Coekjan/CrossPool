#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>
#include <xpool/transport/arena.hpp>
#include <xpool/transport/layout.hpp>
#include <xpool/transport/protocol.hpp>
#include <xpool/utils/checked.hpp>

TEST(TransportArenaHandleTest, EncodesDecodesAndIndexesOrderedContainers) {
  const auto text = std::string(sizeof(cudaIpcMemHandle_t) * 2, '1');
  const auto handle = xpool::transport::ArenaHandle::decode(text);
  auto values = std::map<xpool::transport::ArenaHandle, int>{{handle, 7}};

  EXPECT_EQ(handle.encode(), text);
  EXPECT_EQ(values.at(xpool::transport::ArenaHandle::decode(text)), 7);
  EXPECT_THROW(xpool::transport::ArenaHandle::decode("ab"), c10::Error);
}

TEST(TransportArenaTest, DefaultOwnerIsEmpty) {
  xpool::transport::Arena arena;

  EXPECT_FALSE(static_cast<bool>(arena));
  EXPECT_NO_THROW(arena.destroy());
}

TEST(TransportArenaLayoutTest, PlansSingularMailboxRegions) {
  const auto layout = xpool::transport::ArenaLayout::create(3, 2, 1, 4, 0, 1, 8, 4, c10::ScalarType::BFloat16);

  EXPECT_EQ(layout.header.magic, xpool::transport::kTransportArenaMagic);
  EXPECT_EQ(layout.header.abi_version, xpool::abi::kVersion);
  EXPECT_EQ(layout.header.layout_size, sizeof(xpool::transport::ArenaLayout));
  EXPECT_EQ(layout.instance_index, 3U);
  EXPECT_EQ(layout.instance_rank, 2U);
  EXPECT_EQ(layout.atn_tp_rank, 1U);
  EXPECT_EQ(layout.atn_tp_size, 4U);
  EXPECT_EQ(layout.atn_dp_rank, 0U);
  EXPECT_EQ(layout.atn_dp_size, 1U);
  EXPECT_EQ(layout.payload_dtype, c10::ScalarType::BFloat16);
  EXPECT_EQ(layout.payload_row_bytes, 8U);
  EXPECT_EQ(layout.header.state_offset % alignof(xpool::transport::ArenaState), 0U);
  EXPECT_EQ(layout.header.state_offset,
            xpool::utils::checked::align_up(std::size_t{sizeof(xpool::transport::ArenaLayout)},
                                            std::size_t{alignof(xpool::transport::ArenaState)}));
  EXPECT_EQ(layout.mailbox_offset % alignof(xpool::transport::Mailbox), 0U);
  EXPECT_EQ(layout.input_payload_offset % xpool::arena::kPayloadAlignment, 0U);
  EXPECT_GT(layout.mailbox_offset, layout.header.state_offset);
  EXPECT_GT(layout.input_payload_offset, layout.mailbox_offset);
  EXPECT_GT(layout.output_payload_offset, layout.input_payload_offset);
  EXPECT_EQ(layout.dp_rank_payload_rows_offset, 0U);
  EXPECT_NO_THROW(layout.validate());
}

TEST(TransportArenaLayoutTest, AllocatesDpRankPayloadRowsOnlyForDpWorld) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 1, 2, 8, 4, c10::ScalarType::Half);

  EXPECT_GT(layout.dp_rank_payload_rows_offset, layout.output_payload_offset);
  EXPECT_NO_THROW(layout.validate());
}

TEST(TransportArenaLayoutTest, RejectsNonCanonicalGeometry) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 8, 4, c10::ScalarType::BFloat16);

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

TEST(TransportArenaLayoutTest, PreservesGeometryBeyondUint32) {
  const auto beyond_uint32 = std::size_t{std::numeric_limits<std::uint32_t>::max()} + 1U;
  const auto layout = xpool::transport::ArenaLayout::create(beyond_uint32, beyond_uint32, 0, 1, 0, 1, beyond_uint32, 2,
                                                            c10::ScalarType::Half);

  EXPECT_EQ(layout.instance_index, beyond_uint32);
  EXPECT_EQ(layout.instance_rank, beyond_uint32);
  EXPECT_EQ(layout.payload_row_capacity, beyond_uint32);
  EXPECT_NO_THROW(layout.validate());
}

TEST(TransportArenaLayoutTest, AcceptsOddHiddenGeometry) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 8, 3, c10::ScalarType::Half);

  EXPECT_NO_THROW(layout.validate());
}
