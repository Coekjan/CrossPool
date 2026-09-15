#include <xpool/kv/layout.hpp>

#include <array>
#include <cstddef>

#include <c10/util/Exception.h>

#include <xpool/abi.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::kv {

namespace {

struct CapacityChannelRegions {
  xpool::utils::layout::LayoutRegion header;
  xpool::utils::layout::LayoutRegion pools;
  xpool::utils::layout::LayoutRegion groups;
  xpool::utils::layout::LayoutRegion partitions;
  std::size_t total_bytes;

  CapacityChannelRegions(std::size_t pool_count, std::size_t group_count, std::size_t partition_count) {
    using xpool::utils::layout::LayoutRegionSpec;
    const auto plan = xpool::utils::layout::LayoutPlan<4>{
        std::array{
            LayoutRegionSpec::object<CapacityChannelHeader>("kv capacity channel header"),
            LayoutRegionSpec::array<PoolEntry>("kv capacity pool entries", pool_count),
            LayoutRegionSpec::array<GroupEntry>("kv capacity group entries", group_count),
            LayoutRegionSpec::array<PartitionEntry>("kv capacity partition entries", partition_count),
        },
        alignof(std::uint64_t)};
    auto index = std::size_t{0};
    header = plan[index++];
    pools = plan[index++];
    groups = plan[index++];
    partitions = plan[index++];
    TORCH_CHECK(index == plan.regions.size(), "xpool kv capacity channel region plan is incomplete");
    total_bytes = plan.total_bytes;
  }
};

} // namespace

CapacityChannelLayout CapacityChannelLayout::create(std::size_t pool_count, std::size_t group_count,
                                                    std::size_t partition_count) {
  TORCH_CHECK(pool_count != 0 && group_count != 0 && partition_count != 0, "xpool kv channel counts must be positive");
  const auto regions = CapacityChannelRegions{pool_count, group_count, partition_count};
  TORCH_CHECK(regions.header.offset == 0, "xpool kv capacity channel header must start at offset zero");
  return {
      .pool_count = pool_count,
      .group_count = group_count,
      .partition_count = partition_count,
      .pools_offset = regions.pools.offset,
      .groups_offset = regions.groups.offset,
      .partitions_offset = regions.partitions.offset,
      .total_bytes = regions.total_bytes,
  };
}

void CapacityChannelLayout::validate(const CapacityChannelHeader &header, std::size_t mapping_bytes) const {
  TORCH_CHECK(header.magic == kCapacityChannelMagic, "xpool kv channel magic does not match");
  TORCH_CHECK(header.abi_version == xpool::abi::kVersion && header.reserved == 0,
              "xpool kv channel ABI version does not match");
  TORCH_CHECK(header.pool_count == pool_count && header.group_count == group_count &&
                  header.partition_count == partition_count && header.total_bytes == total_bytes &&
                  mapping_bytes == total_bytes,
              "xpool kv channel header does not match its canonical layout");
}

} // namespace xpool::kv
