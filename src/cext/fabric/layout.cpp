#include <c10/util/Exception.h>

#include <array>
#include <climits>
#include <cstddef>

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::fabric {

namespace {

struct FabricArenaRegions {
  xpool::utils::layout::LayoutRegion layout;
  xpool::utils::layout::LayoutRegion state;
  xpool::utils::layout::LayoutRegion instance_entries;
  xpool::utils::layout::LayoutRegion layer_entries;
  xpool::utils::layout::LayoutRegion atnagent_pes;
  xpool::utils::layout::LayoutRegion ffnagent_pes;
  xpool::utils::layout::LayoutRegion submission_publications;
  xpool::utils::layout::LayoutRegion admission_publications;
  xpool::utils::layout::LayoutRegion lane_execution_publications;
  xpool::utils::layout::LayoutRegion input_ready_publications;
  xpool::utils::layout::LayoutRegion routing_metadata_ready_publications;
  xpool::utils::layout::LayoutRegion partial_ready_publications;
  xpool::utils::layout::LayoutRegion ffnagent_completion_publications;
  xpool::utils::layout::LayoutRegion output_commit_publications;
  xpool::utils::layout::LayoutRegion output_acknowledgement_publications;
  xpool::utils::layout::LayoutRegion lane_payload_storage;
  xpool::utils::layout::LayoutRegion routing_metadata;
  std::size_t total_bytes;

  FabricArenaRegions(std::size_t atnagent_count, std::size_t ffnagent_count, std::size_t instance_count,
                     std::size_t executor_lane_count, std::size_t layer_entry_count,
                     std::size_t atnagent_pe_entry_count, std::size_t ffnagent_pe_entry_count,
                     std::size_t lane_payload_capacity_bytes, std::size_t routing_metadata_stride_bytes) {
    const auto submission_count = xpool::utils::checked::prod(atnagent_count, instance_count);
    const auto ffnagent_lane_count = xpool::utils::checked::prod(ffnagent_count, executor_lane_count);
    const auto lane_payload_bytes =
        xpool::utils::checked::prod(kExecutorLanePayloadBufferCount, executor_lane_count, lane_payload_capacity_bytes);
    const auto routing_lane_count = routing_metadata_stride_bytes == 0 ? std::size_t{0} : executor_lane_count;
    const auto routing_publication_bytes =
        xpool::utils::checked::prod(routing_lane_count, sizeof(Publication<RoutingMetadataReady>));
    const auto routing_metadata_bytes = xpool::utils::checked::prod(routing_lane_count, routing_metadata_stride_bytes);
    using xpool::utils::layout::LayoutRegionSpec;
    const auto specs = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<ArenaLayout>("Fabric arena layout"),
        LayoutRegionSpec::object<ArenaState>("Fabric arena state"),
        LayoutRegionSpec::array<InstanceEntry>("Fabric Instance entries", instance_count),
        LayoutRegionSpec::array<LayerEntry>("Fabric layer entries", layer_entry_count),
        LayoutRegionSpec::array<int>("AtnAgent PE entries", atnagent_pe_entry_count),
        LayoutRegionSpec::array<int>("FfnAgent PE entries", ffnagent_pe_entry_count),
        LayoutRegionSpec::array<Publication<Submission>>("Submission publications", submission_count),
        LayoutRegionSpec::array<Publication<Admission>>("Admission publications", instance_count),
        LayoutRegionSpec::array<Publication<LaneExecution>>("Lane Execution publications",
                                                                     executor_lane_count),
        LayoutRegionSpec::array<Publication<InputReady>>("Input Ready publications", executor_lane_count),
        LayoutRegionSpec::bytes("Routing Metadata Ready publications", routing_publication_bytes,
                                routing_lane_count == 0 ? 1 : alignof(Publication<RoutingMetadataReady>)),
        LayoutRegionSpec::array<Publication<PartialReady>>("Partial Ready publications", ffnagent_lane_count),
        LayoutRegionSpec::array<Publication<FfnAgentCompletion>>("FfnAgent Completion publications",
                                                                       ffnagent_lane_count),
        LayoutRegionSpec::array<Publication<OutputCommit>>("Output Commit publications", instance_count),
        LayoutRegionSpec::array<Publication<OutputAcknowledgement>>("Output Acknowledgement publications",
                                                                             submission_count),
        LayoutRegionSpec::bytes("Lane payload storage", lane_payload_bytes, xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::bytes("Routing Metadata", routing_metadata_bytes,
                                routing_lane_count == 0 ? 1 : xpool::arena::kPayloadAlignment),
    });
    const auto plan = xpool::utils::layout::LayoutPlan{specs, xpool::arena::kAllocationAlignment};
    auto index = std::size_t{0};
    layout = plan[index++];
    state = plan[index++];
    instance_entries = plan[index++];
    layer_entries = plan[index++];
    atnagent_pes = plan[index++];
    ffnagent_pes = plan[index++];
    submission_publications = plan[index++];
    admission_publications = plan[index++];
    lane_execution_publications = plan[index++];
    input_ready_publications = plan[index++];
    routing_metadata_ready_publications = plan[index++];
    partial_ready_publications = plan[index++];
    ffnagent_completion_publications = plan[index++];
    output_commit_publications = plan[index++];
    output_acknowledgement_publications = plan[index++];
    lane_payload_storage = plan[index++];
    routing_metadata = plan[index++];
    TORCH_CHECK(index == plan.regions.size(), "xpool Fabric arena region plan is incomplete");
    total_bytes = plan.total_bytes;
  }
};

} // namespace

ArenaLayout ArenaLayout::create(std::size_t atnagent_count, std::size_t ffnagent_count,
                                            std::size_t instance_count, std::size_t executor_lane_count,
                                            std::size_t layer_entry_count, std::size_t atnagent_pe_entry_count,
                                            std::size_t ffnagent_pe_entry_count, std::size_t maximum_lane_payload_bytes,
                                            std::size_t maximum_routing_metadata_elements) {
  TORCH_CHECK(atnagent_count != 0 && ffnagent_count != 0 && instance_count != 0 && executor_lane_count != 0 &&
                  layer_entry_count != 0 && atnagent_pe_entry_count != 0 && ffnagent_pe_entry_count != 0,
              "xpool Fabric arena requires positive topology and table counts");
  TORCH_CHECK(xpool::utils::checked::sum(atnagent_count, ffnagent_count) <= static_cast<std::size_t>(INT_MAX),
              "xpool Fabric participant count exceeds the NVSHMEM PE domain");
  TORCH_CHECK(maximum_lane_payload_bytes != 0, "xpool Fabric arena requires a positive lane payload capacity");
  const auto lane_payload_capacity_bytes =
      xpool::utils::checked::align_up(maximum_lane_payload_bytes, xpool::arena::kPayloadAlignment);
  const auto routing_metadata_bytes =
      xpool::utils::checked::prod(maximum_routing_metadata_elements, std::size_t{sizeof(std::int32_t) + sizeof(float)});
  const auto routing_metadata_stride_bytes =
      routing_metadata_bytes == 0
          ? std::size_t{0}
          : xpool::utils::checked::align_up(routing_metadata_bytes, xpool::arena::kPayloadAlignment);
  const auto regions = FabricArenaRegions{
      atnagent_count,
      ffnagent_count,
      instance_count,
      executor_lane_count,
      layer_entry_count,
      atnagent_pe_entry_count,
      ffnagent_pe_entry_count,
      lane_payload_capacity_bytes,
      routing_metadata_stride_bytes,
  };
  return ArenaLayout{
      .header = {.magic = kFabricArenaMagic,
                 .abi_version = xpool::abi::kVersion,
                 .layout_size = sizeof(ArenaLayout),
                 .total_bytes = regions.total_bytes,
                 .state_offset = regions.state.offset},
      .atnagent_count = atnagent_count,
      .ffnagent_count = ffnagent_count,
      .instance_count = instance_count,
      .executor_lane_count = executor_lane_count,
      .layer_entry_count = layer_entry_count,
      .atnagent_pe_entry_count = atnagent_pe_entry_count,
      .ffnagent_pe_entry_count = ffnagent_pe_entry_count,
      .instance_entries_offset_bytes = regions.instance_entries.offset,
      .layer_entries_offset_bytes = regions.layer_entries.offset,
      .atnagent_pes_offset_bytes = regions.atnagent_pes.offset,
      .ffnagent_pes_offset_bytes = regions.ffnagent_pes.offset,
      .submission_publications_offset_bytes = regions.submission_publications.offset,
      .admission_publications_offset_bytes = regions.admission_publications.offset,
      .lane_execution_publications_offset_bytes = regions.lane_execution_publications.offset,
      .input_ready_publications_offset_bytes = regions.input_ready_publications.offset,
      .routing_metadata_ready_publications_offset_bytes =
          routing_metadata_stride_bytes == 0 ? 0 : regions.routing_metadata_ready_publications.offset,
      .partial_ready_publications_offset_bytes = regions.partial_ready_publications.offset,
      .ffnagent_completion_publications_offset_bytes = regions.ffnagent_completion_publications.offset,
      .output_commit_publications_offset_bytes = regions.output_commit_publications.offset,
      .output_acknowledgement_publications_offset_bytes = regions.output_acknowledgement_publications.offset,
      .lane_payload_storage_offset_bytes = regions.lane_payload_storage.offset,
      .lane_payload_capacity_bytes = lane_payload_capacity_bytes,
      .routing_metadata_offset_bytes = routing_metadata_stride_bytes == 0 ? 0 : regions.routing_metadata.offset,
      .routing_metadata_stride_bytes = routing_metadata_stride_bytes,
  };
}

void ArenaLayout::validate() const {
  TORCH_CHECK(header.magic == kFabricArenaMagic, "xpool Fabric arena header magic does not match");
  TORCH_CHECK(header.abi_version == xpool::abi::kVersion && header.layout_size == sizeof(ArenaLayout),
              "xpool Fabric arena layout has an incompatible ABI");
  TORCH_CHECK(atnagent_count != 0 && ffnagent_count != 0 && instance_count != 0 && executor_lane_count != 0 &&
                  layer_entry_count != 0 && atnagent_pe_entry_count != 0 && ffnagent_pe_entry_count != 0,
              "xpool Fabric arena layout has zero topology geometry");
  TORCH_CHECK(lane_payload_capacity_bytes != 0 && lane_payload_capacity_bytes % xpool::arena::kPayloadAlignment == 0 &&
                  routing_metadata_stride_bytes % xpool::arena::kPayloadAlignment == 0,
              "xpool Fabric arena layout has invalid lane capacities");
  const auto regions = FabricArenaRegions{
      atnagent_count,
      ffnagent_count,
      instance_count,
      executor_lane_count,
      layer_entry_count,
      atnagent_pe_entry_count,
      ffnagent_pe_entry_count,
      lane_payload_capacity_bytes,
      routing_metadata_stride_bytes,
  };
  const auto expected_routing_ready =
      routing_metadata_stride_bytes == 0 ? std::size_t{0} : regions.routing_metadata_ready_publications.offset;
  const auto expected_routing_metadata =
      routing_metadata_stride_bytes == 0 ? std::size_t{0} : regions.routing_metadata.offset;
  TORCH_CHECK(header.total_bytes == regions.total_bytes && header.state_offset == regions.state.offset &&
                  instance_entries_offset_bytes == regions.instance_entries.offset &&
                  layer_entries_offset_bytes == regions.layer_entries.offset &&
                  atnagent_pes_offset_bytes == regions.atnagent_pes.offset &&
                  ffnagent_pes_offset_bytes == regions.ffnagent_pes.offset &&
                  submission_publications_offset_bytes == regions.submission_publications.offset &&
                  admission_publications_offset_bytes == regions.admission_publications.offset &&
                  lane_execution_publications_offset_bytes == regions.lane_execution_publications.offset &&
                  input_ready_publications_offset_bytes == regions.input_ready_publications.offset &&
                  routing_metadata_ready_publications_offset_bytes == expected_routing_ready &&
                  partial_ready_publications_offset_bytes == regions.partial_ready_publications.offset &&
                  ffnagent_completion_publications_offset_bytes == regions.ffnagent_completion_publications.offset &&
                  output_commit_publications_offset_bytes == regions.output_commit_publications.offset &&
                  output_acknowledgement_publications_offset_bytes ==
                      regions.output_acknowledgement_publications.offset &&
                  lane_payload_storage_offset_bytes == regions.lane_payload_storage.offset &&
                  routing_metadata_offset_bytes == expected_routing_metadata,
              "xpool Fabric arena layout does not match canonical geometry");
}

std::size_t arena_allocation_bytes(std::size_t atnagent_count, std::size_t ffnagent_count, std::size_t instance_count,
                                   std::size_t executor_lane_count, std::size_t layer_entry_count,
                                   std::size_t atnagent_pe_entry_count, std::size_t ffnagent_pe_entry_count,
                                   std::size_t maximum_lane_payload_bytes,
                                   std::size_t maximum_routing_metadata_elements) {
  return ArenaLayout::create(atnagent_count, ffnagent_count, instance_count, executor_lane_count,
                                   layer_entry_count, atnagent_pe_entry_count, ffnagent_pe_entry_count,
                                   maximum_lane_payload_bytes, maximum_routing_metadata_elements)
      .header.total_bytes;
}

} // namespace xpool::fabric
