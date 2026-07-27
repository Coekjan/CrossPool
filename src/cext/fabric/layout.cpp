#include <c10/util/Exception.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/debug/options.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/fabric/trace.hpp>
#include <xpool/utils/checked.hpp>
#include <xpool/utils/layout.hpp>

namespace xpool::fabric {

namespace {

struct FabricArenaRegions {
  xpool::utils::layout::LayoutRegion layout;
  xpool::utils::layout::LayoutRegion state;
  xpool::utils::layout::LayoutRegion model_layouts;
  xpool::utils::layout::LayoutRegion layer_layouts;
  xpool::utils::layout::LayoutRegion scheduler;
  xpool::utils::layout::LayoutRegion scheduler_entries;
  xpool::utils::layout::LayoutRegion submission_publications;
  xpool::utils::layout::LayoutRegion admission_publications;
  xpool::utils::layout::LayoutRegion invocation_publications;
  xpool::utils::layout::LayoutRegion input_ready_publications;
  xpool::utils::layout::LayoutRegion ffnagent_completion_publications;
  xpool::utils::layout::LayoutRegion result_publications;
  xpool::utils::layout::LayoutRegion acknowledgement_publications;
  xpool::utils::layout::LayoutRegion model_input_payloads;
  xpool::utils::layout::LayoutRegion model_output_payloads;
  xpool::utils::layout::LayoutRegion executor_input_payloads;
  xpool::utils::layout::LayoutRegion executor_output_payloads;
  xpool::utils::layout::LayoutRegion trace_records;
  std::size_t total_bytes;

  FabricArenaRegions(std::size_t atnagent_count, std::size_t ffnagent_count, std::size_t model_count,
                     std::size_t layer_count, std::size_t executor_count, std::size_t model_payloads_bytes,
                     std::size_t executor_payload_capacity_bytes, std::size_t trace_capacity) {
    const auto model_publication_count = xpool::utils::checked::prod(atnagent_count, model_count);
    const auto completion_count = xpool::utils::checked::prod(ffnagent_count, executor_count);
    const auto executor_payloads_bytes =
        xpool::utils::checked::prod(executor_count, executor_payload_capacity_bytes);
    using xpool::utils::layout::LayoutRegionSpec;
    const auto specs = std::to_array<LayoutRegionSpec>({
        LayoutRegionSpec::object<FabricArenaLayout>("Fabric arena layout"),
        LayoutRegionSpec::object<FabricArenaState>("Fabric arena state"),
        LayoutRegionSpec::array<FabricModelLayout>("Fabric model layouts", model_count),
        LayoutRegionSpec::array<FabricLayerLayout>("Fabric layer layouts", layer_count),
        LayoutRegionSpec::object<FfnScheduler>("FFN Scheduler"),
        LayoutRegionSpec::array<FfnSchedulerEntry>("FFN Scheduler entries", model_count),
        LayoutRegionSpec::array<FabricPublication<FfnSubmission>>("Submission publications", model_publication_count),
        LayoutRegionSpec::array<FabricPublication<FfnExecutionAdmission>>("Admission publications",
                                                                          model_publication_count),
        LayoutRegionSpec::array<FabricPublication<FfnInvocation>>("Invocation publications", executor_count),
        LayoutRegionSpec::array<FabricPublication<FfnInputReady>>("InputReady publications", executor_count),
        LayoutRegionSpec::array<FabricPublication<FfnAgentCompletion>>("FfnAgent Completion publications",
                                                                       completion_count),
        LayoutRegionSpec::array<FabricPublication<FfnResult>>("Result publications", model_publication_count),
        LayoutRegionSpec::array<FabricPublication<FfnResultAcknowledgement>>("Acknowledgement publications",
                                                                             model_publication_count),
        LayoutRegionSpec::bytes("model input payloads", model_payloads_bytes,
                                xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::bytes("model output payloads", model_payloads_bytes,
                                xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::bytes("Executor input payloads", executor_payloads_bytes,
                                xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::bytes("Executor output payloads", executor_payloads_bytes,
                                xpool::arena::kPayloadAlignment),
        LayoutRegionSpec::array<FabricTraceRecord>("Fabric trace records", trace_capacity),
    });
    const xpool::utils::layout::LayoutPlan plan{
        specs, xpool::arena::kAllocationAlignment};
    auto index = std::size_t{0};
    layout = plan[index++];
    state = plan[index++];
    model_layouts = plan[index++];
    layer_layouts = plan[index++];
    scheduler = plan[index++];
    scheduler_entries = plan[index++];
    submission_publications = plan[index++];
    admission_publications = plan[index++];
    invocation_publications = plan[index++];
    input_ready_publications = plan[index++];
    ffnagent_completion_publications = plan[index++];
    result_publications = plan[index++];
    acknowledgement_publications = plan[index++];
    model_input_payloads = plan[index++];
    model_output_payloads = plan[index++];
    executor_input_payloads = plan[index++];
    executor_output_payloads = plan[index++];
    trace_records = plan[index++];
    total_bytes = plan.total_bytes;
    TORCH_CHECK(index == plan.regions.size(), "xpool Fabric arena regions did not consume every entry");
  }
};

} // namespace

void FabricModelLayout::validate() const {
  TORCH_CHECK(xpool::abi::TensorDType::is_valid(dtype), "xpool Fabric model layout has an invalid dtype");
  TORCH_CHECK(hidden_size != 0 && atn_tp_size != 0 && atn_dp_size != 0 && layer_count != 0,
              "xpool Fabric model layout has zero topology geometry");
  xpool::utils::checked::prod(atn_tp_size, atn_dp_size);
  TORCH_CHECK(decode_payload_offset % xpool::arena::kPayloadAlignment == 0 &&
                  decode_payload_capacity_bytes != 0 &&
                  decode_payload_capacity_bytes % xpool::arena::kPayloadAlignment == 0 &&
                  prefill_payload_capacity_bytes != 0 &&
                  prefill_payload_capacity_bytes % xpool::arena::kPayloadAlignment == 0,
              "xpool Fabric model layout has invalid payload alignment or capacity");
  xpool::utils::checked::sum(layer_begin, layer_count);
  xpool::utils::checked::sum(decode_payload_offset, decode_payload_capacity_bytes);
}

void FabricLayerLayout::validate() const {
  TORCH_CHECK(FfnLayerKind::is_valid(kind), "xpool Fabric layer layout has an invalid layer kind");
}

FabricArenaLayout FabricArenaLayout::create(std::size_t atnagent_count, std::size_t ffnagent_count,
                                            std::size_t executor_count, std::size_t model_count,
                                            std::size_t layer_count, std::size_t model_payloads_bytes,
                                            std::size_t executor_payload_capacity_bytes) {
  TORCH_CHECK(atnagent_count != 0 && ffnagent_count != 0 && executor_count != 0 && model_count != 0 &&
                  layer_count != 0,
              "xpool Fabric arena requires positive topology and table counts");
  TORCH_CHECK(xpool::utils::checked::sum(atnagent_count, ffnagent_count) <= static_cast<std::size_t>(INT_MAX),
              "xpool Fabric participant count exceeds the NVSHMEM PE domain");
  TORCH_CHECK(model_payloads_bytes != 0 &&
                  model_payloads_bytes % xpool::arena::kPayloadAlignment == 0 &&
                  executor_payload_capacity_bytes != 0 &&
                  executor_payload_capacity_bytes % xpool::arena::kPayloadAlignment == 0,
              "xpool Fabric arena requires positive aligned payload capacities");
  const auto trace_capacity = xpool::debug::options().fabric_observer.capacity();
  const FabricArenaRegions regions{atnagent_count, ffnagent_count, model_count, layer_count, executor_count,
                                   model_payloads_bytes, executor_payload_capacity_bytes, trace_capacity};
  return FabricArenaLayout{
      .header = {.magic = kFabricArenaMagic,
                 .abi_version = xpool::abi::kAbiVersion,
                 .layout_size = sizeof(FabricArenaLayout),
                 .total_bytes = regions.total_bytes,
                 .state_offset = regions.state.offset},
      .atnagent_count = atnagent_count,
      .ffnagent_count = ffnagent_count,
      .model_count = model_count,
      .layer_count = layer_count,
      .executor_count = executor_count,
      .model_layouts_offset = regions.model_layouts.offset,
      .layer_layouts_offset = regions.layer_layouts.offset,
      .scheduler_offset = regions.scheduler.offset,
      .scheduler_entries_offset = regions.scheduler_entries.offset,
      .submission_publications_offset = regions.submission_publications.offset,
      .admission_publications_offset = regions.admission_publications.offset,
      .invocation_publications_offset = regions.invocation_publications.offset,
      .input_ready_publications_offset = regions.input_ready_publications.offset,
      .ffnagent_completion_publications_offset = regions.ffnagent_completion_publications.offset,
      .result_publications_offset = regions.result_publications.offset,
      .acknowledgement_publications_offset = regions.acknowledgement_publications.offset,
      .model_input_payloads_offset = regions.model_input_payloads.offset,
      .model_output_payloads_offset = regions.model_output_payloads.offset,
      .executor_input_payloads_offset = regions.executor_input_payloads.offset,
      .executor_output_payloads_offset = regions.executor_output_payloads.offset,
      .executor_payload_capacity_bytes = executor_payload_capacity_bytes,
      .trace = {.records_offset = trace_capacity == 0 ? 0 : regions.trace_records.offset,
                .capacity = trace_capacity},
  };
}

void FabricArenaLayout::validate() const {
  TORCH_CHECK(header.magic == kFabricArenaMagic, "xpool Fabric arena header magic does not match");
  TORCH_CHECK(header.abi_version == xpool::abi::kAbiVersion &&
                  header.layout_size == sizeof(FabricArenaLayout),
              "xpool Fabric arena layout has an incompatible ABI");
  TORCH_CHECK(atnagent_count != 0 && ffnagent_count != 0 && executor_count != 0 && model_count != 0 &&
                  layer_count != 0,
              "xpool Fabric arena layout has zero topology geometry");
  TORCH_CHECK(xpool::utils::checked::sum(atnagent_count, ffnagent_count) <= static_cast<std::size_t>(INT_MAX),
              "xpool Fabric participant count exceeds the NVSHMEM PE domain");
  TORCH_CHECK(model_input_payloads_offset < model_output_payloads_offset &&
                  model_output_payloads_offset < executor_input_payloads_offset,
              "xpool Fabric payload regions are not in canonical order");
  const auto model_payloads_bytes = model_output_payloads_offset - model_input_payloads_offset;
  TORCH_CHECK(model_payloads_bytes != 0 &&
                  model_payloads_bytes % xpool::arena::kPayloadAlignment == 0 &&
                  executor_payload_capacity_bytes != 0 &&
                  executor_payload_capacity_bytes % xpool::arena::kPayloadAlignment == 0,
              "xpool Fabric arena layout has invalid payload capacities");
  const FabricArenaRegions regions{atnagent_count, ffnagent_count, model_count, layer_count, executor_count,
                                   model_payloads_bytes, executor_payload_capacity_bytes, trace.capacity};
  const auto expected_trace_offset = trace.capacity == 0 ? std::size_t{0} : regions.trace_records.offset;
  TORCH_CHECK(header.total_bytes == regions.total_bytes && header.state_offset == regions.state.offset &&
                  model_layouts_offset == regions.model_layouts.offset &&
                  layer_layouts_offset == regions.layer_layouts.offset &&
                  scheduler_offset == regions.scheduler.offset &&
                  scheduler_entries_offset == regions.scheduler_entries.offset &&
                  submission_publications_offset == regions.submission_publications.offset &&
                  admission_publications_offset == regions.admission_publications.offset &&
                  invocation_publications_offset == regions.invocation_publications.offset &&
                  input_ready_publications_offset == regions.input_ready_publications.offset &&
                  ffnagent_completion_publications_offset == regions.ffnagent_completion_publications.offset &&
                  result_publications_offset == regions.result_publications.offset &&
                  acknowledgement_publications_offset == regions.acknowledgement_publications.offset &&
                  model_input_payloads_offset == regions.model_input_payloads.offset &&
                  model_output_payloads_offset == regions.model_output_payloads.offset &&
                  executor_input_payloads_offset == regions.executor_input_payloads.offset &&
                  executor_output_payloads_offset == regions.executor_output_payloads.offset &&
                  trace.records_offset == expected_trace_offset,
              "xpool Fabric arena layout does not match canonical geometry");
}

} // namespace xpool::fabric
