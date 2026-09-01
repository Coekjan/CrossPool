#pragma once

/// \file xpool/fabric/arena.cuh
/// \brief Device-side implementations for typed Fabric arena views.

#include <cstddef>
#include <cstdint>
#include <limits>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/protocol.cuh>

namespace xpool::fabric {

template <typename T>
XPOOL_DEVICE_FN inline T *ArenaView::pointer_at(std::size_t offset, std::size_t index) const {
  return reinterpret_cast<T *>(base_ + offset) + index;
}

// One routing block stores its capacity-sized int32 ID array before its
// capacity-sized float weight array. Accessors expose only the live prefix.
XPOOL_DEVICE_FN inline cuda::std::span<std::int32_t> RoutingMetadataBlock::topk_ids_destination() const {
  return {reinterpret_cast<std::int32_t *>(base_), live_elements_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const std::int32_t> RoutingMetadataBlock::topk_ids() const {
  return {reinterpret_cast<const std::int32_t *>(base_), live_elements_};
}

XPOOL_DEVICE_FN inline cuda::std::span<float> RoutingMetadataBlock::topk_weights_destination() const {
  return {reinterpret_cast<float *>(base_ + capacity_elements_ * sizeof(std::int32_t)), live_elements_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const float> RoutingMetadataBlock::topk_weights() const {
  return {reinterpret_cast<const float *>(base_ + capacity_elements_ * sizeof(std::int32_t)), live_elements_};
}

XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> RoutingMetadataBlock::publication_payload() const {
  return {base_, capacity_elements_ * (sizeof(std::int32_t) + sizeof(float))};
}

XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> AtnAgentLanePayloadView::input_destination() const {
  return {buffers_[0], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const std::uint8_t> AtnAgentLanePayloadView::input() const {
  return {buffers_[0], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> AtnAgentLanePayloadView::output_destination() const {
  return {buffers_[1], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const std::uint8_t> AtnAgentLanePayloadView::output() const {
  return {buffers_[1], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const std::uint8_t> FfnAgentLanePayloadView::input() const {
  return {buffers_[0], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> FfnAgentLanePayloadView::partial_destination() const {
  return {buffers_[1], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const std::uint8_t> FfnAgentLanePayloadView::partial() const {
  return {buffers_[1], capacity_};
}

// Once input consumption finishes, complete delivery reuses buffer zero as
// staging; no second input interpretation remains live at that point.
XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> FfnAgentLanePayloadView::complete_output_staging_destination() const {
  return {buffers_[0], capacity_};
}

XPOOL_DEVICE_FN inline cuda::std::span<const std::uint8_t> FfnAgentLanePayloadView::complete_output_staging() const {
  return {buffers_[0], capacity_};
}

XPOOL_DEVICE_FN inline const ArenaLayout &ArenaView::layout() const {
  return *reinterpret_cast<const ArenaLayout *>(base_);
}

XPOOL_DEVICE_FN inline ArenaState &ArenaView::state() const {
  return *pointer_at<ArenaState>(layout().header.state_offset);
}

XPOOL_DEVICE_FN inline xpool::ffn::ResultCode ArenaView::cancellation_result() const {
  const auto &failure = state().failure;
  return failure.published() ? failure.payload.result_code : xpool::ffn::ResultCode::Shutdown;
}

XPOOL_DEVICE_FN inline DeliveryVariant delivery_variant(const InstanceEntry &instance,
                                                         xpool::ffn::OutputRequirement output_requirement) {
  if (output_requirement == xpool::ffn::OutputRequirement::PerRankComplete) {
    return DeliveryVariant::ReplicatedComplete;
  }
  xpool::abort_if(output_requirement != xpool::ffn::OutputRequirement::GroupSumComplete ||
                  !instance.group_sum_complete_admitted);
  const auto output_count = instance.atn_tp_size * instance.atn_dp_size;
  return instance.ffn_tp_size <= output_count ? DeliveryVariant::DirectPartial : DeliveryVariant::SingleComplete;
}

XPOOL_DEVICE_FN inline const InstanceEntry &ArenaView::instance_entry(std::size_t instance_index) const {
  xpool::abort_if(instance_index >= layout().instance_count);
  return *pointer_at<const InstanceEntry>(layout().instance_entries_offset_bytes, instance_index);
}

XPOOL_DEVICE_FN inline const LayerEntry &ArenaView::layer_entry(std::size_t instance_index,
                                                                            std::size_t layer_ordinal) const {
  const auto &instance = instance_entry(instance_index);
  xpool::abort_if(layer_ordinal >= instance.layer_count);
  return *pointer_at<const LayerEntry>(layout().layer_entries_offset_bytes, instance.layer_begin + layer_ordinal);
}

XPOOL_DEVICE_FN inline cuda::std::span<const int> ArenaView::atnagent_pes(std::size_t instance_index) const {
  const auto &instance = instance_entry(instance_index);
  const auto count = instance.atn_tp_size * instance.atn_dp_size;
  return {pointer_at<const int>(layout().atnagent_pes_offset_bytes, instance.atnagent_pe_begin), count};
}

XPOOL_DEVICE_FN inline cuda::std::span<const int> ArenaView::ffnagent_pes(std::size_t instance_index,
                                                                                std::size_t layer_ordinal) const {
  const auto &instance = instance_entry(instance_index);
  xpool::abort_if(layer_ordinal >= instance.layer_count);
  const auto begin = instance.ffnagent_pe_begin + layer_ordinal * instance.ffn_tp_size;
  return {pointer_at<const int>(layout().ffnagent_pes_offset_bytes, begin), instance.ffn_tp_size};
}

XPOOL_DEVICE_FN inline RoutingMetadataBlock
ArenaView::routing_metadata(std::size_t executor_lane_index, std::size_t payload_row_capacity) const {
  xpool::abort_if(executor_lane_index >= layout().executor_lane_count || payload_row_capacity == 0);
  const auto &execution = lane_execution_publication(executor_lane_index).record;
  xpool::abort_if(execution.validate() != xpool::ffn::ResultCode::Ok ||
                  execution.key.instance_index >= layout().instance_count ||
                  execution.payload_rows > payload_row_capacity);
  const auto &layer = layer_entry(execution.key.instance_index, execution.layer_ordinal);
  xpool::abort_if(layer.kind != xpool::ffn::LayerKind::Moe || layer.effective_topk == 0 ||
                  payload_row_capacity > std::numeric_limits<std::size_t>::max() / layer.effective_topk);
  const auto capacity_elements = payload_row_capacity * layer.effective_topk;
  const auto live_elements = execution.payload_rows * layer.effective_topk;
  xpool::abort_if(capacity_elements >
                      std::numeric_limits<std::size_t>::max() / (sizeof(std::int32_t) + sizeof(float)) ||
                  capacity_elements * (sizeof(std::int32_t) + sizeof(float)) > layout().routing_metadata_stride_bytes);
  return RoutingMetadataBlock{
      pointer_at<std::uint8_t>(layout().routing_metadata_offset_bytes,
                               executor_lane_index * layout().routing_metadata_stride_bytes),
      capacity_elements,
      live_elements,
  };
}

XPOOL_DEVICE_FN inline Publication<Submission> &
ArenaView::submission_publication(std::size_t source_atnagent_index, std::size_t instance_index) const {
  xpool::abort_if(source_atnagent_index >= layout().atnagent_count || instance_index >= layout().instance_count);
  return *pointer_at<Publication<Submission>>(
      layout().submission_publications_offset_bytes, source_atnagent_index * layout().instance_count + instance_index);
}

XPOOL_DEVICE_FN inline Publication<Admission> &
ArenaView::admission_publication(std::size_t instance_index) const {
  xpool::abort_if(instance_index >= layout().instance_count);
  return *pointer_at<Publication<Admission>>(layout().admission_publications_offset_bytes, instance_index);
}

XPOOL_DEVICE_FN inline Publication<LaneExecution> &
ArenaView::lane_execution_publication(std::size_t executor_lane_index) const {
  xpool::abort_if(executor_lane_index >= layout().executor_lane_count);
  return *pointer_at<Publication<LaneExecution>>(layout().lane_execution_publications_offset_bytes,
                                                          executor_lane_index);
}

XPOOL_DEVICE_FN inline Publication<InputReady> &
ArenaView::input_ready_publication(std::size_t executor_lane_index) const {
  xpool::abort_if(executor_lane_index >= layout().executor_lane_count);
  return *pointer_at<Publication<InputReady>>(layout().input_ready_publications_offset_bytes,
                                                       executor_lane_index);
}

XPOOL_DEVICE_FN inline Publication<RoutingMetadataReady> &
ArenaView::routing_metadata_ready_publication(std::size_t executor_lane_index) const {
  xpool::abort_if(executor_lane_index >= layout().executor_lane_count ||
                  layout().routing_metadata_ready_publications_offset_bytes == 0);
  return *pointer_at<Publication<RoutingMetadataReady>>(
      layout().routing_metadata_ready_publications_offset_bytes, executor_lane_index);
}

XPOOL_DEVICE_FN inline Publication<PartialReady> &
ArenaView::partial_ready_publication(std::size_t source_ffnagent_index, std::size_t executor_lane_index) const {
  xpool::abort_if(source_ffnagent_index >= layout().ffnagent_count ||
                  executor_lane_index >= layout().executor_lane_count);
  return *pointer_at<Publication<PartialReady>>(layout().partial_ready_publications_offset_bytes,
                                                         source_ffnagent_index * layout().executor_lane_count +
                                                             executor_lane_index);
}

XPOOL_DEVICE_FN inline Publication<FfnAgentCompletion> &
ArenaView::ffnagent_completion_publication(std::size_t source_ffnagent_index,
                                                 std::size_t executor_lane_index) const {
  xpool::abort_if(source_ffnagent_index >= layout().ffnagent_count ||
                  executor_lane_index >= layout().executor_lane_count);
  return *pointer_at<Publication<FfnAgentCompletion>>(layout().ffnagent_completion_publications_offset_bytes,
                                                            source_ffnagent_index * layout().executor_lane_count +
                                                                executor_lane_index);
}

XPOOL_DEVICE_FN inline Publication<OutputCommit> &
ArenaView::output_commit_publication(std::size_t instance_index) const {
  xpool::abort_if(instance_index >= layout().instance_count);
  return *pointer_at<Publication<OutputCommit>>(layout().output_commit_publications_offset_bytes,
                                                         instance_index);
}

XPOOL_DEVICE_FN inline Publication<OutputAcknowledgement> &
ArenaView::output_acknowledgement_publication(std::size_t source_atnagent_index,
                                                    std::size_t instance_index) const {
  xpool::abort_if(source_atnagent_index >= layout().atnagent_count || instance_index >= layout().instance_count);
  return *pointer_at<Publication<OutputAcknowledgement>>(
      layout().output_acknowledgement_publications_offset_bytes,
      source_atnagent_index * layout().instance_count + instance_index);
}

XPOOL_DEVICE_FN inline AtnAgentLanePayloadView ArenaView::atnagent_lane_payload(std::size_t executor_lane_index) const {
  xpool::abort_if(executor_lane_index >= layout().executor_lane_count);
  const auto capacity = layout().lane_payload_capacity_bytes;
  const auto lane_count = layout().executor_lane_count;
  return AtnAgentLanePayloadView{
      {pointer_at<std::uint8_t>(layout().lane_payload_storage_offset_bytes, executor_lane_index * capacity),
       pointer_at<std::uint8_t>(layout().lane_payload_storage_offset_bytes,
                                (lane_count + executor_lane_index) * capacity)},
      capacity};
}

XPOOL_DEVICE_FN inline FfnAgentLanePayloadView ArenaView::ffnagent_lane_payload(std::size_t executor_lane_index) const {
  xpool::abort_if(executor_lane_index >= layout().executor_lane_count);
  const auto capacity = layout().lane_payload_capacity_bytes;
  const auto lane_count = layout().executor_lane_count;
  return FfnAgentLanePayloadView{
      {pointer_at<std::uint8_t>(layout().lane_payload_storage_offset_bytes, executor_lane_index * capacity),
       pointer_at<std::uint8_t>(layout().lane_payload_storage_offset_bytes,
                                (lane_count + executor_lane_index) * capacity)},
      capacity};
}

} // namespace xpool::fabric
