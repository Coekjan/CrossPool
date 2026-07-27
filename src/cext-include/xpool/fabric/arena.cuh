#pragma once

/// \file xpool/fabric/arena.cuh
/// \brief Typed device addressing over one process-local Fabric arena.

#include <cstddef>
#include <cstdint>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/protocol.cuh>
#include <xpool/fabric/scheduler.hpp>
#include <xpool/fabric/trace.cuh>
#include <xpool/macros.hpp>
#include <xpool/trace.cuh>

namespace xpool::fabric {

template <typename T>
XPOOL_DEVICE_FN T *FabricArenaView::pointer_at(std::size_t offset, std::size_t index) const {
  const auto &arena_layout = layout();
  xpool::abort_if(offset == 0 || offset > arena_layout.header.total_bytes);
  const auto remaining = arena_layout.header.total_bytes - offset;
  xpool::abort_if(index > remaining / sizeof(T));
  const auto relative_offset = index * sizeof(T);
  xpool::abort_if(sizeof(T) > remaining - relative_offset);
  return reinterpret_cast<T *>(base_ + offset + relative_offset);
}

XPOOL_DEVICE_FN inline const FabricArenaLayout &FabricArenaView::layout() const {
  xpool::abort_if(base_ == nullptr);
  return *reinterpret_cast<const FabricArenaLayout *>(base_);
}

XPOOL_DEVICE_FN inline FabricArenaState &FabricArenaView::state() const {
  return *pointer_at<FabricArenaState>(layout().header.state_offset);
}

XPOOL_DEVICE_FN inline const FabricModelLayout &FabricArenaView::model_layout(std::size_t model_index) const {
  xpool::abort_if(model_index >= layout().model_count);
  return *pointer_at<const FabricModelLayout>(layout().model_layouts_offset, model_index);
}

XPOOL_DEVICE_FN inline const FabricLayerLayout &FabricArenaView::layer_layout(std::size_t layer_index) const {
  xpool::abort_if(layer_index >= layout().layer_count);
  return *pointer_at<const FabricLayerLayout>(layout().layer_layouts_offset, layer_index);
}

XPOOL_DEVICE_FN inline FfnScheduler &FabricArenaView::scheduler() const {
  return *pointer_at<FfnScheduler>(layout().scheduler_offset);
}

XPOOL_DEVICE_FN inline FfnSchedulerEntry &FabricArenaView::scheduler_entry(std::size_t model_index) const {
  xpool::abort_if(model_index >= layout().model_count);
  return *pointer_at<FfnSchedulerEntry>(layout().scheduler_entries_offset, model_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnSubmission> &
FabricArenaView::submission_publication(std::size_t atnagent_index, std::size_t model_index) const {
  xpool::abort_if(atnagent_index >= layout().atnagent_count || model_index >= layout().model_count);
  return *pointer_at<FabricPublication<FfnSubmission>>(
      layout().submission_publications_offset, atnagent_index * layout().model_count + model_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnExecutionAdmission> &
FabricArenaView::admission_publication(std::size_t atnagent_index, std::size_t model_index) const {
  xpool::abort_if(atnagent_index >= layout().atnagent_count || model_index >= layout().model_count);
  return *pointer_at<FabricPublication<FfnExecutionAdmission>>(
      layout().admission_publications_offset, atnagent_index * layout().model_count + model_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnInvocation> &
FabricArenaView::invocation_publication(std::size_t executor_index) const {
  xpool::abort_if(executor_index >= layout().executor_count);
  return *pointer_at<FabricPublication<FfnInvocation>>(layout().invocation_publications_offset, executor_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnInputReady> &
FabricArenaView::input_ready_publication(std::size_t executor_index) const {
  xpool::abort_if(executor_index >= layout().executor_count);
  return *pointer_at<FabricPublication<FfnInputReady>>(layout().input_ready_publications_offset, executor_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnAgentCompletion> &
FabricArenaView::ffnagent_completion_publication(std::size_t ffnagent_index, std::size_t executor_index) const {
  xpool::abort_if(ffnagent_index >= layout().ffnagent_count || executor_index >= layout().executor_count);
  return *pointer_at<FabricPublication<FfnAgentCompletion>>(
      layout().ffnagent_completion_publications_offset,
      ffnagent_index * layout().executor_count + executor_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnResult> &
FabricArenaView::result_publication(std::size_t atnagent_index, std::size_t model_index) const {
  xpool::abort_if(atnagent_index >= layout().atnagent_count || model_index >= layout().model_count);
  return *pointer_at<FabricPublication<FfnResult>>(
      layout().result_publications_offset, atnagent_index * layout().model_count + model_index);
}

XPOOL_DEVICE_FN inline FabricPublication<FfnResultAcknowledgement> &
FabricArenaView::acknowledgement_publication(std::size_t atnagent_index, std::size_t model_index) const {
  xpool::abort_if(atnagent_index >= layout().atnagent_count || model_index >= layout().model_count);
  return *pointer_at<FabricPublication<FfnResultAcknowledgement>>(
      layout().acknowledgement_publications_offset, atnagent_index * layout().model_count + model_index);
}

XPOOL_DEVICE_FN inline std::uint8_t *FabricArenaView::model_input_payload(std::size_t model_index) const {
  return pointer_at<std::uint8_t>(layout().model_input_payloads_offset,
                                  model_layout(model_index).decode_payload_offset);
}

XPOOL_DEVICE_FN inline std::uint8_t *FabricArenaView::model_output_payload(std::size_t model_index) const {
  return pointer_at<std::uint8_t>(layout().model_output_payloads_offset,
                                  model_layout(model_index).decode_payload_offset);
}

XPOOL_DEVICE_FN inline std::uint8_t *FabricArenaView::executor_input_payload(std::size_t executor_index) const {
  xpool::abort_if(executor_index >= layout().executor_count);
  return pointer_at<std::uint8_t>(layout().executor_input_payloads_offset,
                                  executor_index * layout().executor_payload_capacity_bytes);
}

XPOOL_DEVICE_FN inline std::uint8_t *FabricArenaView::executor_output_payload(std::size_t executor_index) const {
  xpool::abort_if(executor_index >= layout().executor_count);
  return pointer_at<std::uint8_t>(layout().executor_output_payloads_offset,
                                  executor_index * layout().executor_payload_capacity_bytes);
}

XPOOL_DEVICE_FN inline xpool::trace::Entry<FabricTraceRecord> FabricArenaView::reserve_trace() const {
  return xpool::trace::Buffer<FabricTraceRecord>{base_, layout().trace, state().trace}.reserve();
}

XPOOL_DEVICE_FN inline FabricTraceRecord *FabricArenaView::find_trace(std::uint64_t local_trace_id) const {
  return xpool::trace::Buffer<FabricTraceRecord>{base_, layout().trace, state().trace}.find(local_trace_id);
}

} // namespace xpool::fabric
