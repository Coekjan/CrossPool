#pragma once

/// \file xpool/devkit/fabric_observer.cuh
/// \brief Device implementations for Fabric Observer state transitions.

#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/utils/time.cuh>

namespace xpool::devkit::fabric_observer {

XPOOL_DEVICE_FN inline void Record::begin_atnagent(std::uint64_t local_trace_id,
                                                   const xpool::fabric::Submission &submission) {
  xpool::abort_if(submission.validate() != xpool::ffn::ResultCode::Ok);
  begin(local_trace_id, submission.key, submission.layer_ordinal, submission.payload_rows,
        submission.output_requirement);
  state_.template emplace<AtnAgentTraceState>(AtnAgentTraceState{
      .dp_rank_payload_rows = submission.dp_rank_payload_rows,
      .forward_mode = submission.forward_mode,
      .dp_row_layout = submission.dp_row_layout,
  });
  require_atnagent().timeline.template record<xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPrepared>(
      xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::submission_published() {
  require_atnagent()
      .timeline.template record<xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPublished,
                                xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPrepared>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::admission_observed(const xpool::fabric::Admission &admission) {
  xpool::abort_if(admission.validate() != xpool::ffn::ResultCode::Ok || admission.key != key_);
  auto &state = require_atnagent();
  state.executor_lane_index = admission.executor_lane_index;
  state.executor_lease_sequence = admission.executor_lease_sequence;
  state.timeline.template record<xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved,
                                 xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPublished>(
      xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::input_ready_published() {
  require_atnagent()
      .timeline.template record<xpool::hooks::FabricAtnAgentProtocolEvent::Kind::InputReadyPublished,
                                xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::output_commit_observed(const xpool::fabric::OutputCommit &commit) {
  xpool::abort_if(commit.validate() != xpool::ffn::ResultCode::Ok || commit.key != key_);
  require_atnagent()
      .timeline.template record<xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputCommitObserved,
                                xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::output_acknowledgement_published() {
  require_atnagent()
      .timeline.template record<xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputAcknowledgementPublished,
                                xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputCommitObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::begin_coordinator(std::uint64_t local_trace_id,
                                                      const xpool::fabric::Invocation &invocation) {
  xpool::abort_if(invocation.validate() != xpool::ffn::ResultCode::Ok);
  begin(local_trace_id, invocation.key, invocation.layer_ordinal, invocation.payload_rows,
        invocation.output_requirement);
  state_.template emplace<CoordinatorTraceState>();
}

XPOOL_DEVICE_FN inline void Record::enqueued(std::uint64_t ready_ticket) {
  auto &state = require_coordinator();
  state.ready_ticket = ready_ticket;
  state.timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Enqueued>(
      xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::scheduled(std::size_t executor_lane_index, std::uint64_t executor_lease_sequence) {
  xpool::abort_if(executor_lease_sequence == 0);
  auto &state = require_coordinator();
  state.executor_lane_index = executor_lane_index;
  state.executor_lease_sequence = executor_lease_sequence;
  state.timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Scheduled,
                                 xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Enqueued>(
      xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::admission_published() {
  require_coordinator()
      .timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::AdmissionPublished,
                                xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Scheduled>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::lane_execution_published() {
  require_coordinator()
      .timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneExecutionPublished,
                                xpool::hooks::FabricCoordinatorProtocolEvent::Kind::AdmissionPublished>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::ffnagent_completions_observed() {
  require_coordinator()
      .timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::FfnAgentCompletionsObserved,
                                xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneExecutionPublished>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::output_commit_published() {
  require_coordinator()
      .timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputCommitPublished,
                                xpool::hooks::FabricCoordinatorProtocolEvent::Kind::FfnAgentCompletionsObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::output_acknowledgements_observed() {
  require_coordinator()
      .timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputAcknowledgementsObserved,
                                xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputCommitPublished>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::lane_released() {
  require_coordinator()
      .timeline.template record<xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneReleased,
                                xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputAcknowledgementsObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::begin_ffnagent(std::uint64_t local_trace_id,
                                                   const xpool::fabric::LaneExecution &execution,
                                                   std::size_t executor_lane_index, std::size_t payload_row_capacity,
                                                   xpool::fabric::DeliveryVariant delivery) {
  xpool::abort_if(execution.validate() != xpool::ffn::ResultCode::Ok || payload_row_capacity == 0 ||
                  !xpool::fabric::is_valid(delivery));
  begin(local_trace_id, execution.key, execution.layer_ordinal, execution.payload_rows, execution.output_requirement);
  state_.template emplace<FfnAgentTraceState>(FfnAgentTraceState{
      .executor_lane_index = executor_lane_index,
      .executor_lease_sequence = execution.executor_lease_sequence,
      .payload_row_capacity = payload_row_capacity,
      .delivery = delivery,
  });
  require_ffnagent().timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::LaneExecutionObserved>(
      xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::input_ready_observed() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::InputReadyObserved,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::LaneExecutionObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::routing_metadata_published() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataPublished,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::InputReadyObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::routing_metadata_observed() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataObserved,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::InputReadyObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::compute_started() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeStarted,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::InputReadyObserved>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::compute_completed() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeCompleted,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeStarted>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::partial_ready_published() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PartialReadyPublished,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeCompleted>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::peer_partials_ready_observed() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PeerPartialsReadyObserved,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeCompleted>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::completion_published() {
  require_ffnagent()
      .timeline.template record<xpool::hooks::FabricFfnAgentProtocolEvent::Kind::CompletionPublished,
                                xpool::hooks::FabricFfnAgentProtocolEvent::Kind::ComputeCompleted>(
          xpool::utils::time::now());
}

XPOOL_DEVICE_FN inline void Record::begin(std::uint64_t local_trace_id, const xpool::fabric::InvocationKey &key,
                                          std::size_t layer_ordinal, std::size_t payload_rows,
                                          xpool::ffn::OutputRequirement output_requirement) {
  xpool::abort_if(local_trace_id == 0 || !key.valid() || payload_rows == 0 ||
                  !cuda::std::holds_alternative<cuda::std::monostate>(state_) ||
                  !xpool::ffn::is_valid(output_requirement));
  local_trace_id_ = local_trace_id;
  key_ = key;
  layer_ordinal_ = layer_ordinal;
  payload_rows_ = payload_rows;
  output_requirement_ = output_requirement;
}

} // namespace xpool::devkit::fabric_observer
