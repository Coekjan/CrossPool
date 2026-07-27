#pragma once

/// \file xpool/fabric/trace.cuh
/// \brief Device implementations for semantic trace state transitions.

#include <xpool/fabric/trace.hpp>
#include <xpool/utils/time.cuh>

namespace xpool::fabric {

XPOOL_DEVICE_FN inline void FabricTraceRecord::begin_atnagent(std::uint64_t local_trace_id,
                                                              const FfnSubmission &submission,
                                                              std::size_t layer_id) {
    begin(local_trace_id, submission.key, submission.layer_ordinal, layer_id, submission.result_handoff,
          submission.dp_padding_mode);
    state_.template emplace<AtnAgentState>(AtnAgentState{
        .submission_payload_rows = submission.payload_rows,
        .local_token_count = submission.local_token_count,
        .forward_mode = submission.forward_mode,
    });
    require_atnagent().timeline.template record<AtnAgentTraceEvent::SubmissionPrepared>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::decode_input_staged() {
    auto &state = require_atnagent();
    state.timeline.template record<AtnAgentTraceEvent::DecodeInputStaged,
                                   AtnAgentTraceEvent::SubmissionPrepared>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::submission_published() {
    auto &state = require_atnagent();
    state.timeline.template record<AtnAgentTraceEvent::SubmissionPublished,
                                   AtnAgentTraceEvent::SubmissionPrepared>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::admission_observed(const FfnExecutionAdmission &admission) {
    xpool::abort_if(admission.validate() != xpool::abi::FfnResultCode::Ok || admission.key != key_);
    auto &state = require_atnagent();
    state.executor_index = admission.executor_index;
    state.execution_mode = admission.execution_mode;
    state.timeline.template record<AtnAgentTraceEvent::AdmissionObserved,
                                   AtnAgentTraceEvent::SubmissionPublished>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::prefill_input_staged() {
    auto &state = require_atnagent();
    state.timeline.template record<AtnAgentTraceEvent::PrefillInputStaged,
                                   AtnAgentTraceEvent::AdmissionObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::prefill_input_published() {
    auto &state = require_atnagent();
    state.timeline.template record<AtnAgentTraceEvent::PrefillInputPublished,
                                   AtnAgentTraceEvent::PrefillInputStaged>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::result_observed(const FfnResult &result) {
    xpool::abort_if(result.validate() != xpool::abi::FfnResultCode::Ok || result.key != key_);
    auto &state = require_atnagent();
    state.result_contribution = result.contribution;
    state.timeline.template record<AtnAgentTraceEvent::ResultObserved,
                                   AtnAgentTraceEvent::AdmissionObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::output_prepared() {
    require_atnagent().timeline.template record<AtnAgentTraceEvent::OutputPrepared,
                                                AtnAgentTraceEvent::ResultObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::transport_evaluated_published() {
    require_atnagent().timeline.template record<AtnAgentTraceEvent::TransportEvaluatedPublished,
                                                AtnAgentTraceEvent::OutputPrepared>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::acknowledgement_published() {
    require_atnagent().timeline.template record<AtnAgentTraceEvent::AcknowledgementPublished,
                                                AtnAgentTraceEvent::TransportEvaluatedPublished>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::begin_coordinator(std::uint64_t local_trace_id, const FfnInvocation &invocation,
                                        std::size_t layer_id) {
    begin(local_trace_id, invocation.key, invocation.layer_ordinal, layer_id, invocation.result_handoff,
          invocation.dp_padding_mode);
    state_.template emplace<CoordinatorState>(CoordinatorState{
        .invocation_payload_rows = invocation.payload_rows,
        .input_pe = invocation.input_pe,
        .execution_mode = invocation.execution_mode,
    });
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::fifo_enqueued(std::uint64_t ready_ticket) {
    xpool::abort_if(ready_ticket == 0);
    auto &state = require_coordinator();
    state.scheduling.template emplace<CoordinatorFifoState>(CoordinatorFifoState{.ready_ticket = ready_ticket});
    state.timeline.template record<CoordinatorTraceEvent::Enqueued>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::random_enqueued() {
    auto &state = require_coordinator();
    state.scheduling.template emplace<CoordinatorRandomState>();
    state.timeline.template record<CoordinatorTraceEvent::Enqueued>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::scheduled(std::size_t executor_index) {
    auto &state = require_coordinator();
    state.executor_index = executor_index;
    state.timeline.template record<CoordinatorTraceEvent::Scheduled,
                                   CoordinatorTraceEvent::Enqueued>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::admissions_published() {
    require_coordinator().timeline.template record<CoordinatorTraceEvent::AdmissionsPublished,
                                                    CoordinatorTraceEvent::Scheduled>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::invocations_published() {
    require_coordinator().timeline.template record<CoordinatorTraceEvent::InvocationsPublished,
                                                    CoordinatorTraceEvent::AdmissionsPublished>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::completions_observed() {
    require_coordinator().timeline.template record<CoordinatorTraceEvent::CompletionsObserved,
                                                    CoordinatorTraceEvent::InvocationsPublished>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::results_published() {
    require_coordinator().timeline.template record<CoordinatorTraceEvent::ResultsPublished,
                                                    CoordinatorTraceEvent::CompletionsObserved>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::acknowledgements_observed() {
    require_coordinator().timeline.template record<CoordinatorTraceEvent::AcknowledgementsObserved,
                                                    CoordinatorTraceEvent::ResultsPublished>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::scheduler_released() {
    require_coordinator().timeline.template record<CoordinatorTraceEvent::SchedulerReleased,
                                                    CoordinatorTraceEvent::AcknowledgementsObserved>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::begin_execution(std::uint64_t local_trace_id, const FfnInvocation &invocation,
                                      std::size_t layer_id, std::size_t executor_index) {
    begin(local_trace_id, invocation.key, invocation.layer_ordinal, layer_id, invocation.result_handoff,
          invocation.dp_padding_mode);
    state_.template emplace<ExecutionState>(ExecutionState{
        .invocation_payload_rows = invocation.payload_rows,
        .input_pe = invocation.input_pe,
        .execution_mode = invocation.execution_mode,
        .executor_index = executor_index,
    });
    require_execution().timeline.template record<ExecutionTraceEvent::InvocationObserved>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::decode_input_pull_started() {
    auto &state = require_execution();
    xpool::abort_if(state.execution_mode != FfnExecutionMode::Decode);
    state.timeline.template record<ExecutionTraceEvent::DecodeInputPullStarted,
                                   ExecutionTraceEvent::InvocationObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::decode_input_pull_completed() {
    require_execution().timeline.template record<ExecutionTraceEvent::DecodeInputPullCompleted,
                                                  ExecutionTraceEvent::DecodeInputPullStarted>(
        xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::prefill_input_ready_observed() {
    auto &state = require_execution();
    xpool::abort_if(state.execution_mode != FfnExecutionMode::Prefill);
    state.timeline.template record<ExecutionTraceEvent::PrefillInputReadyObserved,
                                   ExecutionTraceEvent::InvocationObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::execution_started() {
    auto &state = require_execution();
    const auto input_ready = state.execution_mode == FfnExecutionMode::Decode
                                 ? state.timeline.recorded(ExecutionTraceEvent::DecodeInputPullCompleted)
                                 : state.timeline.recorded(ExecutionTraceEvent::PrefillInputReadyObserved);
    xpool::abort_if(!input_ready);
    state.timeline.template record<ExecutionTraceEvent::ExecutionStarted,
                                   ExecutionTraceEvent::InvocationObserved>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::execution_completed() {
    require_execution().timeline.template record<ExecutionTraceEvent::ExecutionCompleted,
                                                  ExecutionTraceEvent::ExecutionStarted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::completion_published() {
    require_execution().timeline.template record<ExecutionTraceEvent::CompletionPublished,
                                                  ExecutionTraceEvent::ExecutionCompleted>(xpool::utils::time::now());
  }

XPOOL_DEVICE_FN inline void FabricTraceRecord::begin(std::uint64_t local_trace_id, const FfnInvocationKey &key,
                             std::size_t layer_ordinal, std::size_t layer_id, std::uint32_t result_handoff,
                             std::uint32_t dp_padding_mode) {
    xpool::abort_if(local_trace_id == 0 || !key.valid() || !cuda::std::holds_alternative<cuda::std::monostate>(state_) ||
                    !xpool::abi::FfnResultHandoff::is_valid(result_handoff) ||
                    !xpool::abi::DpPaddingMode::is_valid(dp_padding_mode));
    local_trace_id_ = local_trace_id;
    key_ = key;
    layer_ordinal_ = layer_ordinal;
    layer_id_ = layer_id;
    result_handoff_ = result_handoff;
    dp_padding_mode_ = dp_padding_mode;
  }

} // namespace xpool::fabric
