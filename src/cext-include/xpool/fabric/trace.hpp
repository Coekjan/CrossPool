#pragma once

/// \file xpool/fabric/trace.hpp
/// \brief Semantic PE-local traces for distributed Fabric invocations.

#include <cuda/std/variant>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include <xpool/abort.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/macros.hpp>
#include <xpool/trace.hpp>

namespace xpool::fabric {

/// Runtime site that owns one Fabric trace record.
enum class FabricTraceKind : std::uint32_t {
  /// AtnAgent submission, transfer, and result handoff.
  AtnAgent = 1,
  /// Coordinator invocation aggregation and scheduling.
  Coordinator = 2,
  /// FfnAgent Executor input and execution.
  Execution = 3,
};

/// Ordered semantic events in one AtnAgent trace.
enum class AtnAgentTraceEvent : std::uint32_t {
  /// Rank-local submission facts were prepared.
  SubmissionPrepared,
  /// Decode model-owned input payload was staged.
  DecodeInputStaged,
  /// Local Submission publication completed; remote observation is not implied.
  SubmissionPublished,
  /// The invocation's distributed Executor admission was observed.
  AdmissionObserved,
  /// Prefill input was staged into the admitted Executor workspace.
  PrefillInputStaged,
  /// Local Prefill input publication calls completed for every FfnAgent.
  PrefillInputPublished,
  /// Canonical invocation result was observed.
  ResultObserved,
  /// Rank-local Transport output was prepared.
  OutputPrepared,
  /// Terminal result was published to the local Transport mailbox.
  TransportEvaluatedPublished,
  /// Local result-consumption acknowledgement publication completed.
  AcknowledgementPublished,
  /// Number of concrete AtnAgent trace events.
  Count,
};

/// Ordered semantic events in one Coordinator trace.
enum class CoordinatorTraceEvent : std::uint32_t {
  /// Complete matching submissions entered scheduler readiness.
  Enqueued,
  /// Scheduler leased one distributed Executor.
  Scheduled,
  /// Local Admission publication calls completed for every AtnAgent.
  AdmissionsPublished,
  /// Local Invocation publication calls completed for every FfnAgent.
  InvocationsPublished,
  /// Every required FfnAgent completion was observed.
  CompletionsObserved,
  /// Local Result publication calls completed for every AtnAgent.
  ResultsPublished,
  /// Every AtnAgent result acknowledgement was observed.
  AcknowledgementsObserved,
  /// Scheduler entry and Executor lease were released for reuse.
  SchedulerReleased,
  /// Number of concrete Coordinator trace events.
  Count,
};

/// Ordered semantic events in one FfnAgent Execution trace.
enum class ExecutionTraceEvent : std::uint32_t {
  /// FfnAgent observed one admitted invocation.
  InvocationObserved,
  /// Decode input pull from the model-owned payload began.
  DecodeInputPullStarted,
  /// Decode input pull completed.
  DecodeInputPullCompleted,
  /// Prefill Executor input readiness was observed.
  PrefillInputReadyObserved,
  /// FfnAgent entered the execution extension boundary.
  ExecutionStarted,
  /// Execution returned a terminal result code.
  ExecutionCompleted,
  /// Local Completion publication to the Coordinator completed.
  CompletionPublished,
  /// Number of concrete Execution trace events.
  Count,
};

/// One retained semantic Fabric trace with site-specific private state.
class alignas(8) FabricTraceRecord {
public:
  /// Return this PE-local monotonic trace identity.
  /// \return Positive local trace sequence.
  std::uint64_t local_trace_id() const { return local_trace_id_; }

  /// Return the site owning this trace.
  /// \return AtnAgent, Coordinator, or Execution according to active state.
  FabricTraceKind kind() const {
    if (cuda::std::holds_alternative<AtnAgentState>(state_)) {
      return FabricTraceKind::AtnAgent;
    }
    if (cuda::std::holds_alternative<CoordinatorState>(state_)) {
      return FabricTraceKind::Coordinator;
    }
    if (cuda::std::holds_alternative<ExecutionState>(state_)) {
      return FabricTraceKind::Execution;
    }
    xpool::abort();
  }

  /// Return the traced model and invocation sequence.
  /// \return Immutable invocation identity.
  const FfnInvocationKey &key() const { return key_; }

  /// Return the config-order FFN layer ordinal.
  /// \return Model-local layer ordinal.
  std::size_t layer_ordinal() const { return layer_ordinal_; }

  /// Return the model's concrete decoder layer identifier.
  /// \return Decoder layer identifier from the canonical layer table.
  std::size_t layer_id() const { return layer_id_; }

  /// Return the raw abi::FfnResultHandoff value.
  /// \return Stable result-handoff ABI value.
  std::uint32_t result_handoff() const { return result_handoff_; }

  /// Return the raw abi::DpPaddingMode value.
  /// \return Stable DP-padding ABI value.
  std::uint32_t dp_padding_mode() const { return dp_padding_mode_; }

  /// Return AtnAgent submission rows when this is an AtnAgent trace.
  /// \return Physical submission rows, or no value for another trace kind.
  std::optional<std::size_t> submission_payload_rows() const {
    const auto *state = cuda::std::get_if<AtnAgentState>(&state_);
    return state == nullptr ? std::nullopt : std::optional<std::size_t>{state->submission_payload_rows};
  }

  /// Return Coordinator/Execution invocation rows when available.
  /// \return Complete invocation rows, or no value for an AtnAgent trace.
  std::optional<std::size_t> invocation_payload_rows() const {
    if (const auto *state = cuda::std::get_if<CoordinatorState>(&state_)) {
      return state->invocation_payload_rows;
    }
    if (const auto *state = cuda::std::get_if<ExecutionState>(&state_)) {
      return state->invocation_payload_rows;
    }
    return std::nullopt;
  }

  /// Return AtnAgent rank-local live token count when available.
  /// \return Local live-token count, or no value for another trace kind.
  std::optional<std::size_t> local_token_count() const {
    const auto *state = cuda::std::get_if<AtnAgentState>(&state_);
    return state == nullptr ? std::nullopt : std::optional<std::size_t>{state->local_token_count};
  }

  /// Return AtnAgent raw forward mode when available.
  /// \return Stable forward-mode ABI value, or no value for another kind.
  std::optional<std::uint32_t> forward_mode() const {
    const auto *state = cuda::std::get_if<AtnAgentState>(&state_);
    return state == nullptr ? std::nullopt : std::optional<std::uint32_t>{state->forward_mode};
  }

  /// Return the derived Input Publisher PE when available.
  /// \return Input Publisher PE, or no value for an AtnAgent trace.
  std::optional<int> input_pe() const {
    if (const auto *state = cuda::std::get_if<CoordinatorState>(&state_)) {
      return state->input_pe;
    }
    if (const auto *state = cuda::std::get_if<ExecutionState>(&state_)) {
      return state->input_pe;
    }
    return std::nullopt;
  }

  /// Return the admitted Executor after its establishing event.
  /// \return Executor index when established, otherwise no value.
  std::optional<std::size_t> executor_index() const {
    if (const auto *state = cuda::std::get_if<AtnAgentState>(&state_)) {
      return state->timeline.recorded(AtnAgentTraceEvent::AdmissionObserved)
                 ? std::optional<std::size_t>{state->executor_index}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<CoordinatorState>(&state_)) {
      return state->timeline.recorded(CoordinatorTraceEvent::Scheduled)
                 ? std::optional<std::size_t>{state->executor_index}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<ExecutionState>(&state_)) {
      return state->executor_index;
    }
    return std::nullopt;
  }

  /// Return the derived execution mode after its establishing event.
  /// \return Stable execution-mode value when established, otherwise no value.
  std::optional<std::uint32_t> execution_mode() const {
    if (const auto *state = cuda::std::get_if<AtnAgentState>(&state_)) {
      return state->timeline.recorded(AtnAgentTraceEvent::AdmissionObserved)
                 ? std::optional<std::uint32_t>{state->execution_mode}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<CoordinatorState>(&state_)) {
      return state->execution_mode;
    }
    if (const auto *state = cuda::std::get_if<ExecutionState>(&state_)) {
      return state->execution_mode;
    }
    return std::nullopt;
  }

  /// Return the result contribution after AtnAgent Result observation.
  /// \return Stable contribution value when observed, otherwise no value.
  std::optional<std::uint32_t> result_contribution() const {
    const auto *state = cuda::std::get_if<AtnAgentState>(&state_);
    return state != nullptr && state->timeline.recorded(AtnAgentTraceEvent::ResultObserved)
               ? std::optional<std::uint32_t>{state->result_contribution}
               : std::nullopt;
  }

  /// Return a FIFO ticket, or no value for Random Coordinator traces.
  /// \return Positive FIFO ticket, or no value when not applicable.
  std::optional<std::uint64_t> ready_ticket() const {
    const auto *state = cuda::std::get_if<CoordinatorState>(&state_);
    if (state == nullptr) {
      return std::nullopt;
    }
    if (const auto *fifo = cuda::std::get_if<CoordinatorFifoState>(&state->scheduling)) {
      return fifo->ready_ticket;
    }
    return std::nullopt;
  }

  /// Return whether one AtnAgent event was recorded.
  /// \param event AtnAgent event to query.
  /// \return True when the timestamp is present.
  bool recorded(AtnAgentTraceEvent event) const { return require_atnagent().timeline.recorded(event); }

  /// Return whether one Coordinator event was recorded.
  /// \param event Coordinator event to query.
  /// \return True when the timestamp is present.
  bool recorded(CoordinatorTraceEvent event) const { return require_coordinator().timeline.recorded(event); }

  /// Return whether one Execution event was recorded.
  /// \param event Execution event to query.
  /// \return True when the timestamp is present.
  bool recorded(ExecutionTraceEvent event) const { return require_execution().timeline.recorded(event); }

  /// Return one AtnAgent event timestamp.
  /// \param event AtnAgent event to read.
  /// \return Raw global-timer value, or zero when absent.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(AtnAgentTraceEvent event) const {
    return require_atnagent().timeline.timestamp(event);
  }

  /// Return one Coordinator event timestamp.
  /// \param event Coordinator event to read.
  /// \return Raw global-timer value, or zero when absent.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(CoordinatorTraceEvent event) const {
    return require_coordinator().timeline.timestamp(event);
  }

  /// Return one Execution event timestamp.
  /// \param event Execution event to read.
  /// \return Raw global-timer value, or zero when absent.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(ExecutionTraceEvent event) const {
    return require_execution().timeline.timestamp(event);
  }

#if defined(__CUDACC__)
  /// Initialize one AtnAgent trace and record SubmissionPrepared.
  /// \param local_trace_id Positive local trace sequence.
  /// \param submission Valid AtnAgent submission facts.
  /// \param layer_id Concrete decoder layer identifier.
  XPOOL_DEVICE_FN void begin_atnagent(std::uint64_t local_trace_id, const FfnSubmission &submission,
                                     std::size_t layer_id);

  /// Record completion of conservative Decode input staging.
  XPOOL_DEVICE_FN void decode_input_staged();

  /// Record publication of this AtnAgent's Submission.
  XPOOL_DEVICE_FN void submission_published();

  /// Record Admission observation and its derived Executor/mode facts.
  /// \param admission Valid matching Coordinator admission.
  XPOOL_DEVICE_FN void admission_observed(const FfnExecutionAdmission &admission);

  /// Record completion of Prefill input staging into Executor storage.
  XPOOL_DEVICE_FN void prefill_input_staged();

  /// Record Prefill input payload and InputReady publication.
  XPOOL_DEVICE_FN void prefill_input_published();

  /// Record successful Result observation and contribution semantics.
  /// \param result Valid matching successful result.
  XPOOL_DEVICE_FN void result_observed(const FfnResult &result);

  /// Record completion of output copy or zero materialization.
  XPOOL_DEVICE_FN void output_prepared();

  /// Record publication of successful Transport Evaluated.
  XPOOL_DEVICE_FN void transport_evaluated_published();

  /// Record publication of the Fabric result Acknowledgement.
  XPOOL_DEVICE_FN void acknowledgement_published();

  /// Initialize one Coordinator trace without selecting a scheduling policy.
  /// \param local_trace_id Positive local trace sequence.
  /// \param invocation Complete Coordinator-derived invocation.
  /// \param layer_id Concrete decoder layer identifier.
  XPOOL_DEVICE_FN void begin_coordinator(std::uint64_t local_trace_id, const FfnInvocation &invocation,
                                        std::size_t layer_id);

  /// Record FIFO enqueue and its positive ticket.
  /// \param ready_ticket Positive monotonic FIFO ticket.
  XPOOL_DEVICE_FN void fifo_enqueued(std::uint64_t ready_ticket);

  /// Record Random-policy enqueue without a FIFO ticket.
  XPOOL_DEVICE_FN void random_enqueued();

  /// Record scheduling and establish the Executor lease.
  /// \param executor_index Distributed Executor leased to this invocation.
  XPOOL_DEVICE_FN void scheduled(std::size_t executor_index);

  /// Record publication of every AtnAgent Admission.
  XPOOL_DEVICE_FN void admissions_published();

  /// Record publication of the Executor Invocation set.
  XPOOL_DEVICE_FN void invocations_published();

  /// Record observation of every matching FfnAgent Completion.
  XPOOL_DEVICE_FN void completions_observed();

  /// Record publication of every AtnAgent Result.
  XPOOL_DEVICE_FN void results_published();

  /// Record observation of every matching Acknowledgement.
  XPOOL_DEVICE_FN void acknowledgements_observed();

  /// Record release of the model Scheduler entry and Executor.
  XPOOL_DEVICE_FN void scheduler_released();

  /// Initialize one FfnAgent Execution trace and record InvocationObserved.
  /// \param local_trace_id Positive local trace sequence.
  /// \param invocation Complete invocation assigned to the Executor.
  /// \param layer_id Concrete decoder layer identifier.
  /// \param executor_index Distributed Executor executing the invocation.
  XPOOL_DEVICE_FN void begin_execution(std::uint64_t local_trace_id, const FfnInvocation &invocation,
                                      std::size_t layer_id, std::size_t executor_index);

  /// Record the start of a Decode input pull.
  XPOOL_DEVICE_FN void decode_input_pull_started();

  /// Record completion of a Decode input pull.
  XPOOL_DEVICE_FN void decode_input_pull_completed();

  /// Record observation of Prefill InputReady.
  XPOOL_DEVICE_FN void prefill_input_ready_observed();

  /// Record entry into local FfnAgent execution.
  XPOOL_DEVICE_FN void execution_started();

  /// Record completion of local FfnAgent execution.
  XPOOL_DEVICE_FN void execution_completed();

  /// Record publication of this FfnAgent's success-only Completion.
  XPOOL_DEVICE_FN void completion_published();
#endif

private:
  struct AtnAgentState {
    std::size_t submission_payload_rows = 0;
    std::size_t local_token_count = 0;
    std::uint32_t forward_mode = 0;
    std::size_t executor_index = 0;
    std::uint32_t execution_mode = 0;
    std::uint32_t result_contribution = 0;
    xpool::trace::Timeline<AtnAgentTraceEvent> timeline;
  };

  struct CoordinatorFifoState {
    std::uint64_t ready_ticket = 0;
  };

  struct CoordinatorRandomState {};

  using CoordinatorSchedulingState =
      cuda::std::variant<cuda::std::monostate, CoordinatorFifoState, CoordinatorRandomState>;

  struct CoordinatorState {
    std::size_t invocation_payload_rows = 0;
    int input_pe = 0;
    std::uint32_t execution_mode = 0;
    std::size_t executor_index = 0;
    CoordinatorSchedulingState scheduling;
    xpool::trace::Timeline<CoordinatorTraceEvent> timeline;
  };

  struct ExecutionState {
    std::size_t invocation_payload_rows = 0;
    int input_pe = 0;
    std::uint32_t execution_mode = 0;
    std::size_t executor_index = 0;
    xpool::trace::Timeline<ExecutionTraceEvent> timeline;
  };

  using State =
      cuda::std::variant<cuda::std::monostate, AtnAgentState, CoordinatorState, ExecutionState>;

#if defined(__CUDACC__)
  XPOOL_DEVICE_FN void begin(std::uint64_t local_trace_id, const FfnInvocationKey &key,
                             std::size_t layer_ordinal, std::size_t layer_id, std::uint32_t result_handoff,
                             std::uint32_t dp_padding_mode);
#endif

  XPOOL_HOST_DEVICE_FN AtnAgentState &require_atnagent() {
    auto *state = cuda::std::get_if<AtnAgentState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  XPOOL_HOST_DEVICE_FN const AtnAgentState &require_atnagent() const {
    const auto *state = cuda::std::get_if<AtnAgentState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  XPOOL_HOST_DEVICE_FN CoordinatorState &require_coordinator() {
    auto *state = cuda::std::get_if<CoordinatorState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  XPOOL_HOST_DEVICE_FN const CoordinatorState &require_coordinator() const {
    const auto *state = cuda::std::get_if<CoordinatorState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  XPOOL_HOST_DEVICE_FN ExecutionState &require_execution() {
    auto *state = cuda::std::get_if<ExecutionState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  XPOOL_HOST_DEVICE_FN const ExecutionState &require_execution() const {
    const auto *state = cuda::std::get_if<ExecutionState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  std::uint64_t local_trace_id_ = 0;
  FfnInvocationKey key_{};
  std::size_t layer_ordinal_ = 0;
  std::size_t layer_id_ = 0;
  std::uint32_t result_handoff_ = 0;
  std::uint32_t dp_padding_mode_ = 0;
  State state_{};
};

/// Static model topology retained beside one PE-local snapshot.
struct FabricTraceModelTopology {
  /// Tensor-parallel AtnAgent width.
  std::size_t atn_tp_size;
  /// Data-parallel AtnAgent width.
  std::size_t atn_dp_size;
};

/// Host-owned snapshot copied from one PE-local Fabric trace buffer.
struct FabricTraceSnapshot {
  /// Concrete NVSHMEM PE whose local records were copied.
  int pe;
  /// Number of AtnAgent PEs in the generation.
  std::size_t atnagent_count;
  /// Number of FfnAgent PEs in the generation.
  std::size_t ffnagent_count;
  /// Ordered model topology needed to interpret AtnAgent coordinates.
  std::vector<FabricTraceModelTopology> model_topologies;
  /// Number of trace identities requested on this PE.
  std::uint64_t sequence;
  /// Number of reservations dropped after capacity exhaustion.
  std::uint64_t dropped;
  /// Retained local records in monotonic identity order.
  std::vector<FabricTraceRecord> records;
};

static_assert(std::is_trivially_copyable_v<FabricTraceModelTopology>);
static_assert(std::is_trivially_copyable_v<FabricTraceRecord>);

} // namespace xpool::fabric
