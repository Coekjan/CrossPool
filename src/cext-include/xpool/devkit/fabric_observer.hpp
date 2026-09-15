#pragma once

/// \file xpool/devkit/fabric_observer.hpp
/// \brief Process-local Fabric observation records and Host readout.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include <cuda/std/variant>

#include <xpool/abort.hpp>
#include <xpool/fabric/hooks.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/trace.hpp>

namespace xpool::devkit::fabric_observer {

/// Runtime role that owns one Fabric trace record.
enum class RecordKind : std::uint32_t {
  AtnAgent = 1,
  Coordinator = 2,
  FfnAgent = 3,
};

/// One retained semantic Fabric trace with role-specific private state.
class alignas(8) Record {
public:
  /// Return the PE-local monotonically assigned trace identifier.
  std::uint64_t local_trace_id() const { return local_trace_id_; }
  /// Return the invocation key observed by this role.
  const xpool::fabric::InvocationKey &key() const { return key_; }
  /// Return the model-local layer ordinal.
  std::size_t layer_ordinal() const { return layer_ordinal_; }
  /// Return the number of live payload rows.
  std::size_t payload_rows() const { return payload_rows_; }
  /// Return the number of AtnAgent outputs requested by the invocation.
  xpool::ffn::OutputRequirement output_requirement() const { return output_requirement_; }

  /// Return the runtime role that produced this record.
  RecordKind kind() const {
    if (cuda::std::holds_alternative<AtnAgentTraceState>(state_)) {
      return RecordKind::AtnAgent;
    }
    if (cuda::std::holds_alternative<CoordinatorTraceState>(state_)) {
      return RecordKind::Coordinator;
    }
    if (cuda::std::holds_alternative<FfnAgentTraceState>(state_)) {
      return RecordKind::FfnAgent;
    }
    xpool::abort();
  }

  /// Return the AtnAgent's DP-local row count when applicable.
  std::optional<std::size_t> dp_rank_payload_rows() const {
    const auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_);
    return state == nullptr ? std::nullopt : std::optional<std::size_t>{state->dp_rank_payload_rows};
  }

  /// Return the AtnAgent forward mode when applicable.
  std::optional<xpool::ffn::ForwardMode> forward_mode() const {
    const auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_);
    return state == nullptr ? std::nullopt : std::optional{state->forward_mode};
  }

  /// Return the AtnAgent DP row layout when applicable.
  std::optional<xpool::ffn::DpRowLayout> dp_row_layout() const {
    const auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_);
    return state == nullptr ? std::nullopt : std::optional{state->dp_row_layout};
  }

  /// Return the lane once the role has observed or chosen one.
  /// \return The executor lane index, or no value before assignment.
  std::optional<std::size_t> executor_lane_index() const {
    if (const auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_)) {
      return state->timeline.recorded(xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved)
                 ? std::optional<std::size_t>{state->executor_lane_index}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<CoordinatorTraceState>(&state_)) {
      return state->timeline.recorded(xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Scheduled)
                 ? std::optional<std::size_t>{state->executor_lane_index}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<FfnAgentTraceState>(&state_)) {
      return state->executor_lane_index;
    }
    return std::nullopt;
  }

  /// Return the lane lease sequence once the role has observed or chosen one.
  /// \return The lease sequence, or no value before assignment.
  std::optional<std::uint64_t> executor_lease_sequence() const {
    if (const auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_)) {
      return state->timeline.recorded(xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved)
                 ? std::optional<std::uint64_t>{state->executor_lease_sequence}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<CoordinatorTraceState>(&state_)) {
      return state->timeline.recorded(xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Scheduled)
                 ? std::optional<std::uint64_t>{state->executor_lease_sequence}
                 : std::nullopt;
    }
    if (const auto *state = cuda::std::get_if<FfnAgentTraceState>(&state_)) {
      return state->executor_lease_sequence;
    }
    return std::nullopt;
  }

  /// Return the coordinator ready ticket after enqueue.
  std::optional<std::uint64_t> ready_ticket() const {
    const auto *state = cuda::std::get_if<CoordinatorTraceState>(&state_);
    return state == nullptr || state->ready_ticket == 0 ? std::nullopt
                                                        : std::optional<std::uint64_t>{state->ready_ticket};
  }

  /// Return the FfnAgent's observed row capacity.
  std::optional<std::size_t> payload_row_capacity() const {
    const auto *state = cuda::std::get_if<FfnAgentTraceState>(&state_);
    return state == nullptr ? std::nullopt : std::optional<std::size_t>{state->payload_row_capacity};
  }

  /// Return the FfnAgent's observed delivery variant.
  std::optional<xpool::fabric::DeliveryVariant> delivery() const {
    const auto *state = cuda::std::get_if<FfnAgentTraceState>(&state_);
    if (state == nullptr) {
      return std::nullopt;
    }
    return state->delivery;
  }

  /// Test whether an AtnAgent event was recorded.
  bool recorded(xpool::hooks::FabricAtnAgentProtocolEvent::Kind event) const {
    return require_atnagent().timeline.recorded(event);
  }
  /// Test whether a Coordinator event was recorded.
  bool recorded(xpool::hooks::FabricCoordinatorProtocolEvent::Kind event) const {
    return require_coordinator().timeline.recorded(event);
  }
  /// Test whether an FfnAgent event was recorded.
  bool recorded(xpool::hooks::FabricFfnAgentProtocolEvent::Kind event) const {
    return require_ffnagent().timeline.recorded(event);
  }

  /// Return an AtnAgent event timestamp in GPU global-timer nanoseconds.
  /// Timestamps are comparable only inside the same GPU clock domain.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(xpool::hooks::FabricAtnAgentProtocolEvent::Kind event) const {
    return require_atnagent().timeline.timestamp(event);
  }
  /// Return a Coordinator event timestamp in GPU global-timer nanoseconds.
  /// Timestamps are comparable only inside the same GPU clock domain.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(xpool::hooks::FabricCoordinatorProtocolEvent::Kind event) const {
    return require_coordinator().timeline.timestamp(event);
  }
  /// Return an FfnAgent event timestamp in GPU global-timer nanoseconds.
  /// Timestamps are comparable only inside the same GPU clock domain.
  XPOOL_HOST_DEVICE_FN std::uint64_t timestamp(xpool::hooks::FabricFfnAgentProtocolEvent::Kind event) const {
    return require_ffnagent().timeline.timestamp(event);
  }

#if defined(__CUDACC__)
  /// Initialize an AtnAgent trace from a submission.
  XPOOL_DEVICE_FN void begin_atnagent(std::uint64_t local_trace_id, const xpool::fabric::Submission &submission);
  /// Record that the AtnAgent published its submission.
  XPOOL_DEVICE_FN void submission_published();
  /// Record the admission observed by the AtnAgent.
  XPOOL_DEVICE_FN void admission_observed(const xpool::fabric::Admission &admission);
  /// Record that the AtnAgent published input readiness.
  XPOOL_DEVICE_FN void input_ready_published();
  /// Record the output commit observed by the AtnAgent.
  XPOOL_DEVICE_FN void output_commit_observed(const xpool::fabric::OutputCommit &commit);
  /// Record that the AtnAgent published its output acknowledgement.
  XPOOL_DEVICE_FN void output_acknowledgement_published();

  /// Initialize a Coordinator trace from a ready invocation.
  XPOOL_DEVICE_FN void begin_coordinator(std::uint64_t local_trace_id, const xpool::fabric::Invocation &invocation);
  /// Record scheduler enqueue and its FIFO ticket.
  XPOOL_DEVICE_FN void enqueued(std::uint64_t ready_ticket);
  /// Record the scheduler's lane decision.
  XPOOL_DEVICE_FN void scheduled(std::size_t executor_lane_index, std::uint64_t executor_lease_sequence);
  /// Record that the Coordinator published admission.
  XPOOL_DEVICE_FN void admission_published();
  /// Record that the Coordinator published lane execution.
  XPOOL_DEVICE_FN void lane_execution_published();
  /// Record that all assigned FfnAgents completed.
  XPOOL_DEVICE_FN void ffnagent_completions_observed();
  /// Record that the Coordinator published output commit.
  XPOOL_DEVICE_FN void output_commit_published();
  /// Record that all required output acknowledgements arrived.
  XPOOL_DEVICE_FN void output_acknowledgements_observed();
  /// Record that the Coordinator released the lane lease.
  XPOOL_DEVICE_FN void lane_released();

  /// Initialize an FfnAgent trace from a lane execution.
  XPOOL_DEVICE_FN void begin_ffnagent(std::uint64_t local_trace_id, const xpool::fabric::LaneExecution &execution,
                                      std::size_t executor_lane_index, std::size_t payload_row_capacity,
                                      xpool::fabric::DeliveryVariant delivery);
  /// Record that the FfnAgent observed input readiness.
  XPOOL_DEVICE_FN void input_ready_observed();
  /// Record that the routing owner published routing metadata.
  XPOOL_DEVICE_FN void routing_metadata_published();
  /// Record that this FfnAgent observed routing metadata.
  XPOOL_DEVICE_FN void routing_metadata_observed();
  /// Record the start of FFN computation.
  XPOOL_DEVICE_FN void compute_started();
  /// Record the completion of FFN computation.
  XPOOL_DEVICE_FN void compute_completed();
  /// Record that this FfnAgent published partial readiness.
  XPOOL_DEVICE_FN void partial_ready_published();
  /// Record that peer partials became ready.
  XPOOL_DEVICE_FN void peer_partials_ready_observed();
  /// Record that this FfnAgent published completion.
  XPOOL_DEVICE_FN void completion_published();
#endif

private:
  struct AtnAgentTraceState {
    std::size_t dp_rank_payload_rows = 0;
    xpool::ffn::ForwardMode forward_mode = xpool::ffn::ForwardMode::Idle;
    xpool::ffn::DpRowLayout dp_row_layout = xpool::ffn::DpRowLayout::None;
    std::size_t executor_lane_index = 0;
    std::uint64_t executor_lease_sequence = 0;
    xpool::utils::trace::Timeline<xpool::hooks::FabricAtnAgentProtocolEvent::Kind> timeline;
  };

  struct CoordinatorTraceState {
    std::size_t executor_lane_index = 0;
    std::uint64_t executor_lease_sequence = 0;
    std::uint64_t ready_ticket = 0;
    xpool::utils::trace::Timeline<xpool::hooks::FabricCoordinatorProtocolEvent::Kind> timeline;
  };

  struct FfnAgentTraceState {
    std::size_t executor_lane_index = 0;
    std::uint64_t executor_lease_sequence = 0;
    std::size_t payload_row_capacity = 0;
    xpool::fabric::DeliveryVariant delivery = xpool::fabric::DeliveryVariant::DirectPartial;
    xpool::utils::trace::Timeline<xpool::hooks::FabricFfnAgentProtocolEvent::Kind> timeline;
  };

  using State = cuda::std::variant<cuda::std::monostate, AtnAgentTraceState, CoordinatorTraceState, FfnAgentTraceState>;

#if defined(__CUDACC__)
  XPOOL_DEVICE_FN void begin(std::uint64_t local_trace_id, const xpool::fabric::InvocationKey &key,
                             std::size_t layer_ordinal, std::size_t payload_rows,
                             xpool::ffn::OutputRequirement output_requirement);
#endif

  XPOOL_HOST_DEVICE_FN AtnAgentTraceState &require_atnagent() {
    auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }
  XPOOL_HOST_DEVICE_FN const AtnAgentTraceState &require_atnagent() const {
    const auto *state = cuda::std::get_if<AtnAgentTraceState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }
  XPOOL_HOST_DEVICE_FN CoordinatorTraceState &require_coordinator() {
    auto *state = cuda::std::get_if<CoordinatorTraceState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }
  XPOOL_HOST_DEVICE_FN const CoordinatorTraceState &require_coordinator() const {
    const auto *state = cuda::std::get_if<CoordinatorTraceState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }
  XPOOL_HOST_DEVICE_FN FfnAgentTraceState &require_ffnagent() {
    auto *state = cuda::std::get_if<FfnAgentTraceState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }
  XPOOL_HOST_DEVICE_FN const FfnAgentTraceState &require_ffnagent() const {
    const auto *state = cuda::std::get_if<FfnAgentTraceState>(&state_);
    xpool::abort_if(state == nullptr);
    return *state;
  }

  std::uint64_t local_trace_id_ = 0;
  xpool::fabric::InvocationKey key_{};
  std::size_t layer_ordinal_ = 0;
  std::size_t payload_rows_ = 0;
  xpool::ffn::OutputRequirement output_requirement_ = xpool::ffn::OutputRequirement::PerRankComplete;
  State state_{};
};

/// Static Instance topology retained beside one PE-local snapshot.
struct ModelTopology {
  /// Attention tensor-parallel participant count.
  std::size_t atn_tp_size;
  /// Attention data-parallel participant count.
  std::size_t atn_dp_size;
};

/// Host-owned snapshot copied from one PE-local Fabric trace buffer.
struct Snapshot {
  /// NVSHMEM PE that produced this process-local snapshot.
  int pe;
  /// Joined AtnAgent count.
  std::size_t atnagent_count;
  /// Joined FfnAgent count.
  std::size_t ffnagent_count;
  /// Static topology for each model instance.
  std::vector<ModelTopology> model_topologies;
  /// Total number of records reserved since observer installation.
  std::uint64_t sequence;
  /// Number of records not retained after capacity was exhausted.
  std::uint64_t dropped;
  /// Retained records in reservation order.
  std::vector<Record> records;
};

static_assert(std::is_trivially_copyable_v<ModelTopology>);
static_assert(std::is_trivially_copyable_v<Record>);

/// Return the current process-local Fabric snapshot when observation is enabled.
/// \pre The related Fabric participant is drained or otherwise synchronized.
/// \return A Host-owned snapshot, or no value before installation or after finalization.
/// \throws c10::Error when CUDA copy or retained observer state is invalid.
std::optional<Snapshot> read();

/// Return the process-local Fabric Observer allocation size.
/// \param instance_count Positive number of Fabric Instances.
/// \param executor_lane_count Positive number of executor Lanes.
/// \return Required bytes under current Debug Options, or zero when disabled.
/// \throws c10::Error when geometry is invalid or size arithmetic overflows.
std::size_t allocation_bytes(std::size_t instance_count, std::size_t executor_lane_count);

} // namespace xpool::devkit::fabric_observer
