#include <xpool/fabric/atnagent.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cooperative_groups.h>
#include <nvshmem.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/protocol.cuh>
#include <xpool/ffn.hpp>
#include <xpool/hooks.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/cooperative.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::fabric::atnagent {

namespace {

struct InvocationContext {
  std::size_t atnagent_index = 0;
  std::size_t atnagent_rank = 0;
  std::size_t instance_index = 0;
  std::size_t layer_ordinal = 0;
  std::size_t payload_rows = 0;
  std::size_t dp_rank_payload_rows = 0;
  std::size_t payload_bytes = 0;
  xpool::ffn::ForwardMode forward_mode = xpool::ffn::ForwardMode::Idle;
  xpool::ffn::OutputRequirement output_requirement = xpool::ffn::OutputRequirement::PerRankComplete;
  xpool::ffn::DpRowLayout dp_row_layout = xpool::ffn::DpRowLayout::None;
  xpool::ffn::ResultCode result_code = xpool::ffn::ResultCode::ProtocolMismatch;
};

XPOOL_DEVICE_FN xpool::ffn::ResultCode wait_result_code(const ArenaView &arena, xpool::utils::wait::Status result) {
  switch (result) {
  case xpool::utils::wait::Status::Pending:
    break;
  case xpool::utils::wait::Status::Ready:
    return xpool::ffn::ResultCode::Ok;
  case xpool::utils::wait::Status::Cancelled:
    return arena.cancellation_result();
  case xpool::utils::wait::Status::TimedOut:
    return xpool::ffn::ResultCode::Timeout;
  }
  xpool::abort();
}

XPOOL_DEVICE_FN InvocationContext resolve_context(const ArenaView &arena,
                                                  const xpool::transport::ArenaView &transport_arena) {
  auto context = InvocationContext{};
  const auto &layout = arena.layout();
  const auto &transport_layout = transport_arena.layout();
  const auto &mailbox = transport_arena.mailbox();
  const auto &request = mailbox.request;
  const auto pe = nvshmem_my_pe();

  xpool::abort_if(pe < 0 || static_cast<std::size_t>(pe) >= layout.atnagent_count ||
                  transport_layout.instance_index >= layout.instance_count);
  context.atnagent_index = static_cast<std::size_t>(pe);
  context.instance_index = transport_layout.instance_index;
  context.layer_ordinal = request.layer_ordinal;
  context.payload_rows = mailbox.payload_rows;
  context.forward_mode = request.forward_mode;
  context.output_requirement = request.output_requirement;
  context.dp_row_layout = request.dp_row_layout;

  const auto &instance = arena.instance_entry(context.instance_index);
  const auto atnagent_pes = arena.atnagent_pes(context.instance_index);
  xpool::abort_if(instance.atn_tp_size != transport_layout.atn_tp_size ||
                  instance.atn_dp_size != transport_layout.atn_dp_size ||
                  instance.payload_dtype != transport_layout.payload_dtype ||
                  instance.hidden_size != transport_layout.hidden_size ||
                  instance.payload_row_bytes != transport_layout.payload_row_bytes);
  context.atnagent_rank = instance.atnagent_index(transport_layout.atn_tp_rank, transport_layout.atn_dp_rank);
  xpool::abort_if(context.atnagent_rank >= atnagent_pes.size() || atnagent_pes[context.atnagent_rank] != pe);

  if (context.layer_ordinal >= instance.layer_count) {
    return context;
  }
  if (context.payload_rows > std::numeric_limits<std::size_t>::max() / instance.payload_row_bytes) {
    return context;
  }
  context.payload_bytes = context.payload_rows * instance.payload_row_bytes;
  if (context.payload_bytes > layout.lane_payload_capacity_bytes) {
    return context;
  }

  const auto rank_payload_rows = transport_arena.dp_rank_payload_rows();
  context.dp_rank_payload_rows =
      instance.atn_dp_size == 1 ? context.payload_rows : rank_payload_rows[transport_layout.atn_dp_rank];
  if (context.dp_rank_payload_rows > context.payload_rows) {
    return context;
  }
  context.result_code = xpool::ffn::ResultCode::Ok;
  return context;
}

XPOOL_DEVICE_FN xpool::ffn::ResultCode converge_failure(const ArenaView &arena,
                                                        const cooperative_groups::thread_block &group,
                                                        const InvocationKey &key, std::size_t layer_ordinal,
                                                        xpool::ffn::ResultCode candidate) {
  XPOOL_DEVICE_SHARED xpool::ffn::ResultCode canonical_result;
  if (group.thread_rank() == 0) {
    arena.state().failure.try_publish(arena.layout().coordinator_pe(), candidate, key, layer_ordinal);
  }
  group.sync();
  if (group.thread_rank() == 0) {
    const auto converged =
        xpool::utils::wait::until(xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds),
                                  [&] { return arena.state().failure.published(); });
    xpool::abort_if(converged != xpool::utils::wait::Status::Ready);
    canonical_result = arena.state().failure.payload.result_code;
  }
  group.sync();
  return canonical_result;
}

} // namespace

XPOOL_DEVICE_FN xpool::ffn::ResultCode execute(const ArenaView &arena,
                                               const xpool::transport::ArenaView &transport_arena) {
  const auto group = cooperative_groups::this_thread_block();
  XPOOL_DEVICE_SHARED InvocationContext context;
  XPOOL_DEVICE_SHARED std::uint64_t invocation_sequence;
  XPOOL_DEVICE_SHARED xpool::ffn::ResultCode result_value;

  if (group.thread_rank() == 0) {
    context = resolve_context(arena, transport_arena);
    auto &submission = arena.submission_publication(context.atnagent_index, context.instance_index);
    const auto previous_sequence = submission.record.key.invocation_sequence;
    xpool::abort_if(previous_sequence == std::numeric_limits<std::uint64_t>::max());
    invocation_sequence = previous_sequence + 1;
    result_value = context.result_code;
  }
  group.sync();

  const auto key = InvocationKey{
      .instance_index = context.instance_index,
      .invocation_sequence = invocation_sequence,
  };
  if (result_value != xpool::ffn::ResultCode::Ok) {
    return result_value;
  }

  // Phase: Submit - This rank publishes one contribution; the Coordinator
  // admits the invocation only after every AtnAgent publication agrees.
  auto &submission = arena.submission_publication(context.atnagent_index, context.instance_index);
  if (group.thread_rank() == 0) {
    submission.record = Submission{
        .key = key,
        .layer_ordinal = context.layer_ordinal,
        .payload_rows = context.payload_rows,
        .dp_row_layout = context.dp_row_layout,
        .output_requirement = context.output_requirement,
        .dp_rank_payload_rows = context.dp_rank_payload_rows,
        .forward_mode = context.forward_mode,
    };
    xpool::hooks::FabricAtnAgentProtocolEvent::hooks(
        {.arena = arena,
         .instance_index = context.instance_index,
         .kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPrepared,
         .submission = &submission.record});
  }
  group.sync();
  submission.publish_record(group, arena.layout().coordinator_pe());
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricAtnAgentProtocolEvent::hooks(
        {.arena = arena,
         .instance_index = context.instance_index,
         .kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind::SubmissionPublished});
  }

  // Phase: Await Admission - One validated Admission fixes the Lane and lease
  // used by every subsequent publication for this invocation.
  const auto deadline = xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds);
  auto &admission_publication = arena.admission_publication(context.instance_index);
  if (group.thread_rank() == 0) {
    const auto wait_result = xpool::utils::wait::until(
        deadline, [&] { return admission_publication.test_at_least(invocation_sequence); },
        [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
    result_value = wait_result_code(arena, wait_result);
    if (result_value == xpool::ffn::ResultCode::Ok &&
        (admission_publication.validate_expected(invocation_sequence, key) != xpool::ffn::ResultCode::Ok ||
         admission_publication.record.executor_lane_index >= arena.layout().executor_lane_count)) {
      result_value = xpool::ffn::ResultCode::ProtocolMismatch;
    }
    if (result_value == xpool::ffn::ResultCode::Ok) {
      xpool::hooks::FabricAtnAgentProtocolEvent::hooks(
          {.arena = arena,
           .instance_index = context.instance_index,
           .kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind::AdmissionObserved,
           .admission = &admission_publication.record});
    }
  }
  group.sync();
  if (result_value != xpool::ffn::ResultCode::Ok) {
    if (result_value == xpool::ffn::ResultCode::ProtocolMismatch || result_value == xpool::ffn::ResultCode::Timeout) {
      return converge_failure(arena, group, key, context.layer_ordinal, result_value);
    }
    return result_value;
  }

  const auto admission = admission_publication.record;
  const auto &instance = arena.instance_entry(context.instance_index);
  const auto ffnagent_pes = arena.ffnagent_pes(context.instance_index, context.layer_ordinal);
  const auto output_group_size = instance.atn_tp_size * instance.atn_dp_size;
  // Phase: Publish Input - Paired ranks make their payload remotely visible
  // before InputReady; modulo mapping avoids a rank-zero gather when F > A.
  if (context.atnagent_rank < ffnagent_pes.size()) {
    auto lane_payload = arena.atnagent_lane_payload(admission.executor_lane_index);
    auto payload = lane_payload.input_destination().first(context.payload_bytes);
    xpool::utils::cooperative::copy(group, payload, transport_arena.input_payload().first(context.payload_bytes));
    auto &input_ready = arena.input_ready_publication(admission.executor_lane_index);
    if (group.thread_rank() == 0) {
      input_ready.record = InputReady{
          .key = key,
          .executor_lease_sequence = admission.executor_lease_sequence,
      };
    }
    group.sync();
    for (auto rank = std::size_t{0}; rank < ffnagent_pes.size(); ++rank) {
      if (rank % output_group_size == context.atnagent_rank) {
        input_ready.publish_payload(group, ffnagent_pes[rank], payload);
      }
    }
    if (group.thread_rank() == 0) {
      xpool::hooks::FabricAtnAgentProtocolEvent::hooks(
          {.arena = arena,
           .instance_index = context.instance_index,
           .kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind::InputReadyPublished});
    }
  }

  // Phase: Await Commit - OutputCommit proves that the Coordinator observed
  // every participating FfnAgent completion for this invocation.
  auto &commit = arena.output_commit_publication(context.instance_index);
  if (group.thread_rank() == 0) {
    const auto wait_result = xpool::utils::wait::until(
        deadline, [&] { return commit.test_at_least(invocation_sequence); },
        [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
    result_value = wait_result_code(arena, wait_result);
    if (result_value == xpool::ffn::ResultCode::Ok &&
        commit.validate_expected(invocation_sequence, key) != xpool::ffn::ResultCode::Ok) {
      result_value = xpool::ffn::ResultCode::ProtocolMismatch;
    }
    if (result_value == xpool::ffn::ResultCode::Ok) {
      xpool::hooks::FabricAtnAgentProtocolEvent::hooks(
          {.arena = arena,
           .instance_index = context.instance_index,
           .kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputCommitObserved,
           .commit = &commit.record});
    }
  }
  group.sync();
  if (result_value != xpool::ffn::ResultCode::Ok) {
    if (result_value == xpool::ffn::ResultCode::ProtocolMismatch || result_value == xpool::ffn::ResultCode::Timeout) {
      return converge_failure(arena, group, key, context.layer_ordinal, result_value);
    }
    return result_value;
  }

  // Phase: Materialize Output - Each rank copies its admitted delivery view or
  // deterministically zero-fills the complete Transport result it still owns.
  const auto delivery = delivery_variant(instance, context.output_requirement);
  const auto receives_output =
      delivery == DeliveryVariant::ReplicatedComplete ||
      (delivery == DeliveryVariant::DirectPartial && context.atnagent_rank < instance.ffn_tp_size) ||
      (delivery == DeliveryVariant::SingleComplete && context.atnagent_rank == 0);
  if (receives_output) {
    const auto output =
        arena.atnagent_lane_payload(admission.executor_lane_index).output().first(context.payload_bytes);
    xpool::utils::cooperative::copy(group, transport_arena.output_payload().first(context.payload_bytes), output);
  } else {
    // Non-receivers still own a complete Transport result buffer and publish a
    // deterministic zero result to their local caller.
    xpool::utils::cooperative::fill(group, transport_arena.output_payload().first(context.payload_bytes),
                                    std::uint8_t{0});
  }
  // Phase: Acknowledge - Publish only after the local Transport result is
  // complete; the Coordinator cannot release the Lane before every rank does so.
  auto &acknowledgement = arena.output_acknowledgement_publication(context.atnagent_index, context.instance_index);
  if (group.thread_rank() == 0) {
    acknowledgement.record = OutputAcknowledgement{.key = key};
  }
  group.sync();
  acknowledgement.publish_record(group, arena.layout().coordinator_pe());
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricAtnAgentProtocolEvent::hooks(
        {.arena = arena,
         .instance_index = context.instance_index,
         .kind = xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputAcknowledgementPublished});
  }
  return xpool::ffn::ResultCode::Ok;
}

} // namespace xpool::fabric::atnagent
