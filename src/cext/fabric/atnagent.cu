#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <nvshmem.h>

#include <xpool/abi.hpp>
#include <xpool/abort.hpp>
#include <xpool/atomic.cuh>
#include <xpool/fabric/atnagent.cuh>
#include <xpool/fabric/protocol.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/cooperative.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::fabric::atnagent {

namespace {

struct InvocationContext {
  int atnagent_index = -1;
  std::size_t model_index = 0;
  std::size_t layer_ordinal = 0;
  std::size_t layer_id = 0;
  std::size_t payload_rows = 0;
  std::size_t local_token_count = 0;
  std::size_t payload_bytes = 0;
  std::uint32_t forward_mode = 0;
  std::uint32_t result_handoff = 0;
  std::uint32_t dp_padding_mode = 0;
  xpool::abi::FfnResultCode result_code{};

  XPOOL_DEVICE_FN bool input_publisher() const { return atnagent_index == 0; }
};

XPOOL_DEVICE_FN xpool::abi::FfnResultCode cancellation_result(const FabricArenaView &arena) {
  const auto &failure = arena.state().failure;
  if (failure.published()) {
    return xpool::abi::FfnResultCode{failure.payload.result_code};
  }
  return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Shutdown};
}

XPOOL_DEVICE_FN xpool::abi::FfnResultCode wait_result_code(
    const FabricArenaView &arena, xpool::utils::wait::Result result) {
  switch (result) {
  case xpool::utils::wait::Result::Ready:
    return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
  case xpool::utils::wait::Result::Cancelled:
    return cancellation_result(arena);
  case xpool::utils::wait::Result::TimedOut:
    return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Timeout};
  }
  xpool::abort();
}

/// Derive one rank-local invocation context from Transport and Fabric geometry.
/// No Fabric-visible state is written until result_code becomes Ok.
XPOOL_DEVICE_FN InvocationContext resolve_context(
    const FabricArenaView &arena,
    const xpool::transport::TransportArenaView &transport_arena) {
  auto context = InvocationContext{};
  const auto &layout = arena.layout();
  const auto &transport_layout = transport_arena.layout();
  const auto &mailbox = transport_arena.mailbox();
  const auto &request = mailbox.request;

  context.atnagent_index = nvshmem_my_pe();
  context.model_index = transport_layout.instance_index;
  context.layer_ordinal = request.layer_ordinal;
  context.payload_rows = mailbox.payload_rows;
  context.forward_mode = request.forward_mode;
  context.result_handoff = request.result_handoff;
  context.dp_padding_mode = request.dp_padding_mode;

  xpool::abort_if(context.atnagent_index < 0 ||
                  static_cast<std::size_t>(context.atnagent_index) >= layout.atnagent_count ||
                  context.model_index >= layout.model_count || !request.valid() || context.payload_rows == 0 ||
                  context.payload_rows > transport_layout.max_tokens);

  const auto &model = arena.model_layout(context.model_index);
  xpool::abort_if(model.atn_tp_size * model.atn_dp_size != layout.atnagent_count ||
                  model.atn_tp_size != transport_layout.atn_tp_size ||
                  model.atn_dp_size != transport_layout.atn_dp_size ||
                  model.atnagent_index(transport_layout.atn_tp_rank, transport_layout.atn_dp_rank) !=
                      static_cast<std::size_t>(context.atnagent_index) ||
                  model.dtype != transport_layout.dtype || model.hidden_size != transport_layout.hidden_size);

  if (context.layer_ordinal >= model.layer_count) {
    return context;
  }
  context.layer_id = arena.layer_layout(model.layer_begin + context.layer_ordinal).layer_id;
  const auto element_bytes = xpool::abi::TensorDType{model.dtype}.bytes();
  if (context.payload_rows > std::numeric_limits<std::size_t>::max() / model.hidden_size / element_bytes) {
    return context;
  }
  context.payload_bytes = context.payload_rows * model.hidden_size * element_bytes;

  const auto *token_counts = transport_arena.dp_token_counts();
  if (model.atn_dp_size == 1) {
    if (token_counts != nullptr || request.dp_padding_mode != xpool::abi::DpPaddingMode::None) {
      return context;
    }
    context.local_token_count = context.payload_rows;
  } else {
    if (token_counts == nullptr || request.dp_padding_mode == xpool::abi::DpPaddingMode::None) {
      return context;
    }
    context.local_token_count = token_counts[transport_layout.atn_dp_rank];
    if (context.local_token_count > context.payload_rows) {
      return context;
    }
  }
  context.result_code = xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
  return context;
}

/// Publish one terminal result to the local Transport mailbox after every
/// participating thread has finished reading or writing its payload.
XPOOL_DEVICE_FN void publish_transport_result(
    const cooperative_groups::thread_block_tile<32> &group,
    const xpool::transport::TransportArenaView &transport_arena,
    xpool::transport::TransportTraceRecord *transport_trace,
    xpool::abi::FfnResultCode result_code) {
  if (group.thread_rank() == 0) {
    if (transport_trace != nullptr) {
      transport_trace->execution_completed();
      transport_trace->evaluated(result_code);
    }
    transport_arena.mailbox().publish_result(result_code);
  }
  group.sync();
}

/// Attempt canonical Fabric failure publication and publish this request's
/// terminal error to its local Transport mailbox.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode publish_protocol_failure(
    const FabricArenaView &arena,
    const xpool::transport::TransportArenaView &transport_arena,
    xpool::transport::TransportTraceRecord *transport_trace,
    const cooperative_groups::thread_block_tile<32> &group,
    const FfnInvocationKey &key,
    std::size_t layer_ordinal,
    xpool::abi::FfnResultCode result_code) {
  if (group.thread_rank() == 0) {
    arena.state().failure.try_publish(arena.layout().coordinator_pe(), result_code, key, layer_ordinal);
  }
  group.sync();
  publish_transport_result(group, transport_arena, transport_trace, result_code);
  return result_code;
}

} // namespace

/// Execute one validated Transport request through the distributed Fabric
/// submission, admission, payload, result, and acknowledgement protocol.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode execute(
    const FabricArenaView &arena,
    const xpool::transport::TransportArenaView &transport_arena,
    xpool::transport::TransportTraceRecord *transport_trace) {
  const auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

  __shared__ InvocationContext context;
  __shared__ std::uint64_t invocation_sequence;
  __shared__ std::uint32_t result_value;
  __shared__ FabricTraceRecord *fabric_trace;

  // Phase: Resolve - Derive and validate rank-local request facts before publishing
  // any Fabric-visible state.
  if (group.thread_rank() == 0) {
    context = resolve_context(arena, transport_arena);
    auto &submission = arena.submission_publication(
        static_cast<std::size_t>(context.atnagent_index), context.model_index);
    const auto previous_sequence = submission.record.key.invocation_sequence;
    xpool::abort_if(previous_sequence == std::numeric_limits<std::uint64_t>::max());
    invocation_sequence = previous_sequence + 1;
    result_value = context.result_code.value();
    fabric_trace = nullptr;
  }
  group.sync();

  const auto key = FfnInvocationKey{
      .model_index = context.model_index,
      .invocation_sequence = invocation_sequence,
  };
  if (result_value != xpool::abi::FfnResultCode::Ok) {
    return publish_protocol_failure(
        arena, transport_arena, transport_trace, group, key, context.layer_ordinal,
        xpool::abi::FfnResultCode{result_value});
  }

  auto &submission_publication = arena.submission_publication(
      static_cast<std::size_t>(context.atnagent_index), context.model_index);
  // PE 0 is the canonical Input Publisher for the all-AtnAgent invocation.
  // This contextual role controls payload movement here; trace records retain
  // only intrinsic event ordering and do not duplicate the derived role.
  const auto input_publisher = context.input_publisher();
  // Phase: Submit - Materialize the immutable Submission and bind its trace before
  // Decode payload or publication can become visible.
  if (group.thread_rank() == 0) {
    submission_publication.record = FfnSubmission{
        .key = key,
        .layer_ordinal = context.layer_ordinal,
        .payload_rows = context.payload_rows,
        .local_token_count = context.local_token_count,
        .forward_mode = context.forward_mode,
        .result_handoff = context.result_handoff,
        .dp_padding_mode = context.dp_padding_mode,
    };
    const auto trace_entry = arena.reserve_trace();
    if (trace_entry) {
      fabric_trace = &trace_entry.record();
      fabric_trace->begin_atnagent(trace_entry.sequence(), submission_publication.record, context.layer_id);
    }
  }
  group.sync();

  const auto &model = arena.model_layout(context.model_index);
  if (input_publisher && context.forward_mode != xpool::abi::XPoolForwardMode::Extend) {
    if (context.payload_bytes > model.decode_payload_capacity_bytes) {
      return publish_protocol_failure(
          arena, transport_arena, transport_trace, group, key, context.layer_ordinal,
          xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch});
    }
    xpool::utils::cooperative::copy(
        group, arena.model_input_payload(context.model_index), transport_arena.input_payload(), context.payload_bytes);
    if (group.thread_rank() == 0 && fabric_trace != nullptr) {
      fabric_trace->decode_input_staged();
    }
  }

  submission_publication.publish(group, invocation_sequence, arena.layout().coordinator_pe());
  if (group.thread_rank() == 0 && fabric_trace != nullptr) {
    fabric_trace->submission_published();
  }

  // Phase: Admit - Observe the Coordinator's matching Executor lease and execution
  // mode before entering mode-specific transfer.
  const auto deadline =
      xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds);
  auto &admission_publication = arena.admission_publication(
      static_cast<std::size_t>(context.atnagent_index), context.model_index);
  if (group.thread_rank() == 0) {
    const auto wait_result = xpool::utils::wait::until(
        deadline,
        [&] { return admission_publication.observe() >= invocation_sequence; },
        [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
    result_value = wait_result_code(arena, wait_result).value();
    if (result_value == xpool::abi::FfnResultCode::Ok &&
        (admission_publication.observe() != invocation_sequence ||
         admission_publication.validate() != xpool::abi::FfnResultCode::Ok ||
         admission_publication.record.key != key ||
         admission_publication.record.executor_index >= arena.layout().executor_count)) {
      result_value = xpool::abi::FfnResultCode::ProtocolMismatch;
    }
    if (result_value == xpool::abi::FfnResultCode::Ok) {
      if (transport_trace != nullptr) {
        transport_trace->execution_admitted();
      }
      if (fabric_trace != nullptr) {
        fabric_trace->admission_observed(admission_publication.record);
      }
    }
  }
  group.sync();
  if (result_value != xpool::abi::FfnResultCode::Ok) {
    const auto result_code = xpool::abi::FfnResultCode{result_value};
    if (result_code == xpool::abi::FfnResultCode::ProtocolMismatch) {
      return publish_protocol_failure(
          arena, transport_arena, transport_trace, group, key, context.layer_ordinal, result_code);
    }
    publish_transport_result(group, transport_arena, transport_trace, result_code);
    return result_code;
  }

  const auto admission = admission_publication.record;
  const auto prefill = admission.execution_mode == FfnExecutionMode::Prefill;
  // Phase: Prefill staging - Only the Input Publisher pushes payload into the leased
  // Executor region and then publishes InputReady to every FfnAgent.
  if (input_publisher && prefill) {
    auto *input = arena.executor_input_payload(admission.executor_index);
    xpool::utils::cooperative::copy(group, input, transport_arena.input_payload(), context.payload_bytes);
    if (group.thread_rank() == 0 && fabric_trace != nullptr) {
      fabric_trace->prefill_input_staged();
    }

    auto &input_ready = arena.input_ready_publication(admission.executor_index);
    if (group.thread_rank() == 0) {
      input_ready.record = FfnInputReady{.key = key};
    }
    group.sync();
    const auto &layout = arena.layout();
    for (auto ffnagent_index = std::size_t{0}; ffnagent_index < layout.ffnagent_count;
         ++ffnagent_index) {
      input_ready.publish(
          group, invocation_sequence,
          static_cast<int>(layout.atnagent_count + ffnagent_index), input, context.payload_bytes);
    }
    if (group.thread_rank() == 0 && fabric_trace != nullptr) {
      fabric_trace->prefill_input_published();
    }
  }

  // Phase: Evaluate - Observe the matching distributed Result before materializing a
  // rank-local Transport output.
  auto &result_publication = arena.result_publication(
      static_cast<std::size_t>(context.atnagent_index), context.model_index);
  if (group.thread_rank() == 0) {
    const auto wait_result = xpool::utils::wait::until(
        deadline,
        [&] { return result_publication.observe() >= invocation_sequence; },
        [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
    result_value = wait_result_code(arena, wait_result).value();
    if (result_value == xpool::abi::FfnResultCode::Ok &&
        (result_publication.observe() != invocation_sequence ||
         result_publication.validate() != xpool::abi::FfnResultCode::Ok ||
         result_publication.record.key != key)) {
      result_value = xpool::abi::FfnResultCode::ProtocolMismatch;
    }
    if (result_value == xpool::abi::FfnResultCode::Ok && fabric_trace != nullptr) {
      fabric_trace->result_observed(result_publication.record);
    }
  }
  group.sync();
  if (result_value != xpool::abi::FfnResultCode::Ok) {
    const auto result_code = xpool::abi::FfnResultCode{result_value};
    if (result_code == xpool::abi::FfnResultCode::ProtocolMismatch) {
      return publish_protocol_failure(
          arena, transport_arena, transport_trace, group, key, context.layer_ordinal, result_code);
    }
    publish_transport_result(group, transport_arena, transport_trace, result_code);
    return result_code;
  }

  // Phase: Materialize - Copy the full contribution or construct the required zero
  // contribution before publishing the Transport result.
  const auto contribution = FfnResultContribution{result_publication.record.contribution};
  if (contribution == FfnResultContribution::Full) {
    const auto *source = prefill
                             ? arena.executor_output_payload(admission.executor_index)
                             : arena.model_output_payload(context.model_index);
    xpool::utils::cooperative::copy(group, transport_arena.output_payload(), source, context.payload_bytes);
  } else {
    xpool::utils::cooperative::fill(
        group, transport_arena.output_payload(), std::uint8_t{0}, context.payload_bytes);
  }
  if (group.thread_rank() == 0 && fabric_trace != nullptr) {
    fabric_trace->output_prepared();
  }

  const auto result_code = xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
  publish_transport_result(group, transport_arena, transport_trace, result_code);
  if (group.thread_rank() == 0 && fabric_trace != nullptr) {
    fabric_trace->transport_evaluated_published();
  }

  // Phase: Acknowledge - Publication proves this AtnAgent consumed the Result and lets
  // the Coordinator release the Scheduler entry and Executor lease.
  auto &acknowledgement = arena.acknowledgement_publication(
      static_cast<std::size_t>(context.atnagent_index), context.model_index);
  if (group.thread_rank() == 0) {
    acknowledgement.record = FfnResultAcknowledgement{.key = key};
  }
  group.sync();
  acknowledgement.publish(group, invocation_sequence, arena.layout().coordinator_pe());
  if (group.thread_rank() == 0 && fabric_trace != nullptr) {
    fabric_trace->acknowledgement_published();
  }
  return result_code;
}

} // namespace xpool::fabric::atnagent
