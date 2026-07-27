#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cooperative_groups.h>
#include <cuda/launch>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <xpool/abi.hpp>
#include <xpool/abort.hpp>
#include <xpool/atomic.cuh>
#include <xpool/fabric/arena.cuh>
#include <xpool/fabric/ffnagent.hpp>
#include <xpool/fabric/protocol.cuh>
#include <xpool/fabric/scheduler.cuh>
#include <xpool/ffnagent/executor.cuh>
#include <xpool/macros.hpp>
#include <xpool/utils/wait.cuh>

namespace xpool::fabric {

namespace {

constexpr int kResidentBlockSize = 256;

enum class RendezvousStatus : std::uint32_t {
  Pending,
  Ready,
  Failed,
};

struct RendezvousResult {
  RendezvousStatus status = RendezvousStatus::Pending;
  FfnInvocation invocation{};
  FfnInvocationKey failure_key{};
  std::size_t failure_layer_ordinal = 0;
};

XPOOL_DEVICE_FN bool invocation_equal_except_local_facts(
    const FfnSubmission &left,
    const FfnSubmission &right) {
  return left.key == right.key && left.layer_ordinal == right.layer_ordinal &&
         left.payload_rows == right.payload_rows &&
         left.result_handoff == right.result_handoff &&
         left.dp_padding_mode == right.dp_padding_mode;
}

/// Converge one model's next all-AtnAgent Submission sequence into a canonical
/// invocation without mutating Scheduler or publication state.
XPOOL_DEVICE_FN RendezvousResult derive_invocation(
    const FabricArenaView &arena,
    std::size_t model_index) {
  const auto &layout = arena.layout();
  const auto &model = arena.model_layout(model_index);
  const auto expected_sequence = arena.scheduler_entry(model_index).next_sequence();
  auto result = RendezvousResult{
      .failure_key = FfnInvocationKey{
          .model_index = model_index,
          .invocation_sequence = expected_sequence,
      },
  };

  // Phase: Observe - Require every AtnAgent to present exactly the Scheduler's
  // next sequence before deriving any shared invocation facts.
  auto pending = false;
  for (auto atnagent_index = std::size_t{0}; atnagent_index < layout.atnagent_count;
       ++atnagent_index) {
    const auto &publication = arena.submission_publication(atnagent_index, model_index);
    const auto sequence = publication.observe();
    if (sequence < expected_sequence) {
      pending = true;
      continue;
    }
    if (sequence > expected_sequence ||
        publication.validate() != xpool::abi::FfnResultCode::Ok ||
        publication.record.key != result.failure_key) {
      result.status = RendezvousStatus::Failed;
      result.failure_layer_ordinal = publication.record.layer_ordinal;
      return result;
    }
  }
  if (pending) {
    return result;
  }

  // Phase: Identity - Validate model-level fields that must agree across every
  // AtnAgent in the invocation.
  const auto &first = arena.submission_publication(0, model_index).record;
  result.failure_layer_ordinal = first.layer_ordinal;
  if (first.layer_ordinal >= model.layer_count ||
      !xpool::abi::FfnResultHandoff::is_valid(first.result_handoff) ||
      !xpool::abi::DpPaddingMode::is_valid(first.dp_padding_mode)) {
    result.status = RendezvousStatus::Failed;
    return result;
  }
  for (auto atnagent_index = std::size_t{1}; atnagent_index < layout.atnagent_count;
       ++atnagent_index) {
    const auto &submission = arena.submission_publication(atnagent_index, model_index).record;
    if (!invocation_equal_except_local_facts(first, submission)) {
      result.status = RendezvousStatus::Failed;
      return result;
    }
  }

  // Phase: Topology - Validate TP replicas and aggregate canonical DP-local
  // token counts into the physical invocation geometry.
  auto payload_rows = std::size_t{0};
  auto maximum_local_tokens = std::size_t{0};
  auto has_extend = false;
  auto has_decode = false;
  for (auto dp_rank = std::size_t{0}; dp_rank < model.atn_dp_size; ++dp_rank) {
    const auto canonical_index = model.atnagent_index(0, dp_rank);
    const auto &canonical = arena.submission_publication(canonical_index, model_index).record;
    for (auto tp_rank = std::size_t{1}; tp_rank < model.atn_tp_size; ++tp_rank) {
      const auto peer_index = model.atnagent_index(tp_rank, dp_rank);
      const auto &peer = arena.submission_publication(peer_index, model_index).record;
      if (peer.local_token_count != canonical.local_token_count ||
          peer.forward_mode != canonical.forward_mode) {
        result.status = RendezvousStatus::Failed;
        return result;
      }
    }
    if (canonical.local_token_count >
        std::numeric_limits<std::size_t>::max() - payload_rows) {
      result.status = RendezvousStatus::Failed;
      return result;
    }
    payload_rows += canonical.local_token_count;
    maximum_local_tokens = std::max(maximum_local_tokens, canonical.local_token_count);
    has_extend = has_extend || canonical.forward_mode == xpool::abi::XPoolForwardMode::Extend;
    has_decode = has_decode || canonical.forward_mode == xpool::abi::XPoolForwardMode::Decode;
  }

  // Phase: Padding - Prove the selected padding mode agrees with the aggregate
  // DP geometry presented by the Input Publisher.
  if (model.atn_dp_size == 1) {
    if (first.dp_padding_mode != xpool::abi::DpPaddingMode::None) {
      result.status = RendezvousStatus::Failed;
      return result;
    }
  } else if (first.dp_padding_mode == xpool::abi::DpPaddingMode::SumLen) {
    if (payload_rows != first.payload_rows) {
      result.status = RendezvousStatus::Failed;
      return result;
    }
  } else if (first.dp_padding_mode == xpool::abi::DpPaddingMode::MaxLen) {
    if (maximum_local_tokens >
            std::numeric_limits<std::size_t>::max() / model.atn_dp_size ||
        maximum_local_tokens * model.atn_dp_size != first.payload_rows) {
      result.status = RendezvousStatus::Failed;
      return result;
    }
  } else {
    result.status = RendezvousStatus::Failed;
    return result;
  }

  // Phase: Materialize - Select the execution mode and validate the completed
  // canonical invocation against immutable arena geometry.
  if (!has_extend && !has_decode) {
    result.status = RendezvousStatus::Failed;
    return result;
  }
  const auto execution_mode = has_extend ? FfnExecutionMode::Prefill
                                         : FfnExecutionMode::Decode;
  result.invocation = FfnInvocation{
      .key = result.failure_key,
      .layer_ordinal = first.layer_ordinal,
      .payload_rows = first.payload_rows,
      .input_pe = 0,
      .execution_mode = static_cast<std::uint32_t>(execution_mode),
      .result_handoff = first.result_handoff,
      .dp_padding_mode = first.dp_padding_mode,
  };
  result.status = result.invocation.validate(layout, model) ==
                          xpool::abi::FfnResultCode::Ok
                      ? RendezvousStatus::Ready
                      : RendezvousStatus::Failed;
  return result;
}

XPOOL_DEVICE_FN void publish_canonical_failure(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group,
    xpool::abi::FfnResultCode result_code,
    const FfnInvocationKey &key,
    std::size_t layer_ordinal) {
  if (group.thread_rank() == 0) {
    arena.state().failure.try_publish(
        arena.layout().coordinator_pe(), result_code, key, layer_ordinal);
  }
  group.sync();
}

/// Inspect every idle model entry and enqueue only complete, canonical
/// invocations. A mismatch publishes failure and stops this progress pass.
XPOOL_DEVICE_FN bool discover_invocations(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group) {
  for (auto model_index = std::size_t{0}; model_index < arena.layout().model_count;
       ++model_index) {
    if (arena.scheduler_entry(model_index).state() != FfnSchedulerEntry::State::Idle) {
      continue;
    }
    __shared__ RendezvousResult rendezvous;
    if (group.thread_rank() == 0) {
      rendezvous = derive_invocation(arena, model_index);
    }
    group.sync();
    if (rendezvous.status == RendezvousStatus::Failed) {
      publish_canonical_failure(
          arena, group,
          xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch},
          rendezvous.failure_key, rendezvous.failure_layer_ordinal);
      return false;
    }
    if (rendezvous.status == RendezvousStatus::Ready && group.thread_rank() == 0) {
      arena.scheduler().enqueue(arena, rendezvous.invocation);
    }
    group.sync();
  }
  return true;
}

/// Publish a Scheduler decision first to all AtnAgent admission cells and then
/// to every FfnAgent's stable Executor cell.
XPOOL_DEVICE_FN void publish_scheduled_invocation(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group,
    const FfnScheduler::Decision &decision) {
  const auto &invocation = decision.invocation();
  const auto executor_index = decision.executor_index();
  auto *trace = arena.find_trace(
      arena.scheduler_entry(invocation.key.model_index).scheduler_trace_id());
  if (group.thread_rank() == 0 && trace != nullptr) {
    trace->scheduled(executor_index);
  }

  for (auto atnagent_index = std::size_t{0};
       atnagent_index < arena.layout().atnagent_count; ++atnagent_index) {
    auto &admission = arena.admission_publication(
        atnagent_index, invocation.key.model_index);
    if (group.thread_rank() == 0) {
      admission.record = FfnExecutionAdmission{
          .key = invocation.key,
          .executor_index = executor_index,
          .execution_mode = invocation.execution_mode,
      };
    }
    group.sync();
    admission.publish(
        group, invocation.key.invocation_sequence,
        static_cast<int>(atnagent_index));
  }
  if (group.thread_rank() == 0 && trace != nullptr) {
    trace->admissions_published();
  }

  auto &publication = arena.invocation_publication(executor_index);
  if (group.thread_rank() == 0) {
    publication.record = invocation;
  }
  group.sync();
  const auto &layout = arena.layout();
  for (auto ffnagent_index = std::size_t{0}; ffnagent_index < layout.ffnagent_count;
       ++ffnagent_index) {
    publication.publish(
        group, invocation.key.invocation_sequence,
        static_cast<int>(layout.atnagent_count + ffnagent_index));
  }
  if (group.thread_rank() == 0 && trace != nullptr) {
    trace->invocations_published();
  }
}

/// Observe one completion from every FfnAgent for an active Executor without
/// clearing any publication until the complete set is valid.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode inspect_completions(
    const FabricArenaView &arena,
    const FfnInvocation &invocation,
    std::size_t executor_index,
    bool &complete) {
  complete = true;
  for (auto ffnagent_index = std::size_t{0};
       ffnagent_index < arena.layout().ffnagent_count; ++ffnagent_index) {
    const auto &publication = arena.ffnagent_completion_publication(
        ffnagent_index, executor_index);
    const auto sequence = publication.observe();
    if (sequence < invocation.key.invocation_sequence) {
      complete = false;
      continue;
    }
    if (sequence > invocation.key.invocation_sequence ||
        publication.validate() != xpool::abi::FfnResultCode::Ok ||
        publication.record.key != invocation.key) {
      return xpool::abi::FfnResultCode{
          xpool::abi::FfnResultCode::ProtocolMismatch};
    }
  }
  return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
}

/// Return whether every AtnAgent result cell already carries this invocation.
XPOOL_DEVICE_FN bool results_published(
    const FabricArenaView &arena,
    const FfnInvocation &invocation) {
  for (auto atnagent_index = std::size_t{0};
       atnagent_index < arena.layout().atnagent_count; ++atnagent_index) {
    if (arena.result_publication(atnagent_index, invocation.key.model_index).record.key !=
        invocation.key) {
      return false;
    }
  }
  return true;
}

/// Materialize one completed Executor payload into per-AtnAgent Result
/// publications, preserving the configured replicated or zero contribution.
XPOOL_DEVICE_FN void publish_results(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group,
    const FfnInvocation &invocation,
    std::size_t executor_index) {
  const auto &model = arena.model_layout(invocation.key.model_index);
  const auto payload_bytes = invocation.payload_rows * model.hidden_size *
                             xpool::abi::TensorDType{model.dtype}.bytes();
  auto *payload = invocation.execution_mode == FfnExecutionMode::Prefill
                      ? arena.executor_output_payload(executor_index)
                      : arena.model_output_payload(invocation.key.model_index);
  for (auto atnagent_index = std::size_t{0};
       atnagent_index < arena.layout().atnagent_count; ++atnagent_index) {
    const auto contribution =
        invocation.result_handoff == xpool::abi::FfnResultHandoff::ReplicatedFull ||
                atnagent_index == 0
            ? FfnResultContribution::Full
            : FfnResultContribution::Zero;
    auto &result = arena.result_publication(
        atnagent_index, invocation.key.model_index);
    if (group.thread_rank() == 0) {
      result.record = FfnResult{
          .key = invocation.key,
          .contribution = static_cast<std::uint32_t>(contribution),
      };
    }
    group.sync();
    if (contribution == FfnResultContribution::Full) {
      result.publish(
          group, invocation.key.invocation_sequence,
          static_cast<int>(atnagent_index), payload, payload_bytes);
    } else {
      result.publish(
          group, invocation.key.invocation_sequence,
          static_cast<int>(atnagent_index));
    }
  }
}

/// Observe one acknowledgement from every AtnAgent without releasing the
/// Scheduler entry until the complete set is valid.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode inspect_acknowledgements(
    const FabricArenaView &arena,
    const FfnInvocation &invocation,
    bool &complete) {
  complete = true;
  for (auto atnagent_index = std::size_t{0};
       atnagent_index < arena.layout().atnagent_count; ++atnagent_index) {
    const auto &publication = arena.acknowledgement_publication(
        atnagent_index, invocation.key.model_index);
    const auto sequence = publication.observe();
    if (sequence < invocation.key.invocation_sequence) {
      complete = false;
      continue;
    }
    if (sequence > invocation.key.invocation_sequence ||
        publication.validate() != xpool::abi::FfnResultCode::Ok ||
        publication.record.key != invocation.key) {
      return xpool::abi::FfnResultCode{
          xpool::abi::FfnResultCode::ProtocolMismatch};
    }
  }
  return xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
}

/// Advance active Scheduler entries through completion, result publication,
/// acknowledgement, and final Executor release.
XPOOL_DEVICE_FN bool progress_active_invocations(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group) {
  for (auto model_index = std::size_t{0}; model_index < arena.layout().model_count;
       ++model_index) {
    auto &entry = arena.scheduler_entry(model_index);
    if (entry.state() != FfnSchedulerEntry::State::Active) {
      continue;
    }
    const auto invocation = entry.invocation();
    const auto executor_index = entry.executor_index();
    auto *trace = arena.find_trace(entry.scheduler_trace_id());
    // Phase: Complete - Gather every FfnAgent completion before publishing any
    // AtnAgent-visible result.
    if (!results_published(arena, invocation)) {
      __shared__ bool complete;
      __shared__ std::uint32_t result_code;
      if (group.thread_rank() == 0) {
        result_code = inspect_completions(
                          arena, invocation, executor_index, complete)
                          .value();
      }
      group.sync();
      if (result_code != xpool::abi::FfnResultCode::Ok) {
        publish_canonical_failure(
            arena, group, xpool::abi::FfnResultCode{result_code}, invocation.key,
            invocation.layer_ordinal);
        return false;
      }
      if (!complete) {
        continue;
      }
      if (group.thread_rank() == 0) {
        for (auto ffnagent_index = std::size_t{0};
             ffnagent_index < arena.layout().ffnagent_count; ++ffnagent_index) {
          arena.ffnagent_completion_publication(ffnagent_index, executor_index).clear();
        }
        if (trace != nullptr) {
          trace->completions_observed();
        }
      }
      group.sync();
      publish_results(arena, group, invocation, executor_index);
      if (group.thread_rank() == 0 && trace != nullptr) {
        trace->results_published();
      }
      continue;
    }

    // Phase: Acknowledge - Release the Scheduler and Executor only after every
    // AtnAgent proves it consumed the matching result.
    __shared__ bool acknowledged;
    __shared__ std::uint32_t result_code;
    if (group.thread_rank() == 0) {
      result_code = inspect_acknowledgements(arena, invocation, acknowledged).value();
    }
    group.sync();
    if (result_code != xpool::abi::FfnResultCode::Ok) {
      publish_canonical_failure(
          arena, group, xpool::abi::FfnResultCode{result_code}, invocation.key,
          invocation.layer_ordinal);
      return false;
    }
    if (!acknowledged) {
      continue;
    }
    if (group.thread_rank() == 0) {
      if (trace != nullptr) {
        trace->acknowledgements_observed();
      }
      arena.scheduler().release(arena, invocation.key, executor_index);
      if (trace != nullptr) {
        trace->scheduler_released();
      }
    }
    group.sync();
  }
  return true;
}

/// Own generation-wide invocation convergence and scheduling on the sole
/// Coordinator block until canonical failure or shutdown.
XPOOL_DEVICE_FN void coordinator_loop(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group) {
  while (true) {
    if (arena.state().failure.published()) {
      if (group.thread_rank() == 0) {
        arena.state().failure.publish_to_all(
            static_cast<int>(arena.layout().atnagent_count +
                             arena.layout().ffnagent_count));
        arena.request_shutdown();
      }
      group.sync();
      return;
    }
    if (arena.shutdown_requested()) {
      return;
    }
    // Phase: Converge - Aggregate one complete all-AtnAgent Submission set into each
    // model's Scheduler entry, or publish the first protocol failure.
    if (!discover_invocations(arena, group)) {
      continue;
    }
    // Phase: Progress - Materialize completed Results and release fully acknowledged
    // active Executor leases before new admission.
    if (!progress_active_invocations(arena, group)) {
      continue;
    }
    __shared__ FfnScheduler::Decision decision;
    // Phase: Schedule - One thread chooses a ready Invocation and free Executor, then
    // the block publishes Admissions and the distributed Invocation.
    if (group.thread_rank() == 0) {
      decision = arena.scheduler().schedule(arena);
    }
    group.sync();
    if (decision) {
      publish_scheduled_invocation(arena, group, decision);
    } else if (group.thread_rank() == 0) {
      xpool::utils::wait::relax();
    }
    group.sync();
  }
}

/// Wait for the Input Publisher's matching Prefill payload publication on one
/// Executor, treating timeout as a PE-local shutdown request.
XPOOL_DEVICE_FN bool wait_for_prefill_input(
    const FabricArenaView &arena,
    const FfnInvocation &invocation,
    std::size_t executor_index) {
  auto &publication = arena.input_ready_publication(executor_index);
  const auto deadline =
      xpool::utils::wait::Deadline::after(xpool::utils::wait::kDefaultTimeoutNanoseconds);
  const auto result = xpool::utils::wait::until(
      deadline,
      [&] { return publication.observe() >= invocation.key.invocation_sequence; },
      [&] { return arena.shutdown_requested() || arena.state().failure.published(); });
  if (result == xpool::utils::wait::Result::TimedOut) {
    arena.request_shutdown();
    return false;
  }
  return result == xpool::utils::wait::Result::Ready &&
         publication.observe() == invocation.key.invocation_sequence &&
         publication.validate() == xpool::abi::FfnResultCode::Ok &&
         publication.record.key == invocation.key;
}

/// Permanently bind one Resident block to one stable Executor and publish only
/// success completions after mode-specific input acquisition and execution.
XPOOL_DEVICE_FN void execution_loop(
    const FabricArenaView &arena,
    const cooperative_groups::thread_block &group,
    std::size_t executor_index) {
  __shared__ FfnInvocation invocation;
  __shared__ FabricTraceRecord *trace;
  __shared__ std::uint32_t result_code;

  const auto &layout = arena.layout();
  const auto pe = nvshmem_my_pe();
  const auto ffnagent_index = static_cast<std::size_t>(pe) - layout.atnagent_count;
  auto &publication = arena.invocation_publication(executor_index);
  while (true) {
    // Phase: Observe - Wait for the Coordinator to publish one Invocation owned by
    // this stable Executor index.
    __shared__ std::uint64_t sequence;
    if (group.thread_rank() == 0) {
      sequence = publication.observe();
      if (sequence == 0) {
        xpool::utils::wait::relax();
      }
    }
    group.sync();
    if (sequence == 0) {
      if (arena.shutdown_requested() || arena.state().failure.published()) {
        return;
      }
      continue;
    }

    if (group.thread_rank() == 0) {
      invocation = publication.record;
      result_code = publication.validate().value();
      trace = nullptr;
      if (result_code == xpool::abi::FfnResultCode::Ok &&
          invocation.key.model_index >= layout.model_count) {
        result_code = xpool::abi::FfnResultCode::ProtocolMismatch;
      }
      if (result_code == xpool::abi::FfnResultCode::Ok) {
        const auto &model = arena.model_layout(invocation.key.model_index);
        result_code = invocation.validate(layout, model).value();
        if (result_code == xpool::abi::FfnResultCode::Ok) {
          const auto trace_entry = arena.reserve_trace();
          if (trace_entry) {
            trace = &trace_entry.record();
            trace->begin_execution(
                trace_entry.sequence(), invocation,
                arena.layer_layout(model.layer_begin + invocation.layer_ordinal).layer_id,
                executor_index);
          }
        }
      }
    }
    group.sync();
    if (result_code != xpool::abi::FfnResultCode::Ok) {
      publish_canonical_failure(
          arena, group,
          xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch},
          invocation.key, invocation.layer_ordinal);
      if (group.thread_rank() == 0) {
        publication.clear();
      }
      group.sync();
      return;
    }

    const auto &model = arena.model_layout(invocation.key.model_index);
    const auto &layer = arena.layer_layout(model.layer_begin + invocation.layer_ordinal);
    const auto payload_bytes = invocation.payload_rows * model.hidden_size *
                               xpool::abi::TensorDType{model.dtype}.bytes();
    const auto prefill = invocation.execution_mode == FfnExecutionMode::Prefill;
    auto *input = prefill ? arena.executor_input_payload(executor_index)
                          : arena.model_input_payload(invocation.key.model_index);
    auto *output = prefill ? arena.executor_output_payload(executor_index)
                           : arena.model_output_payload(invocation.key.model_index);

    // Phase: Acquire input - Prefill consumes Executor-owned pushed storage; Decode
    // pulls the fixed model payload from the elected Input Publisher.
    if (prefill) {
      if (group.thread_rank() == 0) {
        result_code = wait_for_prefill_input(arena, invocation, executor_index)
                          ? xpool::abi::FfnResultCode::Ok
                          : xpool::abi::FfnResultCode::Timeout;
        if (result_code == xpool::abi::FfnResultCode::Ok && trace != nullptr) {
          trace->prefill_input_ready_observed();
        }
      }
    } else {
      if (group.thread_rank() == 0 && trace != nullptr) {
        trace->decode_input_pull_started();
      }
      group.sync();
      nvshmemx_getmem_block(input, input, payload_bytes, invocation.input_pe);
      if (group.thread_rank() == 0 && trace != nullptr) {
        trace->decode_input_pull_completed();
      }
    }
    group.sync();
    if (result_code != xpool::abi::FfnResultCode::Ok) {
      return;
    }

    // Phase: Execute - Every FfnAgent evaluates its local layer contribution after the
    // input path has completed for the whole block.
    if (group.thread_rank() == 0 && trace != nullptr) {
      trace->execution_started();
    }
    group.sync();
    const auto execution_result =
        xpool::ffnagent::executor::execute(group, invocation, model, layer, input, output);
    if (group.thread_rank() == 0) {
      result_code = execution_result.value();
      if (result_code == xpool::abi::FfnResultCode::Ok && trace != nullptr) {
        trace->execution_completed();
      }
    }
    group.sync();
    if (result_code != xpool::abi::FfnResultCode::Ok) {
      const auto code = xpool::abi::FfnResultCode{result_code};
      if (code == xpool::abi::FfnResultCode::ProtocolMismatch ||
          code == xpool::abi::FfnResultCode::NotImplemented) {
        publish_canonical_failure(arena, group, code, invocation.key, invocation.layer_ordinal);
      }
      if (group.thread_rank() == 0) {
        if (prefill) {
          arena.input_ready_publication(executor_index).clear();
        }
        publication.clear();
      }
      group.sync();
      return;
    }

    if (group.thread_rank() == 0) {
      if (prefill) {
        arena.input_ready_publication(executor_index).clear();
      }
      publication.clear();
    }
    group.sync();

    // Phase: Complete - Clear consumed input publications, then publish this FfnAgent's
    // success-only Completion to the Coordinator.
    auto &completion = arena.ffnagent_completion_publication(
        ffnagent_index, executor_index);
    if (group.thread_rank() == 0) {
      completion.record = FfnAgentCompletion{.key = invocation.key};
    }
    group.sync();
    completion.publish(group, invocation.key.invocation_sequence, layout.coordinator_pe());
    if (group.thread_rank() == 0 && trace != nullptr) {
      trace->completion_published();
    }
  }
}

/// Run one cooperative FfnAgent Resident grid whose blocks have stable
/// Coordinator or Executor ownership for the complete generation.
__global__ void ffnagent_resident_kernel(FabricArenaView arena, bool coordinator) {
  const auto grid = cooperative_groups::this_grid();
  const auto block = cooperative_groups::this_thread_block();
  // Phase: Startup - Publish readiness only after every cooperative block is resident.
  grid.sync();
  if (blockIdx.x == 0 && block.thread_rank() == 0) {
    xpool::atomic::store_release(
        arena.state().ffnagent_resident_ready, std::uint32_t{1});
  }
  grid.sync();

  // Phase: Role dispatch - Block zero on the Coordinator PE owns convergence and
  // scheduling; every remaining block permanently owns one Executor.
  if (coordinator && blockIdx.x == 0) {
    coordinator_loop(arena, block);
  } else {
    const auto executor_index = static_cast<std::size_t>(blockIdx.x) -
                                (coordinator ? std::size_t{1} : std::size_t{0});
    execution_loop(arena, block, executor_index);
  }
  // Phase: Drain convergence - No Resident block exits before all roles have stopped.
  grid.sync();
}

} // namespace

void launch_ffnagent_kernel(
    FabricArenaView arena,
    const FabricArenaLayout &layout,
    cudaStream_t stream) {
  TORCH_CHECK(!arena.empty(), "xpool FfnAgent Resident requires a joined Fabric arena");
  TORCH_CHECK(stream != nullptr, "xpool FfnAgent Resident requires a CUDA stream");
  const auto pe = nvshmem_my_pe();
  TORCH_CHECK(
      pe >= static_cast<int>(layout.atnagent_count) &&
          pe < static_cast<int>(layout.atnagent_count + layout.ffnagent_count),
      "xpool FfnAgent Resident requires an FfnAgent PE");
  const auto coordinator = pe == layout.coordinator_pe();
  const auto block_count = layout.executor_count + (coordinator ? 1U : 0U);

  auto device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  auto cooperative_launch = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(
      cooperative_launch != 0,
      "xpool FfnAgent Resident requires cooperative-launch support");

  auto blocks_per_multiprocessor = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_multiprocessor, ffnagent_resident_kernel,
      kResidentBlockSize, 0));
  auto multiprocessor_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  const auto resident_capacity =
      static_cast<std::size_t>(blocks_per_multiprocessor) *
      static_cast<std::size_t>(multiprocessor_count);
  TORCH_CHECK(
      block_count <= resident_capacity,
      "xpool FfnAgent Resident requires ", block_count,
      " concurrently resident blocks but the device supports ", resident_capacity);

  const auto config = cuda::make_config(
      cuda::block_dims<kResidentBlockSize>(), cuda::grid_dims(block_count),
      cuda::cooperative_launch{});
  cuda::launch(
      cuda::stream_ref{stream}, config, ffnagent_resident_kernel, arena, coordinator);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xpool::fabric
