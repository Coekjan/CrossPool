#include <xpool/fabric/coordinator.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/launch>
#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <xpool/abort.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/fabric/protocol.cuh>
#include <xpool/fabric/scheduler.cuh>
#include <xpool/ffn.hpp>
#include <xpool/hooks.cuh>
#include <xpool/macros.hpp>
#include <xpool/utils/cooperative.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::fabric {

namespace {

constexpr auto kBlockSize = 256U;

enum class InspectionStatus : std::uint32_t {
  Pending,
  Ready,
  ProtocolMismatch,
};

struct SubmissionRendezvous {
  InspectionStatus status = InspectionStatus::Pending;
  Invocation invocation{};
};

XPOOL_DEVICE_FN void publish_canonical_failure(const ArenaView &arena, const cooperative_groups::thread_block &group,
                                               xpool::ffn::ResultCode result_code, const InvocationKey &key,
                                               std::size_t layer_ordinal) {
  if (group.thread_rank() == 0) {
    arena.state().failure.try_publish(arena.layout().coordinator_pe(), result_code, key, layer_ordinal);
  }
  group.sync();
}

XPOOL_DEVICE_FN cuda::std::optional<xpool::ffn::ForwardMode>
inspect_submission_rows(const ArenaView &arena, const InstanceEntry &instance, cuda::std::span<const int> atnagent_pes,
                        const Submission &canonical) {
  const auto row_layout = canonical.dp_row_layout;
  if ((instance.atn_dp_size == 1 && row_layout != xpool::ffn::DpRowLayout::None) ||
      (instance.atn_dp_size > 1 && row_layout == xpool::ffn::DpRowLayout::None)) {
    return cuda::std::nullopt;
  }

  auto any_prefill = false;
  auto any_non_idle = false;
  auto packed_payload_rows = std::size_t{0};
  auto uniform_payload_rows = std::size_t{0};
  for (auto dp_rank = std::size_t{0}; dp_rank < instance.atn_dp_size; ++dp_rank) {
    const auto canonical_index = dp_rank * instance.atn_tp_size;
    const auto &dp_submission = arena
                                    .submission_publication(static_cast<std::size_t>(atnagent_pes[canonical_index]),
                                                            canonical.key.instance_index)
                                    .record;
    const auto mode = dp_submission.forward_mode;
    const auto rank_payload_rows = dp_submission.dp_rank_payload_rows;
    if ((row_layout == xpool::ffn::DpRowLayout::None &&
         (mode == xpool::ffn::ForwardMode::Idle || rank_payload_rows != canonical.payload_rows)) ||
        (row_layout == xpool::ffn::DpRowLayout::PackedByRank &&
         ((mode == xpool::ffn::ForwardMode::Idle) != (rank_payload_rows == 0))) ||
        (row_layout == xpool::ffn::DpRowLayout::UniformByRank &&
         (rank_payload_rows == 0 || (dp_rank != 0 && rank_payload_rows != uniform_payload_rows)))) {
      return cuda::std::nullopt;
    }

    any_prefill = any_prefill || mode == xpool::ffn::ForwardMode::Prefill;
    any_non_idle = any_non_idle || mode != xpool::ffn::ForwardMode::Idle;
    if (row_layout == xpool::ffn::DpRowLayout::PackedByRank) {
      if (rank_payload_rows > canonical.payload_rows - packed_payload_rows) {
        return cuda::std::nullopt;
      }
      packed_payload_rows += rank_payload_rows;
    } else if (row_layout == xpool::ffn::DpRowLayout::UniformByRank && dp_rank == 0) {
      uniform_payload_rows = rank_payload_rows;
    }

    for (auto tp_rank = std::size_t{1}; tp_rank < instance.atn_tp_size; ++tp_rank) {
      const auto replica_index = canonical_index + tp_rank;
      const auto &replica = arena
                                .submission_publication(static_cast<std::size_t>(atnagent_pes[replica_index]),
                                                        canonical.key.instance_index)
                                .record;
      if (replica.dp_rank_payload_rows != dp_submission.dp_rank_payload_rows ||
          replica.forward_mode != dp_submission.forward_mode) {
        return cuda::std::nullopt;
      }
    }
  }

  if (!any_non_idle ||
      (row_layout == xpool::ffn::DpRowLayout::PackedByRank && packed_payload_rows != canonical.payload_rows) ||
      (row_layout == xpool::ffn::DpRowLayout::UniformByRank &&
       (canonical.payload_rows % instance.atn_dp_size != 0 ||
        canonical.payload_rows / instance.atn_dp_size != uniform_payload_rows))) {
    return cuda::std::nullopt;
  }
  return any_prefill ? xpool::ffn::ForwardMode::Prefill : xpool::ffn::ForwardMode::Decode;
}

XPOOL_DEVICE_FN SubmissionRendezvous inspect_submission_rendezvous(const ArenaView &arena, std::size_t instance_index,
                                                                   std::uint64_t expected_sequence) {
  auto rendezvous = SubmissionRendezvous{};
  const auto &instance = arena.instance_entry(instance_index);
  const auto atnagent_pes = arena.atnagent_pes(instance_index);
  const auto key = InvocationKey{
      .instance_index = instance_index,
      .invocation_sequence = expected_sequence,
  };

  // Phase: Await Publications - Do not inspect a Submission until every
  // participating AtnAgent reached the expected sequence.
  for (const auto source_pe : atnagent_pes) {
    if (!arena.submission_publication(static_cast<std::size_t>(source_pe), instance_index)
             .test_at_least(expected_sequence)) {
      return rendezvous;
    }
  }
  rendezvous.invocation.key = key;

  // Phase: Agree Submission - Use the first publication as the canonical
  // description, then validate and compare every independently published copy.
  const auto &first =
      arena.submission_publication(static_cast<std::size_t>(atnagent_pes.front()), instance_index).record;
  for (const auto source_pe : atnagent_pes) {
    const auto &publication = arena.submission_publication(static_cast<std::size_t>(source_pe), instance_index);
    const auto &submission = publication.record;
    if (publication.validate_expected(expected_sequence, key) != xpool::ffn::ResultCode::Ok ||
        submission.layer_ordinal != first.layer_ordinal || submission.payload_rows != first.payload_rows ||
        submission.output_requirement != first.output_requirement || submission.dp_row_layout != first.dp_row_layout) {
      rendezvous.status = InspectionStatus::ProtocolMismatch;
      return rendezvous;
    }
  }
  if (first.layer_ordinal >= instance.layer_count ||
      (first.output_requirement == xpool::ffn::OutputRequirement::GroupSumComplete &&
       !instance.group_sum_complete_admitted)) {
    rendezvous.status = InspectionStatus::ProtocolMismatch;
    return rendezvous;
  }

  // Phase: Aggregate Rows - DP canonical records and their TP replicas must
  // agree on physical row geometry and select one capacity profile.
  const auto capacity_mode = inspect_submission_rows(arena, instance, atnagent_pes, first);
  if (!capacity_mode.has_value()) {
    rendezvous.status = InspectionStatus::ProtocolMismatch;
    return rendezvous;
  }

  // Phase: Admit Invocation - The agreed live rows must fit both the selected
  // model profile and fixed Lane payload storage.
  const auto profile_capacity = *capacity_mode == xpool::ffn::ForwardMode::Prefill
                                    ? instance.prefill_payload_row_capacity
                                    : instance.decode_payload_row_capacity;
  if (first.payload_rows > profile_capacity ||
      first.payload_rows > arena.layout().lane_payload_capacity_bytes / instance.payload_row_bytes) {
    rendezvous.status = InspectionStatus::ProtocolMismatch;
    return rendezvous;
  }

  rendezvous.status = InspectionStatus::Ready;
  rendezvous.invocation.layer_ordinal = first.layer_ordinal;
  rendezvous.invocation.payload_rows = first.payload_rows;
  rendezvous.invocation.output_requirement = first.output_requirement;
  return rendezvous;
}

XPOOL_DEVICE_FN void discover_invocations(const ArenaView &arena, const cooperative_groups::thread_block &group,
                                          Scheduler &scheduler) {
  XPOOL_DEVICE_SHARED SubmissionRendezvous rendezvous;
  for (auto instance_index = std::size_t{0}; instance_index < arena.layout().instance_count; ++instance_index) {
    if (group.thread_rank() == 0) {
      rendezvous = {};
      if (!scheduler.has_unresolved_invocation(instance_index)) {
        const auto previous_sequence = arena.output_commit_publication(instance_index).record.key.invocation_sequence;
        if (previous_sequence == std::numeric_limits<std::uint64_t>::max()) {
          rendezvous.status = InspectionStatus::ProtocolMismatch;
          rendezvous.invocation.key =
              InvocationKey{.instance_index = instance_index, .invocation_sequence = previous_sequence};
        } else {
          rendezvous = inspect_submission_rendezvous(arena, instance_index, previous_sequence + 1);
        }
      }
    }
    group.sync();
    if (rendezvous.status == InspectionStatus::ProtocolMismatch) {
      publish_canonical_failure(arena, group, xpool::ffn::ResultCode::ProtocolMismatch, rendezvous.invocation.key,
                                rendezvous.invocation.layer_ordinal);
      return;
    }
    if (rendezvous.status == InspectionStatus::Ready && group.thread_rank() == 0) {
      const auto ticket = scheduler.enqueue(rendezvous.invocation);
      xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
          {.arena = arena,
           .instance_index = instance_index,
           .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Enqueued,
           .invocation = &rendezvous.invocation,
           .ready_ticket = ticket});
    }
    group.sync();
  }
}

XPOOL_DEVICE_FN InspectionStatus inspect_completions(const ArenaView &arena, const Invocation &invocation,
                                                     std::size_t executor_lane_index, std::uint64_t lease_sequence) {
  auto status = InspectionStatus::Ready;
  const auto ffnagent_pes = arena.ffnagent_pes(invocation.key.instance_index, invocation.layer_ordinal);
  for (const auto source_pe : ffnagent_pes) {
    const auto source_index = static_cast<std::size_t>(source_pe) - arena.layout().atnagent_count;
    const auto &publication = arena.ffnagent_completion_publication(source_index, executor_lane_index);
    if (!publication.test_at_least(lease_sequence)) {
      status = InspectionStatus::Pending;
      continue;
    }
    if (publication.validate_expected(lease_sequence, invocation.key) != xpool::ffn::ResultCode::Ok) {
      return InspectionStatus::ProtocolMismatch;
    }
  }
  return status;
}

XPOOL_DEVICE_FN InspectionStatus inspect_acknowledgements(const ArenaView &arena, const Invocation &invocation) {
  auto status = InspectionStatus::Ready;
  for (const auto source_pe : arena.atnagent_pes(invocation.key.instance_index)) {
    const auto &publication =
        arena.output_acknowledgement_publication(static_cast<std::size_t>(source_pe), invocation.key.instance_index);
    if (!publication.test_at_least(invocation.key.invocation_sequence)) {
      status = InspectionStatus::Pending;
      continue;
    }
    if (publication.validate_expected(invocation.key.invocation_sequence, invocation.key) !=
        xpool::ffn::ResultCode::Ok) {
      return InspectionStatus::ProtocolMismatch;
    }
  }
  return status;
}

XPOOL_DEVICE_FN void progress_active_invocations(const ArenaView &arena, const cooperative_groups::thread_block &group,
                                                 Scheduler &scheduler) {
  XPOOL_DEVICE_SHARED InspectionStatus inspection_status;
  XPOOL_DEVICE_SHARED bool canonical_failure_published;
  for (auto instance_index = std::size_t{0}; instance_index < arena.layout().instance_count; ++instance_index) {
    const auto active = scheduler.active_decision(instance_index);
    if (!active.has_value()) {
      continue;
    }
    const auto invocation = active->invocation();
    const auto lane = active->executor_lane_index();
    const auto lease = arena.lane_execution_publication(lane).record.executor_lease_sequence;
    auto &commit = arena.output_commit_publication(instance_index);
    if (commit.record.key != invocation.key) {
      // Completion-to-Commit: OutputCommit is published only after every
      // participating FfnAgent reports the matching Lane lease complete.
      if (group.thread_rank() == 0) {
        inspection_status = inspect_completions(arena, invocation, lane, lease);
      }
      group.sync();
      if (inspection_status == InspectionStatus::ProtocolMismatch) {
        publish_canonical_failure(arena, group, xpool::ffn::ResultCode::ProtocolMismatch, invocation.key,
                                  invocation.layer_ordinal);
        return;
      }
      if (inspection_status == InspectionStatus::Pending) {
        continue;
      }
      if (group.thread_rank() == 0) {
        canonical_failure_published = arena.state().failure.published();
      }
      group.sync();
      if (!canonical_failure_published && group.thread_rank() == 0) {
        xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
            {.arena = arena,
             .instance_index = instance_index,
             .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::FfnAgentCompletionsObserved});
      }
      if (!canonical_failure_published && group.thread_rank() == 0) {
        commit.record = OutputCommit{.key = invocation.key};
      }
      group.sync();
      if (canonical_failure_published) {
        return;
      }
      for (const auto destination_pe : arena.atnagent_pes(instance_index)) {
        commit.publish_record(group, destination_pe);
      }
      if (group.thread_rank() == 0) {
        xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
            {.arena = arena,
             .instance_index = instance_index,
             .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputCommitPublished});
      }
      continue;
    }

    // Acknowledgement-to-Lane-Release: A Lane remains active after OutputCommit
    // until every AtnAgent consumed or deliberately zero-filled its local result.
    if (group.thread_rank() == 0) {
      inspection_status = inspect_acknowledgements(arena, invocation);
    }
    group.sync();
    if (inspection_status == InspectionStatus::ProtocolMismatch) {
      publish_canonical_failure(arena, group, xpool::ffn::ResultCode::ProtocolMismatch, invocation.key,
                                invocation.layer_ordinal);
      return;
    }
    if (inspection_status == InspectionStatus::Pending) {
      continue;
    }
    if (group.thread_rank() == 0) {
      xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
          {.arena = arena,
           .instance_index = instance_index,
           .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::OutputAcknowledgementsObserved});
      scheduler.release(invocation.key, lane);
      xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
          {.arena = arena,
           .instance_index = instance_index,
           .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneReleased});
    }
    group.sync();
  }
}

XPOOL_DEVICE_FN void publish_scheduled_invocation(const ArenaView &arena, const cooperative_groups::thread_block &group,
                                                  const Invocation &invocation, std::size_t lane) {
  auto &lane_publication = arena.lane_execution_publication(lane);
  const auto previous_lease = lane_publication.record.executor_lease_sequence;
  xpool::abort_if(previous_lease == std::numeric_limits<std::uint64_t>::max());
  const auto lease = previous_lease + 1;
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
        {.arena = arena,
         .instance_index = invocation.key.instance_index,
         .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::Scheduled,
         .executor_lane_index = lane,
         .executor_lease_sequence = lease});
  }

  // Publish Admission first so AtnAgents can place input into the selected
  // Lane before FfnAgents begin waiting on that Lane's execution record.
  auto &admission = arena.admission_publication(invocation.key.instance_index);
  if (group.thread_rank() == 0) {
    admission.record = Admission{
        .key = invocation.key,
        .executor_lane_index = lane,
        .executor_lease_sequence = lease,
    };
  }
  group.sync();
  for (const auto destination_pe : arena.atnagent_pes(invocation.key.instance_index)) {
    admission.publish_record(group, destination_pe);
  }
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
        {.arena = arena,
         .instance_index = invocation.key.instance_index,
         .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::AdmissionPublished});
  }

  if (group.thread_rank() == 0) {
    lane_publication.record = LaneExecution{
        .key = invocation.key,
        .executor_lease_sequence = lease,
        .layer_ordinal = invocation.layer_ordinal,
        .payload_rows = invocation.payload_rows,
        .output_requirement = invocation.output_requirement,
    };
  }
  group.sync();
  for (const auto destination_pe : arena.ffnagent_pes(invocation.key.instance_index, invocation.layer_ordinal)) {
    lane_publication.publish_record(group, destination_pe);
  }
  if (group.thread_rank() == 0) {
    xpool::hooks::FabricCoordinatorProtocolEvent::hooks(
        {.arena = arena,
         .instance_index = invocation.key.instance_index,
         .kind = xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneExecutionPublished});
  }
}

XPOOL_DEVICE_FN void coordinator_loop(const ArenaView &arena, const cooperative_groups::thread_block &group,
                                      Scheduler &scheduler) {
  while (true) {
    if (arena.state().failure.published()) {
      if (group.thread_rank() == 0) {
        arena.state().failure.publish_to_all(
            static_cast<int>(arena.layout().atnagent_count + arena.layout().ffnagent_count));
      }
      group.sync();
      return;
    }
    if (arena.shutdown_requested()) {
      return;
    }
    // Phase: Discover - Complete AtnAgent Submission rendezvous become FIFO
    // scheduler entries; incomplete rendezvous remain invisible to scheduling.
    discover_invocations(arena, group, scheduler);
    if (arena.state().failure.published()) {
      continue;
    }
    // Phase: Progress - Active invocations advance through commit and
    // acknowledgement before their Lane becomes reusable.
    progress_active_invocations(arena, group, scheduler);
    if (arena.state().failure.published()) {
      continue;
    }

    // Phase: Schedule - At most one ready invocation acquires one free Lane;
    // Admission becomes visible before the matching LaneExecution publication.
    XPOOL_DEVICE_SHARED bool scheduled;
    XPOOL_DEVICE_SHARED Invocation scheduled_invocation;
    XPOOL_DEVICE_SHARED std::size_t scheduled_lane;
    if (group.thread_rank() == 0) {
      const auto decision = scheduler.try_schedule();
      scheduled = decision.has_value();
      if (scheduled) {
        scheduled_invocation = decision->invocation();
        scheduled_lane = decision->executor_lane_index();
      } else {
        xpool::utils::wait::relax();
      }
    }
    group.sync();
    if (scheduled) {
      publish_scheduled_invocation(arena, group, scheduled_invocation, scheduled_lane);
    }
    group.sync();
  }
}

XPOOL_KERNEL_FN void coordinator_kernel(ArenaView arena, FfnAgentControlView control) {
  const auto block = cooperative_groups::this_thread_block();
  if (block.thread_rank() == 0) {
    cuda::atomic_ref{*control.activation_count}.fetch_add(std::uint32_t{1}, cuda::memory_order_release);
  }
  block.sync();
  coordinator_loop(arena, block, *control.coordinator_scheduler);
}

} // namespace

void launch_coordinator(ArenaView arena, const ArenaLayout &layout, FfnAgentControlView control, cudaStream_t stream) {
  TORCH_CHECK(nvshmem_my_pe() == layout.coordinator_pe(), "xpool Fabric Coordinator requires the Coordinator PE");
  const auto config = cuda::make_config(cuda::block_dims<kBlockSize>(), cuda::grid_dims(1));
  cuda::launch(cuda::stream_ref{stream}, config, coordinator_kernel, arena, control);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xpool::fabric
