#include <xpool/fabric/atnagent.cuh>

#include <cstddef>
#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cooperative_groups.h>
#include <cuda/launch>
#include <cuda_runtime_api.h>

#include <xpool/fabric/protocol.cuh>
#include <xpool/ffn.hpp>
#include <xpool/hooks.cuh>
#include <xpool/macros.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/atnagent.hpp>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::transport {

namespace {

constexpr auto kBlockSize = 256U;

XPOOL_DEVICE_FN bool publish_fabric_failure(const ArenaView &transport_arena,
                                            const xpool::fabric::ArenaView &fabric_arena) {
  if (fabric_arena.empty() || !fabric_arena.state().failure.published()) {
    return false;
  }
  const auto result = fabric_arena.state().failure.payload.result_code;
  const auto mailbox_status = transport_arena.mailbox().observe();
  if (mailbox_status == MailboxStatus::Evaluated &&
      transport_arena.mailbox().result_code == xpool::ffn::ResultCode::Ok) {
    return false;
  }
  transport_arena.publish_generation_failure(result);
  return true;
}

// Validate the sole published mailbox request against immutable Transport
// geometry and the presence rules for physical DP-rank rows.
XPOOL_DEVICE_FN bool request_valid(const ArenaView &arena) {
  const auto &mailbox = arena.mailbox();
  const auto &request = mailbox.request;
  const auto dp_rank_payload_rows_present = !arena.dp_rank_payload_rows().empty();
  return request.valid() && mailbox.payload_rows != 0 && mailbox.payload_rows <= arena.layout().payload_row_capacity &&
         ((arena.layout().atn_dp_size == 1 && request.dp_row_layout == xpool::ffn::DpRowLayout::None &&
           !dp_rank_payload_rows_present) ||
          (arena.layout().atn_dp_size > 1 && dp_rank_payload_rows_present &&
           (request.dp_row_layout == xpool::ffn::DpRowLayout::UniformByRank ||
            request.dp_row_layout == xpool::ffn::DpRowLayout::PackedByRank)));
}

// Validate and dispatch one published mailbox request to Fabric.
XPOOL_DEVICE_FN xpool::ffn::ResultCode execute_request(const cooperative_groups::thread_block &group,
                                                       const ArenaView &transport_arena,
                                                       const xpool::fabric::ArenaView &fabric_arena) {
  if (!request_valid(transport_arena)) {
    const auto result = xpool::ffn::ResultCode::ProtocolMismatch;
    if (group.thread_rank() == 0) {
      transport_arena.mailbox().publish_result(result);
      xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
          {.arena = transport_arena,
           .kind = xpool::hooks::TransportProtocolEventKind::ResultPublished,
           .result_code = result});
    }
    group.sync();
    return result;
  }

  if (group.thread_rank() == 0) {
    xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
        {.arena = transport_arena, .kind = xpool::hooks::TransportProtocolEventKind::ExecutionStarted});
  }
  group.sync();
  const auto result = xpool::fabric::atnagent::execute(fabric_arena, transport_arena);
  if (group.thread_rank() == 0) {
    xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
        {.arena = transport_arena, .kind = xpool::hooks::TransportProtocolEventKind::ExecutionCompleted});
    transport_arena.mailbox().publish_result(result);
    xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
        {.arena = transport_arena,
         .kind = xpool::hooks::TransportProtocolEventKind::ResultPublished,
         .result_code = result});
  }
  group.sync();
  return result;
}

// Drive one mailbox to Closed during Resident drain, publishing a terminal
// result for a request that was already visible to the Resident.
XPOOL_DEVICE_FN void finish_terminal_mailbox(const cooperative_groups::thread_block &group, const ArenaView &arena) {
  if (group.thread_rank() == 0) {
    while (true) {
      const auto status = arena.mailbox().observe();
      if (status == MailboxStatus::Idle) {
        if (arena.mailbox().try_close_idle()) {
          break;
        }
        continue;
      }
      if (status == MailboxStatus::Published) {
        xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
            {.arena = arena,
             .kind = xpool::hooks::TransportProtocolEventKind::RequestObserved,
             .payload_rows = arena.mailbox().payload_rows,
             .request = &arena.mailbox().request});
        const auto failure = arena.generation_failure();
        const auto result = failure != xpool::ffn::ResultCode::Ok ? failure : xpool::ffn::ResultCode::Shutdown;
        arena.mailbox().publish_result(result);
        xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
            {.arena = arena, .kind = xpool::hooks::TransportProtocolEventKind::ResultPublished, .result_code = result});
        break;
      }
      xpool::abort_if(status == MailboxStatus::Dormant);
      break;
    }
  }
  group.sync();
}

// Run one cooperative Transport Resident grid with exactly one block owning
// each rank-local arena mailbox and one process-wide drain state.
XPOOL_KERNEL_FN void transport_resident_kernel(cuda::std::span<const ArenaView> arenas,
                                               xpool::fabric::ArenaView fabric_arena, ResidentState *resident_state) {
  const auto grid = cooperative_groups::this_grid();
  const auto group = cooperative_groups::this_thread_block();
  const auto arena = arenas[grid.block_rank()];
  XPOOL_DEVICE_SHARED bool stop;
  XPOOL_DEVICE_SHARED MailboxStatus status;

  // Phase: Startup - Every arena mailbox becomes visible as Idle before any block may
  // observe or execute a request.
  grid.sync();
  if (group.thread_rank() == 0) {
    arena.mailbox().open();
  }
  grid.sync();

  // Phase: Service - Each block exclusively services one arena until process-local
  // drain or a terminal mailbox state is observed.
  while (true) {
    if (group.thread_rank() == 0) {
      if (publish_fabric_failure(arena, fabric_arena)) {
        resident_state->request_drain();
      }
      stop = resident_state->draining();
      status = arena.mailbox().observe();
    }
    group.sync();

    if (stop) {
      if (group.thread_rank() == 0) {
        arena.publish_shutdown();
      }
      group.sync();
      finish_terminal_mailbox(group, arena);
      break;
    }

    if (status == MailboxStatus::Closed) {
      if (group.thread_rank() == 0) {
        resident_state->request_drain();
      }
      group.sync();
      continue;
    }
    if (status != MailboxStatus::Published) {
      if (group.thread_rank() == 0) {
        xpool::utils::wait::relax();
      }
      continue;
    }

    if (group.thread_rank() == 0) {
      xpool::hooks::TransportAtnAgentProtocolEvent::hooks(
          {.arena = arena,
           .kind = xpool::hooks::TransportProtocolEventKind::RequestObserved,
           .payload_rows = arena.mailbox().payload_rows,
           .request = &arena.mailbox().request});
    }
    group.sync();
    const auto result = execute_request(group, arena, fabric_arena);
    if (publish_fabric_failure(arena, fabric_arena) || result == xpool::ffn::ResultCode::ProtocolMismatch ||
        result == xpool::ffn::ResultCode::Timeout) {
      if (group.thread_rank() == 0) {
        resident_state->request_drain();
      }
      group.sync();
    }
  }

  // Phase: Drain - All arena blocks converge before the cooperative Resident exits.
  grid.sync();
}

} // namespace

void launch_resident_kernel(cuda::std::span<const ArenaView> arenas, xpool::fabric::ArenaView fabric_arena,
                            ResidentState *state, cudaStream_t stream) {
  auto device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  auto cooperative_launch = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative_launch != 0, "xpool Transport Resident requires cooperative-launch support");

  auto blocks_per_multiprocessor = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, transport_resident_kernel,
                                                               kBlockSize, 0));
  auto multiprocessor_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  const auto resident_capacity =
      static_cast<std::size_t>(blocks_per_multiprocessor) * static_cast<std::size_t>(multiprocessor_count);
  TORCH_CHECK(arenas.size() <= resident_capacity, "xpool Transport Resident requires ", arenas.size(),
              " concurrently resident blocks but the device supports ", resident_capacity);

  const auto config =
      cuda::make_config(cuda::block_dims<kBlockSize>(), cuda::grid_dims(arenas.size()), cuda::cooperative_launch{});
  cuda::launch(cuda::stream_ref{stream}, config, transport_resident_kernel, arenas, fabric_arena, state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xpool::transport
