#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cooperative_groups.h>
#include <cuda/launch>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/atomic.cuh>
#include <xpool/debug/loopback.cuh>
#include <xpool/debug/options.cuh>
#include <xpool/fabric/atnagent.cuh>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/atnagent.hpp>
#include <xpool/transport/protocol.cuh>
#include <xpool/utils/wait.cuh>

namespace xpool::transport {

namespace {

constexpr int kResidentBlockSize = 32;

using ResidentBlock = cooperative_groups::thread_block_tile<kResidentBlockSize>;

XPOOL_DEVICE_FN bool publish_fabric_failure(
    const TransportArenaView &transport_arena,
    const xpool::fabric::FabricArenaView &fabric_arena) {
  if (fabric_arena.empty() || !fabric_arena.state().failure.published()) {
    return false;
  }
  const auto result = xpool::abi::FfnResultCode{
      fabric_arena.state().failure.payload.result_code};
  transport_arena.publish_generation_failure(result);
  return true;
}

/// Validate the sole published mailbox request against immutable Transport
/// geometry and the presence rules for DP token counts.
XPOOL_DEVICE_FN bool request_valid(const TransportArenaView &arena) {
  const auto &mailbox = arena.mailbox();
  const auto &request = mailbox.request;
  const auto token_counts_present = arena.dp_token_counts() != nullptr;
  return request.valid() && mailbox.payload_rows != 0 &&
         mailbox.payload_rows <= arena.layout().max_tokens &&
         ((arena.layout().atn_dp_size == 1 &&
           request.dp_padding_mode == xpool::abi::DpPaddingMode::None &&
           !token_counts_present) ||
          (arena.layout().atn_dp_size > 1 && token_counts_present &&
           (request.dp_padding_mode == xpool::abi::DpPaddingMode::MaxLen ||
            request.dp_padding_mode == xpool::abi::DpPaddingMode::SumLen)));
}

/// Execute the AtnAgent-local debug transform and publish its terminal mailbox
/// result with block-wide payload completion.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode execute_local_loopback(
    const ResidentBlock &group, const TransportArenaView &arena,
    TransportTraceRecord *trace) {
  auto result = xpool::abi::FfnResultCode{
      xpool::abi::FfnResultCode::NotImplemented};
  if (arena.layout().hidden_size % 2 == 0) {
    if (group.thread_rank() == 0 && trace != nullptr) {
      trace->execution_admitted();
    }
    xpool::debug::rotate_hidden_pairs(
        cooperative_groups::this_thread_block(), arena.output_payload(),
        arena.input_payload(), xpool::abi::TensorDType{arena.layout().dtype},
        (arena.mailbox().payload_rows * arena.layout().hidden_size) /
            std::size_t{2});
    result = xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok};
  }
  group.sync();
  if (group.thread_rank() == 0) {
    if (trace != nullptr) {
      trace->execution_completed();
      trace->evaluated(result);
    }
    arena.mailbox().publish_result(result);
  }
  group.sync();
  return result;
}

/// Validate and dispatch one published mailbox request to either the explicit
/// debug site or the production Fabric path.
XPOOL_DEVICE_FN xpool::abi::FfnResultCode execute_request(
    const ResidentBlock &group, const TransportArenaView &transport_arena,
    const xpool::fabric::FabricArenaView &fabric_arena,
    TransportTraceRecord *trace) {
  if (!request_valid(transport_arena)) {
    const auto result = xpool::abi::FfnResultCode{
        xpool::abi::FfnResultCode::ProtocolMismatch};
    if (group.thread_rank() == 0) {
      if (trace != nullptr) {
        trace->evaluated(result);
      }
      transport_arena.mailbox().publish_result(result);
    }
    group.sync();
    return result;
  }

  if (group.thread_rank() == 0 && trace != nullptr) {
    trace->execution_started();
  }
  group.sync();
  if (xpool::debug::options().loopback.site ==
      xpool::debug::LoopbackSite::AtnAgent) {
    return execute_local_loopback(group, transport_arena, trace);
  }
  return xpool::fabric::atnagent::execute(fabric_arena, transport_arena, trace);
}

/// Drive one mailbox to Closed during Resident drain, publishing a terminal
/// result for a request that was already visible to the Resident.
XPOOL_DEVICE_FN void finish_terminal_mailbox(
    const ResidentBlock &group, const TransportArenaView &arena) {
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
        auto *trace = arena.current_trace();
        if (trace != nullptr) {
          trace->published_observed();
        }
        const auto failure = arena.generation_failure();
        const auto result =
            failure != xpool::abi::FfnResultCode::Ok
                ? failure
                : xpool::abi::FfnResultCode{
                      xpool::abi::FfnResultCode::Shutdown};
        if (trace != nullptr) {
          trace->evaluated(result);
        }
        arena.mailbox().publish_result(result);
        break;
      }
      xpool::abort_if(status == MailboxStatus::Dormant);
      break;
    }
  }
  group.sync();
}

/// Run one cooperative Transport Resident grid with exactly one block owning
/// each rank-local arena mailbox and one process-wide drain state.
__global__ void transport_resident_kernel(
    const TransportArenaView *arenas, std::size_t arena_count,
    xpool::fabric::FabricArenaView fabric_arena,
    TransportResidentState *resident_state) {
  const auto grid = cooperative_groups::this_grid();
  const auto group = cooperative_groups::tiled_partition<kResidentBlockSize>(
      cooperative_groups::this_thread_block());
  xpool::abort_if(arenas == nullptr || resident_state == nullptr ||
                  blockIdx.x >= arena_count);
  const auto arena = arenas[blockIdx.x];

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
    auto stop = std::uint32_t{0};
    auto status_value = std::uint32_t{0};
    if (group.thread_rank() == 0) {
      if (publish_fabric_failure(arena, fabric_arena)) {
        resident_state->request_drain();
      }
      stop = resident_state->draining() ? 1U : 0U;
      status_value = static_cast<std::uint32_t>(arena.mailbox().observe());
    }
    stop = group.shfl(stop, 0);
    status_value = group.shfl(status_value, 0);
    const auto status = static_cast<MailboxStatus>(status_value);

    if (stop != 0U) {
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

    auto *trace = arena.current_trace();
    if (group.thread_rank() == 0 && trace != nullptr) {
      trace->published_observed();
    }
    group.sync();
    const auto result = execute_request(group, arena, fabric_arena, trace);
    if (publish_fabric_failure(arena, fabric_arena) ||
        result == xpool::abi::FfnResultCode::ProtocolMismatch ||
        result == xpool::abi::FfnResultCode::Timeout ||
        result == xpool::abi::FfnResultCode::NotImplemented) {
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

void launch_transport_resident_kernel(
    const TransportArenaView *arenas, std::size_t arena_count,
    xpool::fabric::FabricArenaView fabric_arena,
    TransportResidentState *state, cudaStream_t stream) {
  TORCH_CHECK(arenas != nullptr && arena_count != 0,
              "xpool Transport Resident requires a non-empty device arena array");
  TORCH_CHECK(state != nullptr,
              "xpool Transport Resident requires process-local control state");
  TORCH_CHECK(stream != nullptr,
              "xpool Transport Resident requires a CUDA stream");

  auto device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  auto cooperative_launch = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &cooperative_launch, cudaDevAttrCooperativeLaunch, device));
  TORCH_CHECK(cooperative_launch != 0,
              "xpool Transport Resident requires cooperative-launch support");

  auto blocks_per_multiprocessor = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_multiprocessor, transport_resident_kernel,
      kResidentBlockSize, 0));
  auto multiprocessor_count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  const auto resident_capacity =
      static_cast<std::size_t>(blocks_per_multiprocessor) *
      static_cast<std::size_t>(multiprocessor_count);
  TORCH_CHECK(arena_count <= resident_capacity,
              "xpool Transport Resident requires ", arena_count,
              " concurrently resident blocks but the device supports ",
              resident_capacity);

  const auto config = cuda::make_config(
      cuda::block_dims<kResidentBlockSize>(), cuda::grid_dims(arena_count),
      cuda::cooperative_launch{});
  cuda::launch(cuda::stream_ref{stream}, config, transport_resident_kernel,
               arenas, arena_count, fabric_arena, state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace xpool::transport
